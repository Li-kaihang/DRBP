import torch
import torch.nn as nn
from typing import Optional

class StructureGNN(nn.Module):
    """
    基于 3D 结构的图神经网络.

    节点: 残基 (AA类型 + 二面角)
    边: Cα 距离 ≤ 10Å 的残基对 (k-NN, k=16)
    消息传递: 3 层, 每层用 MLP + residual

    相比 IPA, 显存友好 (O(L×k) vs O(L²))
    """

    def __init__(self, node_dim: int = 256, edge_dim: int = 64,
                 n_layers: int = 4, k_neighbors: int = 16,
                 dropout: float = 0.0,
                 use_dihedral: bool = True, use_frames: bool = True,
                 use_geom: bool = True, use_local: bool = True,
                 use_edge_dists: bool = True, use_edge_relseq: bool = True,
                 use_edge_orient: bool = True):
        super().__init__()
        self.node_dim = node_dim
        self.k = k_neighbors
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        self.use_dihedral = use_dihedral
        self.use_frames = use_frames
        self.use_geom = use_geom
        self.use_local = use_local

        # 节点输入维度
        node_in = 64
        if use_dihedral: node_in += 12
        if use_frames: node_in += 9 + 4
        if use_geom: node_in += 7
        if use_local: node_in += 12
        node_in += 3  # ca_exposure + ca_concavity + ca_electrostatics (表面特征)

        # 边: CA距离 RBF(32) + 序列间隔(1)
        edge_in = 33

        self.aa_embed = nn.Embedding(21, 64, padding_idx=20)
        self.node_proj = nn.Sequential(
            nn.Linear(node_in, node_dim * 2), nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim), nn.ReLU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in, edge_dim), nn.ReLU(),
            nn.Linear(edge_dim, edge_dim), nn.ReLU(),
        )

        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(node_dim, edge_dim, dropout=dropout) for _ in range(n_layers)
        ])
        self.out_dim = node_dim

    def rbf_encode(self, dists: torch.Tensor) -> torch.Tensor:
        """高斯 RBF: exp(-γ × (d - center)²)"""
        diff = dists.unsqueeze(-1) - self.rbf_centers.view(1, 1, 1, -1)  # (B, L, L, E)
        return torch.exp(-self.rbf_gamma * diff ** 2)

    def k_nearest_neighbors(self, dists: torch.Tensor, mask: torch.Tensor, k: int):
        """取每节点最近的 k 个邻居"""
        B, L = dists.shape[:2]
        if mask is not None:
            dists = dists + (1 - mask.unsqueeze(-1)) * 1e9  # mask掉padding
        _, idx = torch.topk(dists, k=min(k, L), dim=-1, largest=False)
        return idx

    def forward(self, batch: dict) -> torch.Tensor:
        """返回 (B, L, node_dim)"""
        B, L = batch['aa_indices'].shape[:2]
        mask = batch.get('mask', None)
        device = batch['aa_indices'].device

        # -- 节点特征 (按 config flag 组装) --
        node_parts = [self.aa_embed(batch['aa_indices'].long().clamp(0, 20))]
        if self.use_dihedral:
            node_parts.append(batch['dihedral_sincos'])
        if self.use_frames:
            fr = batch.get('backbone_frames_R', torch.zeros(B, L, 3, 3, device=device))
            node_parts.append(fr.reshape(B, L, 9))
            node_parts.append(batch.get('backbone_frames_quat', torch.zeros(B, L, 4, device=device)))
        if self.use_geom:
            node_parts.append(batch.get('backbone_geom', torch.zeros(B, L, 7, device=device)))
        if self.use_local:
            node_parts.append(batch.get('local_atom_coords', torch.zeros(B, L, 12, device=device)))
        node_parts.append(batch.get('ca_exposure', torch.zeros(B, L, device=device)).unsqueeze(-1))
        node_parts.append(batch.get('ca_concavity', torch.zeros(B, L, device=device)).unsqueeze(-1))
        node_parts.append(batch.get('ca_electrostatics', torch.zeros(B, L, device=device)).unsqueeze(-1))
        node = torch.cat(node_parts, dim=-1)
        node = self.drop(self.node_proj(node))

        # -- 边特征: CA距离 (GPU上直接算, O(L²)但torch.cdist极快) --
        ca_coords = batch['backbone_frames_t']  # (B, L, 3) = Cα坐标
        ca_dists = torch.cdist(ca_coords, ca_coords)  # (B, L, L)
        # RBF编码距离 + 序列间隔
        rbf = torch.exp(-(ca_dists.unsqueeze(-1) - torch.linspace(0, 30, 32, device=device).view(1,1,1,-1))**2 / 10.0)
        rel_seq = torch.arange(L, device=device).view(1, L, 1) - torch.arange(L, device=device).view(1, 1, L)
        rel_seq = rel_seq.abs().unsqueeze(-1).float() / 100.0  # (1, L, L, 1) normalized
        edge = torch.cat([rbf, rel_seq.expand(B, -1, -1, -1)], dim=-1)  # (B, L, L, 33)
        edge = self.drop(self.edge_proj(edge))

        # -- k-NN --
        knn_idx = self.k_nearest_neighbors(ca_dists, mask, self.k)

        # -- 消息传递 --
        for mp in self.mp_layers:
            node = mp(node, edge, knn_idx, mask)

        return node


class MessagePassingLayer(nn.Module):
    """单层消息传递: node ← node + MLP(node + Σ edge_weight × neighbor)"""

    def __init__(self, node_dim: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, node, edge, knn_idx, mask):
        B, L, N = node.shape
        E = edge.shape[-1]
        k = knn_idx.shape[-1]

        # 取邻居节点: (B, L, k, N) — 用 flatten 索引, 可靠
        node_flat = node.reshape(-1, N)  # (B*L, N)
        offsets = (torch.arange(B, device=node.device) * L).view(B, 1, 1)
        flat_idx = (knn_idx + offsets).reshape(-1)  # (B*L*k,)
        neighbor_nodes = node_flat[flat_idx].reshape(B, L, k, N)

        # 取邻居边: (B, L, k, E)
        edge_flat = edge.reshape(-1, E)  # (B*L*L, E)
        offsets_b = (torch.arange(B, device=node.device) * L * L).view(B, 1, 1)
        offsets_l = (torch.arange(L, device=node.device) * L).view(1, L, 1)
        flat_edge_idx = (offsets_b + offsets_l + knn_idx).reshape(-1)
        neighbor_edges = edge_flat[flat_edge_idx].reshape(B, L, k, E)

        # 注意力权重: (B, L, k, N)
        edge_weight = torch.sigmoid(self.edge_mlp(neighbor_edges))

        # 聚合邻居: (B, L, N)
        messages = (edge_weight * neighbor_nodes).sum(dim=2)
        updated = self.update(torch.cat([node, messages], dim=-1))
        node = self.norm(node + self.drop(updated))

        if mask is not None:
            node = node * mask.unsqueeze(-1)

        return node
