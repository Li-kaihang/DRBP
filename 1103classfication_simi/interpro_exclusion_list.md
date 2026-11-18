# Supplementary Resource — Nucleic-Acid-Binding Exclusion List

This list defines the InterPro domains and UniProt keywords used to identify
operational negative examples during construction of the DRBP-Net classification
dataset. Candidate negatives were retrieved from reviewed UniProtKB/Swiss-Prot
(`reviewed:true`) and excluded if they carried any of the InterPro domains or
keywords below, or if they were annotated with GO:0003677 (DNA binding) or
GO:0003723 (RNA binding).

Source: `data_pipeline/collect_negatives.py` (`DNA_BINDING_DOMAINS`,
`RNA_BINDING_DOMAINS`, `NUCLEIC_ACID_BINDING_GENERAL`, `nuc_keywords`).

## InterPro domains (75 total)

### DNA-binding domains (43)

| # | InterPro ID | Description |
|---|-------------|-------------|
| 1 | IPR000558 | Histone-like DNA-binding |
| 2 | IPR001356 | Homeobox domain |
| 3 | IPR009057 | Homeobox-like domain superfamily |
| 4 | IPR001471 | AP2/ERF domain |
| 5 | IPR013087 | Zinc finger C2H2-type |
| 6 | IPR007087 | Zinc finger C2H2 |
| 7 | IPR001005 | SANT/Myb domain |
| 8 | IPR017930 | Myb domain |
| 9 | IPR000679 | Zinc finger GATA-type |
| 10 | IPR008967 | p53-like transcription factor |
| 11 | IPR010982 | Lambda repressor-like DNA-binding |
| 12 | IPR001387 | Cro/C1-type helix-turn-helix |
| 13 | IPR013032 | EGF-like, conserved site (some bind DNA) |
| 14 | IPR009061 | Putative DNA-binding domain |
| 15 | IPR011991 | ArsR-like winged helix-turn-helix |
| 16 | IPR001845 | HTH ArsR-type |
| 17 | IPR000047 | HTH motif, lambda-like |
| 18 | IPR012287 | Homeodomain-related |
| 19 | IPR017970 | Homeobox, conserved site |
| 20 | IPR020479 | Homeodomain, metazoa |
| 21 | IPR001965 | Zinc finger, PHD-type |
| 22 | IPR019787 | Zinc finger, PHD-finger |
| 23 | IPR011016 | Zinc finger, variant |
| 24 | IPR013083 | Zinc finger, RING/FYVE/PHD-type |
| 25 | IPR001841 | Zinc finger, RING-type |
| 26 | IPR002515 | Zinc finger, C2HC5-type |
| 27 | IPR036060 | Zinc finger, C2HC5-type superfamily |
| 28 | IPR004181 | Zinc finger, MIZ-type |
| 29 | IPR015880 | Zinc finger, C2H2-like |
| 30 | IPR003604 | Zinc finger, U1-type |
| 31 | IPR013085 | Zinc finger, U1-C2H2 |
| 32 | IPR001878 | Zinc finger, CCHC-type |
| 33 | IPR036875 | Zinc finger, CCHC-type superfamily |
| 34 | IPR000315 | Zinc finger, B-box |
| 35 | IPR001876 | Zinc finger, RanBP2-type |
| 36 | IPR036443 | Zinc finger, RanBP2-type superfamily |
| 37 | IPR017907 | Zinc finger, RING-type, conserved site |
| 38 | IPR001293 | Zinc finger, TRAF-type |
| 39 | IPR008913 | Zinc finger, ABP1-type |
| 40 | IPR006600 | BTB/POZ domain (some mediate DNA binding) |
| 41 | IPR000210 | BTB/POZ domain |
| 42 | IPR011333 | BTB/POZ fold |
| 43 | IPR013069 | BTB/POZ |

### RNA-binding domains (25)

| # | InterPro ID | Description |
|---|-------------|-------------|
| 1 | IPR000504 | RRM domain (RNA recognition motif) |
| 2 | IPR012677 | Nucleotide-binding alpha-beta plait domain superfamily |
| 3 | IPR003954 | RNA recognition motif domain, eukaryote |
| 4 | IPR035979 | RNA-binding domain superfamily |
| 5 | IPR002344 | Lupus La protein |
| 6 | IPR006535 | K Homology domain, type 1 |
| 7 | IPR004087 | K Homology domain |
| 8 | IPR004088 | K Homology domain, type 2 |
| 9 | IPR001159 | Double-stranded RNA binding domain |
| 10 | IPR014720 | Double-stranded RNA-binding domain |
| 11 | IPR003029 | S1 domain (ribosomal, RNA-binding) |
| 12 | IPR012340 | Nucleic acid-binding, OB-fold |
| 13 | IPR022967 | S1 domain, RNA-binding |
| 14 | IPR011129 | Cold-shock protein, OB-fold |
| 15 | IPR002059 | Cold-shock protein, DNA-binding-like |
| 16 | IPR019844 | Cold-shock conserved site |
| 17 | IPR012162 | Poly(A)-specific ribonuclease PARN |
| 18 | IPR034397 | RNA-binding, YTH domain |
| 19 | IPR007275 | YTH domain |
| 20 | IPR002713 | FF domain (some RNA-related) |
| 21 | IPR001660 | Sterile alpha motif (SAM) domain (some RNA-binding) |
| 22 | IPR035437 | SNase-like, OB-fold |
| 23 | IPR016027 | Nucleic acid-binding, OB-fold-like |
| 24 | IPR008994 | Nucleic acid-binding, OB-fold |
| 25 | IPR013870 | mRNA (N6-adenosine)-methyltransferase |

### General nucleic-acid-binding domains (7)

| # | InterPro ID | Description |
|---|-------------|-------------|
| 1 | IPR009072 | Histone-fold (general DNA binding) |
| 2 | IPR007125 | Histone H2A/H2B/H3/H4 |
| 3 | IPR002119 | Histone H2A |
| 4 | IPR000164 | Histone H3 |
| 5 | IPR001472 | Bipartite nuclear localization signal |
| 6 | IPR032042 | Helix-turn-helix DNA-binding domain |
| 7 | IPR039702 | Helix-turn-helix domain |

## UniProt keywords excluded (19)

DNA-binding, RNA-binding, Transcription, Helicase, Topoisomerase, Nuclease,
Polymerase, Primase, Ligase, Ribosomal protein, Spliceosome, Chromatin,
Nucleosome, Telomere, Reverse transcriptase, Integrase, Transposase, Repressor,
Activator.

## Query rules

The negative query against the UniProtKB REST API combined the following
conditions with `AND`:

```
(reviewed:true)
AND NOT go:0003677
AND NOT go:0003723
AND NOT (xref:interpro-IPR000558 OR xref:interpro-IPR001356 OR ...)
AND NOT keyword:"DNA-binding" AND NOT keyword:"RNA-binding" AND ...
```

where the InterPro exclusion in the live query used a core subset of 25 domains
(`key_domains`: 12 DNA + 12 RNA + IPR009072). The full 75-domain list above is
defined in the source script (`ALL_EXCLUDE_DOMAINS`) but was **not** applied in
the negative-selection query. Absence of an annotation was treated as an
operational negative, not as experimental evidence of non-binding.
