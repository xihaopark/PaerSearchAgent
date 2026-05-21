# Example Task: edgeR-Style Normalization

Apply a count-based expression-analysis normalization method to RNA-seq count
data for downstream analysis.

Expected output shape:

- sample normalization factors
- normalized CPM values
- a short normalization report

This example is intentionally sparse: the agent should infer that an R/Bioconductor
workflow such as edgeR `DGEList -> calcNormFactors -> cpm` is relevant.

