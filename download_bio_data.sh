#!/usr/bin/env bash
# Download all DSBio raw data files into data/data/bio/
# Run from the DSGym repo root.

set -euo pipefail

DEST="$(dirname "$0")/data/data/bio"
mkdir -p "$DEST"

dl() {
    local name="$1"
    local url="$2"
    local out="$DEST/$name"
    if [[ -f "$out" && -s "$out" ]]; then
        echo "[SKIP] $name (already exists)"
        return
    fi
    echo "[DL]   $name"
    curl -L --retry 3 --retry-delay 5 -C - -o "$out" "$url"
    echo "[OK]   $name ($(du -sh "$out" | cut -f1))"
}

# ── CellxGENE ────────────────────────────────────────────────────────────────
# Collection ff668d5d — Colon/Ileum/Rectum
dl "Rectum.h5ad" \
    "https://datasets.cellxgene.cziscience.com/da301f6b-639a-4556-980a-fc3b640bd96e.h5ad"
dl "Colon.h5ad" \
    "https://datasets.cellxgene.cziscience.com/c68ad7f8-34d4-4425-82c6-0f5dadd916dd.h5ad"
dl "Ileum.h5ad" \
    "https://datasets.cellxgene.cziscience.com/ba118601-6018-449d-9857-42c21df5b285.h5ad"

# Collection 28e9d721 — Newborn lung
dl "Single_cell_transcriptomic_profiling_identifies_molecular_phenotypes_of_newborn_human_lung_cells.h5ad" \
    "https://datasets.cellxgene.cziscience.com/5c27e357-4786-46d4-9635-cb7b37cce452.h5ad"

# Collection ec691f5f — Adult gamma delta T cells
dl "scRNA-seq_of_adult_gamma_delta_T_cells.h5ad" \
    "https://datasets.cellxgene.cziscience.com/e1511f7d-7b21-48e9-89d1-6a84c9dacf7f.h5ad"

# Collection 8f17ac63 — Vascular tissue (aorta, coronary artery, vascular tissue)
dl "slide-seqV2_analysis_of_aorta.h5ad" \
    "https://datasets.cellxgene.cziscience.com/595def56-32ed-4f00-867b-4a9ba314a88f.h5ad"
dl "slide-seqV2_analysis_of_right_coronary_artery.h5ad" \
    "https://datasets.cellxgene.cziscience.com/a248121b-dba4-4949-aa2b-b1eb7c94cfa3.h5ad"
dl "scRNA-seq_of_vascular_tissue.h5ad" \
    "https://datasets.cellxgene.cziscience.com/441fd653-78f4-4654-b3be-1b0bc79b5568.h5ad"

# Collection 77446b76 — Breast tissue
dl "snRNA-seq_analyses_of_breast_tissues_of_healthy_women_of_diverse_genetic_ancestry.h5ad" \
    "https://datasets.cellxgene.cziscience.com/bc4614b7-5ab2-4f30-8e8b-eaeeb840c154.h5ad"

# ── Zenodo 259661 — Sensory neuron QTL multi-omics ───────────────────────────
ZENODO="https://zenodo.org/api/records/259661/files"
dl "RNA_cqn_matrix.txt.gz"      "$ZENODO/RNA_cqn_matrix.txt.gz/content"
dl "RNA_gene_metadata.txt.gz"   "$ZENODO/RNA_gene_metadata.txt.gz/content"
dl "RNA_sample_metadata.txt.gz" "$ZENODO/RNA_sample_metadata.txt.gz/content"
dl "ATAC_cqn_matrix.txt.gz"     "$ZENODO/ATAC_cqn_matrix.txt.gz/content"
dl "ATAC_peak_metadata.txt.gz"  "$ZENODO/ATAC_peak_metadata.txt.gz/content"
dl "ATAC_sample_metadata.txt.gz" "$ZENODO/ATAC_sample_metadata.txt.gz/content"

# ── EBI BioStudies S-BSST16 — Sensory neuron QTL ────────────────────────────
EBI="https://www.ebi.ac.uk/biostudies/files/S-BSST16"
dl "all_basic_counts.open.txt.gz"        "$EBI/all_basic_counts.open.txt.gz"
dl "atac_consensus_peaks.txt.gz"         "$EBI/atac_consensus_peaks.txt.gz"
dl "atac_qtl.rasqual.1k.txt.gz"          "$EBI/atac_qtl.rasqual.1k.txt.gz"
dl "eqtl.fastqtl.500k.nominals.txt.gz"  "$EBI/eqtl.fastqtl.500k.nominals.txt.gz"
dl "eqtl.fastqtl.500k.permutations.txt.gz" "$EBI/eqtl.fastqtl.500k.permutations.txt.gz"
dl "eqtl.rasqual.500k.txt.gz"           "$EBI/eqtl.rasqual.500k.txt.gz"
dl "metadata.open.txt"                   "$EBI/metadata.open.txt"
dl "sqtl.fastqtl.nominals.txt.gz"       "$EBI/sqtl.fastqtl.nominals.txt.gz"
dl "sqtl.fastqtl.permutations.txt.gz"   "$EBI/sqtl.fastqtl.permutations.txt.gz"

# ── Missing: raw_adult_vd_cells_post_qc.h5ad ─────────────────────────────────
# No public data_link in dsbio.json. This file is likely derived from
# scRNA-seq_of_adult_gamma_delta_T_cells.h5ad (downloaded above) via the
# preprocessing pipeline in bio_code/pediatric_gdt_pipeline.ipynb.
# Tasks using this file will remain unrunnable until it is generated or
# obtained from the study authors.
if [[ ! -f "$DEST/raw_adult_vd_cells_post_qc.h5ad" ]]; then
    echo ""
    echo "[WARN] raw_adult_vd_cells_post_qc.h5ad — no public URL found."
    echo "       This file is a derived artifact from scRNA-seq_of_adult_gamma_delta_T_cells.h5ad."
    echo "       Generate it by running bio_code/pediatric_gdt_pipeline.ipynb, or contact the study authors."
fi

echo ""
echo "Done. Files in $DEST:"
ls -lh "$DEST"
