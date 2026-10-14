#!/usr/bin/env bash
# Download all DSPredict full datasets from Kaggle into data/data/dspredict/
# Run from the DSGym repo root.
#
# Prerequisites:
#   pip install kaggle
#   Place your Kaggle API access token at ~/.kaggle/access_token
#   (Download from https://www.kaggle.com/settings → "API" → "Create New Token")
#
# Note: Many competitions require you to accept their rules on Kaggle before
#       the API will allow downloading. Visit each competition page and click
#       "I Understand and Accept" if a download fails with a 403 error.

set -euo pipefail
shopt -s nullglob

DEST="$(dirname "$0")/data/data/dspredict"
mkdir -p "$DEST"

# Check uv is available to run the Kaggle CLI
if ! command -v uv &>/dev/null; then
    echo "[ERROR] uv not found. Install uv before running the Kaggle downloader."
    exit 1
fi

dl() {
    local name="$1"
    local dir="$DEST/$name"
    mkdir -p "$dir"

    # Skip only populated directories without a leftover archive from a failed extraction.
    local file_count
    file_count=$(
        find "$dir" -maxdepth 2 -type f \
            ! -name '.gitkeep' ! -name '.download_failed' ! -name '*.zip' |
            wc -l | tr -d ' '
    )
    local archives=("$dir"/*.zip)
    if [[ "$file_count" -gt 0 && "${#archives[@]}" -eq 0 ]]; then
        echo "[SKIP] $name (already has $file_count file(s))"
        return
    fi

    if [[ "${#archives[@]}" -eq 0 ]]; then
        echo "[DL]   $name"
        if ! uv run kaggle competitions download -c "$name" -p "$dir" --quiet 2>&1; then
            echo "[WARN] $name — download failed (check competition rules acceptance or availability)"
            return
        fi
        archives=("$dir"/*.zip)
    fi

    if [[ "${#archives[@]}" -eq 0 ]]; then
        echo "[ERROR] $name — Kaggle download produced no ZIP archive" >&2
        return 1
    fi

    # Remove files left by a previously interrupted extraction before retrying.
    find "$dir" -mindepth 1 -maxdepth 1 ! -name '*.zip' -exec rm -rf -- {} +
    for zip in "${archives[@]}"; do
        echo "[EXT]  $(basename "$zip")"
        if ! unzip -tq "$zip" || ! unzip -q -o "$zip" -d "$dir" </dev/null; then
            echo "[ERROR] $name — extraction failed; check free disk space and archive integrity." >&2
            df -h "$dir" >&2 || true
            find "$dir" -mindepth 1 -maxdepth 1 ! -name '*.zip' -exec rm -rf -- {} +
            return 1
        fi
    done
    rm -f -- "${archives[@]}"

    local final_count
    final_count=$(find "$dir" -maxdepth 2 -type f | wc -l | tr -d ' ')
    echo "[OK]   $name ($final_count file(s))"
}

# ── Tabular / structured ──────────────────────────────────────────────────────
dl "ashrae-energy-prediction"
dl "career-con-2019"
dl "champs-scalar-coupling"
dl "data-science-bowl-2018"
dl "digit-recognizer"
dl "elo-merchant-category-recommendation"
dl "home-credit-default-risk"
dl "home-data-for-ml-course"
dl "ieee-fraud-detection"
dl "LANL-Earthquake-Prediction"
dl "liverpool-ion-switching"
dl "m5-forecasting-accuracy"
dl "m5-forecasting-uncertainty"
dl "march-machine-learning-mania-2023"
dl "march-machine-learning-mania-2025"
dl "mens-machine-learning-competition-2018"
dl "mens-machine-learning-competition-2019"
dl "mens-march-mania-2022"
dl "microsoft-malware-prediction"
dl "novozymes-enzyme-stability-prediction"
dl "open-problems-single-cell-perturbations"
dl "otto-recommender-system"
dl "predict-ai-model-runtime"
dl "recruit-restaurant-visitor-forecasting"
dl "santander-customer-transaction-prediction"
dl "santander-value-prediction-challenge"
dl "spaceship-titanic"
dl "store-sales-time-series-forecasting"
dl "talkingdata-adtracking-fraud-detection"
dl "ventilator-pressure-prediction"
dl "vsb-power-line-fault-detection"
dl "web-traffic-time-series-forecasting"
dl "womens-machine-learning-competition-2019"

# ── NLP ───────────────────────────────────────────────────────────────────────
dl "gendered-pronoun-resolution"
dl "nlp-getting-started"
dl "stanford-covid-vaccine"
dl "tensorflow-speech-recognition-challenge"
dl "trec-covid-information-retrieval"

# ── Computer vision ───────────────────────────────────────────────────────────
dl "humpback-whale-identification"
dl "imaterialist-challenge-fashion-2018"
dl "imaterialist-challenge-furniture-2018"
dl "inclusive-images-challenge"
dl "planttraits2024"
dl "rsna-pneumonia-detection-challenge"
dl "siim-acr-pneumothorax-segmentation"
dl "sp-society-camera-model-identification"
dl "statoil-iceberg-classifier-challenge"
dl "tgs-salt-identification-challenge"
dl "understanding_cloud_organization"

# ── Geo / spatial ─────────────────────────────────────────────────────────────
dl "geolifeclef-2024"
dl "google-smartphone-decimeter-challenge"
dl "pku-autonomous-driving"

# ── Video understanding ───────────────────────────────────────────────────────
# youtube8m datasets are very large; only frame-level features are needed.
# The Kaggle CLI downloads them as a set of TFRecord shards.
dl "youtube8m-2018"
dl "youtube8m-2019"

echo ""
echo "Done. Summary of data/data/dspredict/:"
for d in "$DEST"/*/; do
    name=$(basename "$d")
    count=$(find "$d" -maxdepth 2 -type f | wc -l | tr -d ' ')
    if [[ "$count" -eq 0 ]]; then
        echo "  [EMPTY] $name"
    else
        echo "  [OK   ] $name  ($count file(s))"
    fi
done
