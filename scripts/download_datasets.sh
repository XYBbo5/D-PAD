#!/bin/bash

# Get the directory of the script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Create the datasets directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/../datasets/long"

# Define URLs for datasets
ETTh1_URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh1.csv"
ETTh2_URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh2.csv"
ETTm1_URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm1.csv"
ETTm2_URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTm2.csv"
GFOLDER_ID="1ohGYWWohJlOlb2gsGTeEq3Wii2egnEPR"


# Download ETT-small datasets
echo "Downloading datasets..."
wget -O "$SCRIPT_DIR/../datasets/long/ETTh1.csv" "$ETTh1_URL" || exit_status=1
wget -O "$SCRIPT_DIR/../datasets/long/ETTh2.csv" "$ETTh2_URL" || exit_status=1
wget -O "$SCRIPT_DIR/../datasets/long/ETTm1.csv" "$ETTm1_URL" || exit_status=1
wget -O "$SCRIPT_DIR/../datasets/long/ETTm2.csv" "$ETTm2_URL" || exit_status=1

# Check if gdown is installed
if command -v gdown &> /dev/null; then
# Download all files from the folder using gdown
  gdown --folder --id "$GFOLDER_ID" -O "$SCRIPT_DIR/../datasets/long/" || exit_status=1
  TIME_SERIES_DIR="$SCRIPT_DIR/../datasets/long/TimeSeriesData"
  if [ -d "$TIME_SERIES_DIR" ]; then
    mv "$TIME_SERIES_DIR"/* "$SCRIPT_DIR/../datasets/long/"
    rmdir "$TIME_SERIES_DIR"
  fi
  # rename WTH.csv to weather.csv
  if [ -f "$SCRIPT_DIR/../datasets/long/WTH.csv" ]; then
    mv "$SCRIPT_DIR/../datasets/long/WTH.csv" "$SCRIPT_DIR/../datasets/long/weather.csv"
  fi
else
  echo "Error: gdown is not installed. Please install gdown by running 'pip install gdown'."
  echo "Skipping download of ECL and Weather datasets"
fi

# Check if all files were downloaded successfully
if [ -z "$exit_status" ]; then
  echo "Datasets downloaded successfully."
else
  echo "Error: Failed to download one or more datasets."
  exit 1
fi