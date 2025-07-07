#!/bin/bash

# Usage:
#   ./download_common_voice.sh "<signed_url>" [output_filename]

# Setup
CACHE_DIR="$HOME/.cache/commonvoice"
mkdir -p "$CACHE_DIR"

# Check for axel
if ! command -v axel &> /dev/null; then
    echo "[INFO] axel not found. Installing..."
    sudo apt update && sudo apt install -y axel || {
        echo "[ERROR] Failed to install axel. Exiting."
        exit 1
    }
fi

# Check input
if [ -z "$1" ]; then
    echo "Usage: $0 \"<signed_url>\" [output_filename]"
    exit 1
fi

SIGNED_URL="$1"

# Determine output filename
if [ -z "$2" ]; then
    BASENAME=$(basename "$(echo "$SIGNED_URL" | cut -d'?' -f1)")
else
    BASENAME="$2"
fi

OUTPUT_PATH="$CACHE_DIR/$BASENAME"

# Start download
echo "[INFO] Downloading to: $OUTPUT_PATH"
cd "$CACHE_DIR" || {
    echo "[ERROR] Failed to change directory to $CACHE_DIR"
    exit 1
}

# Use axel with 10 connections
axel -n 4 -o "$BASENAME" "$SIGNED_URL"
DOWNLOAD_EXIT_CODE=$?

# Check if axel exited successfully and file exists
if [ $DOWNLOAD_EXIT_CODE -ne 0 ] || [ ! -f "$OUTPUT_PATH" ]; then
    echo "[ERROR] Download failed or was interrupted. Exiting."
    exit 1
fi 

# Extract the downloaded tar.gz
tar -xvzf "$BASENAME"

echo "[INFO] Preparing dataset using Python script"
cd -
python3 prepare_common_voice.py
