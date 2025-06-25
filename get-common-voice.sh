#!/bin/bash

# Usage:
#   ./download_common_voice.sh "<signed_url>" [output_filename]

# Setup
CACHE_DIR="$HOME/.cache/commonvoice"
mkdir -p "$CACHE_DIR"

# Check for aria2c
if ! command -v aria2c &> /dev/null; then
    echo "[INFO] aria2c not found. Installing..."
    sudo apt update && sudo apt install -y aria2 || {
        echo "[ERROR] Failed to install aria2. Exiting."
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

aria2c --continue=true -x 16 -s 16 -k 1M -o "$BASENAME" "$SIGNED_URL"
tar -xvzf cv-corpus-22.0-2025-06-20-fa.tar.gz
