#!/bin/bash

# Directories
SRC_DIR="audio_files"
DST_DIR="resampled_audio"

# Ensure destination exists
mkdir -p "$DST_DIR"

# Find all .mp3 files
find "$SRC_DIR" -type f -iname "*.mp3" | sort > mp3_list.txt

# Total count
TOTAL=$(wc -l < mp3_list.txt)

echo "Found $TOTAL MP3 files. Starting resampling..."

# Export function for GNU parallel
resample() {
    infile="$1"
    rel_path="${infile#$SRC_DIR/}"
    out_path="${DST_DIR}/${rel_path%.mp3}.wav"

    # Create subdir if needed
    mkdir -p "$(dirname "$out_path")"

    # Skip if already exists
    if [[ -f "$out_path" ]]; then
        return
    fi

    ffmpeg -loglevel error -y -i "$infile" -ar 16000 "$out_path"
}
export -f resample
export SRC_DIR DST_DIR

# Use parallel with progress
cat mp3_list.txt | parallel --bar -j"$(nproc)" resample {}

echo "✅ Resampling completed. Output saved in $DST_DIR"

