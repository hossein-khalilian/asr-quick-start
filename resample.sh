#!/bin/bash

# Set source and destination directories
SRC_DIR="audio_files"
DST_DIR="resampled_audio"

# Create destination directory if it doesn't exist
mkdir -p "$DST_DIR"

# Loop through all mp3 files in the source directory
for file in "$SRC_DIR"/*.mp3; do
    # Get the base filename without extension
    filename=$(basename "$file" .mp3)
    
    # Define output file path
    output="$DST_DIR/${filename}.wav"
    
    # Resample using ffmpeg
    ffmpeg -y -i "$file" -ar 16000 "$output"
done

echo "Resampling complete. WAV files saved to '$DST_DIR'"

