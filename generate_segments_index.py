#!/usr/bin/env python3
"""
Generate an index file for WAV files in the segments directory
Similar format to index.txt but for segmented audio files
"""

import os
import json
import librosa
from pathlib import Path

def get_audio_duration(audio_file):
    """Get the duration of an audio file"""
    try:
        duration = librosa.get_duration(path=audio_file)
        return duration
    except Exception as e:
        print(f"Error getting duration for {audio_file}: {e}")
        return 0.0

def generate_segments_index(segments_dir="segments", output_file="segments_index.txt"):
    """Generate index file for segments directory"""
    
    # Get all WAV files in segments directory
    segments_path = Path(segments_dir)
    if not segments_path.exists():
        print(f"Error: Directory '{segments_dir}' does not exist!")
        return
    
    wav_files = list(segments_path.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in '{segments_dir}' directory!")
        return
    
    # Sort files by name for consistent ordering
    wav_files.sort()
    
    print(f"Found {len(wav_files)} WAV files in '{segments_dir}' directory")
    print("Calculating durations...")
    
    # Generate index entries
    index_entries = []
    total_duration = 0.0
    
    for wav_file in wav_files:
        print(f"Processing: {wav_file.name}")
        
        # Get absolute path
        abs_path = str(wav_file.absolute())
        
        # Get duration
        duration = get_audio_duration(str(wav_file))
        total_duration += duration
        
        # Create index entry (same format as index.txt)
        entry = {
            "audio_filepath": abs_path,
            "duration": round(duration, 2)
        }
        
        index_entries.append(entry)
    
    # Write index file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in index_entries:
            json_line = json.dumps(entry, separators=(',', ':'))
            f.write(json_line + '\n')
    
    print(f"\n✅ Index file generated: {output_file}")
    print(f"Total files: {len(index_entries)}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")
    
    # Show statistics
    durations = [entry['duration'] for entry in index_entries]
    if durations:
        print(f"\nDuration statistics:")
        print(f"  Min: {min(durations):.2f}s")
        print(f"  Max: {max(durations):.2f}s")
        print(f"  Average: {sum(durations)/len(durations):.2f}s")
    
    # Show first few entries as preview
    print(f"\nFirst 5 entries preview:")
    for i, entry in enumerate(index_entries[:5]):
        filename = Path(entry['audio_filepath']).name
        print(f"  {filename}: {entry['duration']}s")
    
    if len(index_entries) > 5:
        print(f"  ... and {len(index_entries) - 5} more")

if __name__ == "__main__":
    generate_segments_index()
