import torch
import torchaudio
import os
import math
import re
from scipy.io.wavfile import write

# Load Silero VAD
torch.set_num_threads(1)
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


def parse_srt(srt_path):
    """Parse SRT file and return list of subtitle entries with start/end times in seconds"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entries = []
    # Split by double newlines to separate entries
    blocks = re.split(r'\n\s*\n', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        # Parse timestamp line (format: HH:MM:SS,mmm --> HH:MM:SS,mmm)
        timestamp_line = lines[1]
        time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp_line)
        
        if time_match:
            start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = map(int, time_match.groups())
            start_time = start_h * 3600 + start_m * 60 + start_s + start_ms / 1000
            end_time = end_h * 3600 + end_m * 60 + end_s + end_ms / 1000
            
            # Join text lines (everything after timestamp)
            text = ' '.join(lines[2:]).strip()
            
            entries.append({
                'start': start_time,
                'end': end_time,
                'text': text
            })
    
    return entries


def get_text_for_segment(srt_entries, start_sec, end_sec):
    """Extract text from SRT entries that overlap with the given time range"""
    segment_text = []
    
    for entry in srt_entries:
        # Check if there's any overlap between segment time range and subtitle time range
        if entry['end'] >= start_sec and entry['start'] <= end_sec:
            segment_text.append(entry['text'])
    
    return ' '.join(segment_text)


def cut_wav_into_segments(wav_path, srt_path=None, out_dir="segments", max_segment_sec=40, sampling_rate=16000):
    # Load audio (resampled to 16kHz)
    wav = read_audio(wav_path, sampling_rate=sampling_rate)

    # Parse SRT file if provided
    srt_entries = None
    if srt_path and os.path.exists(srt_path):
        srt_entries = parse_srt(srt_path)
        print(f"Loaded {len(srt_entries)} subtitle entries from {srt_path}")

    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sampling_rate)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    segments = []
    chunk_times = []  # Store (start, end) times for each chunk
    current_chunk = []
    current_length = 0
    chunk_start = None

    for seg in speech_timestamps:
        seg_len = (seg['end'] - seg['start']) / sampling_rate
        if current_length == 0:
            chunk_start = seg['start']
        if current_length + seg_len > max_segment_sec and current_chunk:
            # save current chunk
            chunk_tensor = collect_chunks(current_chunk, wav)
            segments.append(chunk_tensor)
            chunk_end = current_chunk[-1]['end']
            chunk_times.append((chunk_start, chunk_end))
            current_chunk = []
            current_length = 0
            chunk_start = seg['start']

        current_chunk.append(seg)
        current_length += seg_len

    # Save last chunk if exists
    if current_chunk:
        chunk_tensor = collect_chunks(current_chunk, wav)
        segments.append(chunk_tensor)
        chunk_end = current_chunk[-1]['end']
        chunk_times.append((chunk_start, chunk_end))

    # Write segments to disk and print their timestamps
    for i, (seg_tensor, times) in enumerate(zip(segments, chunk_times)):
        seg_np = seg_tensor.numpy()
        out_path = os.path.join(out_dir, f"segment_{i+1}.wav")
        write(out_path, sampling_rate, seg_np)
        start_sec = times[0] / sampling_rate
        end_sec = times[1] / sampling_rate
        print(f"Saved {out_path}, length: {len(seg_np) / sampling_rate:.2f} sec, Start: {start_sec:.2f}, End: {end_sec:.2f}")
        
        # Create corresponding text file if SRT is available
        if srt_entries:
            segment_text = get_text_for_segment(srt_entries, start_sec, end_sec)
            if segment_text.strip():  # Only create file if there's text content
                text_path = os.path.join(out_dir, f"segment_{i+1}.txt")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(segment_text.strip())
                print(f"Saved {text_path}")

    return segments


if __name__ == "__main__":
    segments = cut_wav_into_segments("MSBuild2025.wav", "MSBuild2025.srt")
