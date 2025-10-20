#!/usr/bin/env python3
import json
import re

def extract_jsonl_text(jsonl_file_path):
    """Extract transcribed text from JSONL file"""
    
    segments = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data = json.loads(line)
                    
                    # Extract segment info
                    segment_id = data.get('id', '')
                    
                    # Find the assistant's transcription in conversations
                    conversations = data.get('conversations', [])
                    transcription = None
                    
                    for conv in conversations:
                        if conv.get('from') == 'Assistant' and conv.get('type') == 'text':
                            transcription = conv.get('value', '')
                            break
                    
                    if transcription:
                        segments.append({
                            'id': segment_id,
                            'text': transcription.strip()
                        })
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {e}")
                    continue
    
    # Sort segments by segment ID
    segments.sort(key=lambda x: x['id'])
    
    return segments

def main():
    jsonl_file = "./generations.jsonl"
    
    print("=== Extracted Text from JSONL File ===\n")
    
    try:
        segments = extract_jsonl_text(jsonl_file)
        
        # Print all transcriptions in order
        for segment in segments:
            print(segment['text'])
        
        print(f"\n=== Total: {len(segments)} segments extracted ===")
        
        # Also create a combined text version
        combined_text = ' '.join([seg['text'] for seg in segments])
        
        # Break into sentences for readability
        sentences = re.split(r'(?<=[.!?])\s+', combined_text)
        
        # Write to CanaryTranscript.txt
        with open("CanaryTranscript.txt", "w", encoding="utf-8") as output_file:
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    print(sentence)
                    output_file.write(sentence + "\n")
                
        print(f"\n=== Total: {len([s for s in sentences if s.strip()])} sentences ===")
        print("=== Output written to CanaryTranscript.txt ===")
        
    except Exception as e:
        print(f"Error processing JSONL file: {str(e)}")

if __name__ == "__main__":
    main()