import requests
import torch
import os
import io
import time
from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from urllib.request import urlopen


# Define model path
model_path = "Lexius/Phi-4-multimodal-instruct"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True,
    _attn_implementation='flash_attention_2',
).cuda()

# Load generation config
generation_config = GenerationConfig.from_pretrained(model_path)

# Define prompt structure
user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
speech_prompt = "Transcribe the audio to text, and then translate the audio to Chinese. Use <sep> as a separator between the original transcript and the translation."
prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')

# Process all wav files in segments folder
segments_dir = "./segments"
output_dir = "./segments_outputs"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all wav files in segments directory
wav_files = [f for f in os.listdir(segments_dir) if f.endswith('.wav')]
wav_files.sort()  # Sort to process in order

print(f"Found {len(wav_files)} wav files to process")

# Process each wav file
total_audio_duration = 0
total_compute_time = 0

for i, wav_file in enumerate(wav_files):
    audio_path = os.path.join(segments_dir, wav_file)
    output_file = os.path.join(output_dir, f"{wav_file[:-4]}.txt")
    
    print(f"Processing {i+1}/{len(wav_files)}: {wav_file}")
    
    try:
        # Load audio file
        audio, samplerate = sf.read(audio_path)
        audio_duration = len(audio) / samplerate  # Duration in seconds
        
        # Start timing the compute
        compute_start = time.time()
        
        # Process with the model
        inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')
        num_logits_to_keep = 1
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=1000,
            num_logits_to_keep=num_logits_to_keep
        )
        
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # End timing the compute
        compute_end = time.time()
        compute_time = compute_end - compute_start
        
        # Calculate RTFX for this file (but don't display)
        rtfx = audio_duration / compute_time
        
        # Update totals
        total_audio_duration += audio_duration
        total_compute_time += compute_time
        
        # Save response to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"File: {wav_file}\n")
            f.write(f"Response:\n{response}\n")
        
        print(f"✓ Completed {wav_file} -> {output_file}")
        
    except Exception as e:
        print(f"✗ Error processing {wav_file}: {str(e)}")
        # Save error to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"File: {wav_file}\n")
            f.write(f"Error: {str(e)}\n")

# Calculate overall RTFX
overall_rtfx = total_audio_duration / total_compute_time if total_compute_time > 0 else 0
print(f"Processing complete! Results saved in {output_dir}")
print(f"Overall RTFX: {overall_rtfx:.2f}x")

