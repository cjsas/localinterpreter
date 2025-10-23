# Local Interpreter

## Prerequisites

Install required packages:

```bash
pip install -r requirements.txt
```

Make sure you have CUDA-enabled GPU.

---

## Step 1: Voice Activity Detection (VAD)

Split your audio file into segments using Silero-VAD.

### Usage

```bash
python vad.py
```


### Output

```
segments/
├── segment_1.wav
├── segment_1.txt (if SRT provided)
├── segment_2.wav
├── segment_2.txt (if SRT provided)
└── ...
```

---

## Step 2 (Option 1): nvidia/canary-qwen-2.5b

- Run ```generate_segments_index.py```

    - Output ```segments_index.txt```

- (salm_generate.py is from Nemo repo) Run ```python salm_generate.py pretrained_name=nvidia/canary-qwen-2.5b inputs=segments_index.txt output_manifest=generations.jsonl batch_size=128 user_prompt="Transcribe the following:"```

    - Output ```generations.jsonl```

- Run extract_jsonl_text.py
->CanaryTranscript.txt (can be used for WER calculation)

---

## Step 2 (Option 2): microsoft/Phi-4-multimodal-instruct

- Run ```phi4.py```
