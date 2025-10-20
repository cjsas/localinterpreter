from nemo.collections.speechlm2.models import SALM
import re

model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
model = model.cuda()

with open("CanaryTranscript.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()


prompt = "Please translate the following text into Chinese and only output the translated text /no_think"


for i, line in enumerate(lines):
    transcript = line.strip()
    if not transcript:
        continue

    # Run generation
    with model.llm.disable_adapter():
        answer_ids = model.generate(
            prompts=[[{"role": "user", "content": f"{prompt}\n\n{transcript}"}]],
            max_new_tokens=2048,
        )
   
        output = model.tokenizer.ids_to_text(answer_ids[0].cpu())

        #print(output)

        parts = re.split(r"</think>", output)

        # Get the part after the last </think> (if any)
        if len(parts) > 1:
            clean_output = parts[-1].strip()
        else:
            clean_output = output.strip()

        print(clean_output)

