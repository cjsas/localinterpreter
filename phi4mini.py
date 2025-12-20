import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
 
torch.random.manual_seed(0)

model_path = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

system = """
You are a translation engine.
Translate the user input into Chinese exactly.
Do NOT answer questions, give explanations, or add any content.
If the input contains a question, translate it as a question.
Output ONLY the translated text.
"""
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": "Give me a short introduction to large language model."},
]
 
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
 
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
 
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
