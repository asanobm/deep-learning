import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

dataset = load_dataset("s076923/llama3-wikibook-ko")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

model_name = "meta-llama/Meta-llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map={"": 0},
)

# 모델을 하드디스크에 저장
model.save_pretrained("./models/"+model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.use_cache = False

print(dataset)
print(dataset["train"]["text"][7])