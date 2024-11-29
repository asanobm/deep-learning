import torch
from peft import LoraConfig
from datasets import load_dataset
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU 디바이스 설정

dataset = load_dataset("s076923/llama3-wikibook-ko")  # 위키북 한국어 데이터셋 로드

quantization_config = BitsAndBytesConfig(  # 4-bit 양자화 설정
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.float16,
  bnb_4bit_use_double_quant=False
)

model_name = "meta-llama/Meta-llama-3.1-8B-Instruct"  # 사용할 모델 이름

tokenizer = AutoTokenizer.from_pretrained(  # 토크나이저 로드
  model_name,
  trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(  # 모델 로드
  model_name,
  quantization_config=quantization_config,
  device_map={"": 0},
)

tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 설정
model.config.use_cache = False  # 캐시 사용 안함

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 디바이스 설정
model.to(device)  # 모델을 디바이스로 이동

# SFTTrainer 초기화
trainer = SFTTrainer(
    model=model,  # 모델
    args=SFTConfig(  # SFT 설정
      output_dir="./runs/"+model_name,  # 출력 디렉토리
      overwrite_output_dir=True,  # 출력 디렉토리 덮어쓰기
      max_seq_length=64,  # 최대 시퀀스 길이
      dataset_text_field="text",  # 데이터셋의 텍스트 필드 이름
      per_device_train_batch_size=4,  # 디바이스 당 배치 크기
      gradient_accumulation_steps=8,  # 그래디언트 누적 스텝 수
      max_steps=5000,  # 최대 스텝 수
      learning_rate=2e-4,  # 학습률
      warmup_steps=100,  # 워밍업 스텝 수
      logging_steps=100,  # 로깅 스텝 수
      save_steps=100,  # 저장 스텝 수
      eval_steps=100,  # 평가 스텝 수
      fp16=True,  # FP16 사용
      optim="paged_adamw_8bit",  # 옵티마이저
      seed=42,  # 랜덤 시드
    ),
    peft_config=LoraConfig(  # LoRA 설정
      r=64,  # LoRA 랭크
      lora_alpha=2,  # LoRA 알파
      lora_dropout=0.01,  # LoRA 드롭아웃
      task_type="CAUSAL_LM"  # 작업 유형
    ),
    processing_class=tokenizer,  # 토크나이저
    train_dataset=dataset["train"],  # 훈련 데이터셋
)

print(trainer.args)  # 트레이너 설정 출력

trainer.train()  # 모델 훈련

# 가장 훈련이 잘된 체크포인트를 저장
trainer.save_model("./runs/"+model_name)
tokenizer.save_pretrained("./runs/"+model_name)