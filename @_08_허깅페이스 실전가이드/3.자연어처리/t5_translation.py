"""
TMUX를 이용해서 여러 개의 터미널을 동시에 실행할 수 있음.

tmux new -s clearml
tmux attach -t clearml
"""

import os
import clearml
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1개만 사용

# ClearML 프로젝트 초기화
clearml_project = clearml.Task.init(project_name="Translation", task_name="T5 Translation")

# ClearML 프로젝트 설정
import torch
import evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5TokenizerFast, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(example, tokenizer):
  translation = example['translation']
  translation_source = ["en: " + instance["en"] for instance in translation]  # 영어 문장 앞에 "en: " 접두사 추가
  translation_target = ["ko: " + instance["ko"] for instance in translation]  # 한국어 문장 앞에 "ko: " 접두사 추가

  tokenized = tokenizer(
    translation_source,
    text_target=translation_target,
    truncation=True,  # 최대 길이를 초과하는 시퀀스는 잘라냄
    return_tensors="pt",  # PyTorch 텐서로 반환하여 GPU 가속 활용 
    padding=True,  # 배치 내 시퀀스 길이를 통일하기 위해 패딩 적용
  )

  return tokenized

# 데이터셋 로드 및 전처리
dataset = load_dataset("Helsinki-NLP/opus-100", "en-ko")  # OPUS-100 데이터셋의 영어-한국어 쌍 로드
tokenizer = T5TokenizerFast.from_pretrained("KETI-AIR/long-ke-t5-small")  # 한국어-영어 번역을 위한 T5 토크나이저 로드

# 훈련 데이터셋을 10만개로, 검증 데이터셋을 1천개로 줄임
train_dataset = dataset["train"].select(range(100000))
test_dataset = dataset["test"].select(range(1000))

processed_train_dataset = train_dataset.map(
  lambda example: preprocess_data(example, tokenizer),  # 훈련 데이터셋의 각 예제에 전처리 함수 적용 
  batched=True,  # 배치 단위로 전처리하여 속도 향상
  remove_columns=train_dataset.column_names,  # 전처리 후 불필요한 열 제거
)

processed_test_dataset = test_dataset.map(
  lambda example: preprocess_data(example, tokenizer),  # 테스트 데이터셋의 각 예제에 전처리 함수 적용
  batched=True,
  remove_columns=test_dataset.column_names,
)

# 데이터 로더 설정
data_collator = DataCollatorForSeq2Seq(
  tokenizer=tokenizer,
  padding="longest",  # 배치 내 가장 긴 시퀀스에 맞춰 패딩 적용
  return_tensors="pt",  # PyTorch 텐서로 반환
)

# 모델 설정
model = T5ForConditionalGeneration.from_pretrained("KETI-AIR/long-ke-t5-small")  # 사전 학습된 T5 모델 로드

# 학습 파라미터 설정
training_args = Seq2SeqTrainingArguments(
  output_dir="./runs/t5-translation",  # 모델 체크포인트와 로그를 저장할 디렉토리
  # per_device_train_batch_size=16,  # 훈련 배치 크기를 16
  # per_device_eval_batch_size=24,  # 평가 배치 크기를 24
  # learning_rate=5e-5,  # 학습률
  # num_train_epochs=100,  # 총 훈련 에포크 수
  # evaluation_strategy="epoch",  # 에포크 단위로 평가 수행
  # save_strategy="epoch",  # 에포크 단위로 모델 저장
  # logging_strategy="epoch",  # 에포크 단위로 로깅
  # seed=42,  # 재현 가능성을 위한 랜덤 시드 설정
  per_device_train_batch_size=16,  # 훈련 배치 크기
  per_device_eval_batch_size=24,  # 평가 배치 크기
  learning_rate=2e-5,  # 학습률
  num_train_epochs=20,  # 학습 에포크 수
  eval_steps=250,  # 평가 스텝
  logging_steps=250,  # 로그 스텝
  save_steps=250,  # 저장 스텝
  seed=42,  # 랜덤 시드
  fp16=True,  # 플로트 16 비트 연산 활성화
  gradient_accumulation_steps=2,  # 그래디언트 축적 단계를 4로 설정 (메모리 사용량 줄임)
  dataloader_num_workers=4,  # 데이터 로딩에 사용할 서브프로세스 수 (병렬화)
  dataloader_pin_memory=True,  # 텐서를 CUDA 고정 메모리에 할당하여 데이터 로딩 속도 향상
)

trainer = Seq2SeqTrainer(
  model=model,  # 모델 지정
  args=training_args,  # 훈련 인자 설정
  train_dataset=processed_train_dataset,  # 전처리된 훈련 데이터셋 사용
  eval_dataset=processed_test_dataset,  # 전처리된 테스트 데이터셋 사용
  data_collator=data_collator,  # 데이터 콜레이터 함수 지정
  tokenizer=tokenizer,  # 토크나이저 지정
)

trainer.train()  # 모델 훈련 시작

# 모델 평가 지표 로깅
eval_results = trainer.evaluate()  # 훈련된 모델 평가

# 학습된 모델 아티팩트 저장

# 가장 성능이 좋은 모델 저장 
model.save_pretrained("./runs/t5-translation/best-model")  # 최상의 모델을 디스크에 저장

# 모델 평가
true_translated_ids = processed_test_dataset.select(range(100))["labels"]  # 테스트셋에서 100개의 실제 번역 시퀀스 추출
true_translated = tokenizer.batch_decode(true_translated_ids, skip_special_tokens=True)  # 실제 번역 시퀀스를 디코딩

generated_translated = []  # 생성된 번역 문장을 저장할 리스트

with torch.no_grad():  # 그래디언트 계산 비활성화
  for batch in DataLoader(processed_test_dataset.select(range(100)), batch_size=4):  # 테스트셋에서 100개의 예제를 배치 크기 4로 로드
    batch = batch.to(device)  # 배치를 GPU로 이동
    outputs = model.generate(**batch, max_length=1026, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)  # 번역 문장 생성
    batch_translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)  # 생성된 번역 문장 디코딩
    generated_translated.extend(batch_translated)  # 생성된 번역 문장을 리스트에 추가

bleu = evaluate.load("bleu")  # BLEU 평가 지표 로드
blue_score = bleu.compute(predictions=generated_translated, references=true_translated)  # BLEU 점수 계산
