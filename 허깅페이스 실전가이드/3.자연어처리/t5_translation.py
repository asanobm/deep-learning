"""
TMUX를 이용해서 여러 개의 터미널을 동시에 실행할 수 있음.

tmux new -s mlflow
tmux attach -t mlflow
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 첫 번째 GPU를 사용하도록 설정

import torch
import mlflow  # MLflow 임포트 추가
import evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5TokenizerFast, T5ForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(example, tokenizer):
  translation = example['translation']  # 번역된 문장 쌍을 가져옴
  translation_source = ["en: " + instance["en"] for instance in translation]  # 영어 문장에 "en: " 접두사 추가
  translation_target = ["ko: " + instance["ko"] for instance in translation]  # 한국어 문장에 "ko: " 접두사 추가

  tokenized = tokenizer(
    translation_source,
    text_target=translation_target,
    truncation=True,  # 길이가 너무 길면 잘라냄
  )

  return tokenized  # 토큰화된 결과 반환

# MLflow 실험 설정
mlflow.set_experiment("t5-translation")

# MLflow 실행 시작 
with mlflow.start_run():
  model_name = "KETI-AIR/long-ke-t5-small"
  tokenizer = T5TokenizerFast.from_pretrained(model_name)
  model = T5ForConditionalGeneration.from_pretrained(model_name)

  dataset = load_dataset("Helsinki-NLP/opus-100", "en-ko")

  processed_dataset = dataset.map(
    lambda example: preprocess_data(example, tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names,
  )

  sample = processed_dataset["test"][0]  # 테스트 데이터셋에서 샘플 하나 가져오기
  print(sample)  # 샘플 출력
  print("변환된 출발 언어:", tokenizer.decode(sample["input_ids"]))  # 입력 텍스트 디코딩하여 출력
  print("변환된 도착 언어:", tokenizer.decode(sample["labels"]))  # 출력 텍스트 디코딩하여 출력

  seq2seq2_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding="longest",  # 가장 긴 시퀀스에 맞춰 패딩
    return_tensors="pt",  # PyTorch 텐서로 반환
  )

  training_args = Seq2SeqTrainingArguments(
    output_dir="./runs/t5-translation",  # 출력 디렉토리
    logging_dir='./logs',  # 로그 디렉토리
    per_device_train_batch_size=2,  # 훈련 배치 크기
    per_device_eval_batch_size=4,  # 평가 배치 크기
    learning_rate=2e-5,  # 학습률
    num_train_epochs=100,  # 학습 에포크 수
    evaluation_strategy="epoch",  # 에포크 단위로 평가 
    save_strategy="epoch",  # 에포크 단위로 모델 저장
    logging_strategy="epoch",  # 에포크 단위로 로깅
    seed=42,  # 랜덤 시드  
  )

  trainer = Seq2SeqTrainer(
    model=model,
    data_collator=seq2seq2_collator,
    args=training_args,
    train_dataset=processed_dataset["train"].select(range(100000)),
    eval_dataset=processed_dataset["test"].select(range(1000)),
  )

  trainer.train()

  # 모델 평가 지표 로깅
  eval_results = trainer.evaluate()
  mlflow.log_metrics(eval_results)

  # 학습된 모델 아티팩트 저장
  mlflow.pytorch.log_model(model, "model")

  # 가장 성능이 좋은 모델 저장 
  model.save_pretrained("./runs/t5-translation/best-model")

  # 모델 평가
  true_translated_ids = processed_dataset["test"].select(range(100))["labels"]
  true_translated = tokenizer.batch_decode(true_translated_ids, skip_special_tokens=True)

  generated_translated = []

  with torch.no_grad():
    for batch in DataLoader(processed_dataset["test"].select(range(100)), batch_size=4):
      batch = batch.to(device)
      outputs = model.generate(**batch, max_length=1026, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
      batch_translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)
      generated_translated.extend(batch_translated)

  bleu = evaluate.load("bleu")
  blue_score = bleu.compute(predictions=generated_translated, references=true_translated)
  mlflow.log_metrics(blue_score)
  
  # MLflow 실행 완료
  mlflow.end_run()