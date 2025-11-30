import os
import json
import yaml
import torch
import numpy as np
import inspect
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)

# 커스텀 Trainer 클래스 (클래스 가중치 지원)
class WeightedTrainer(Trainer):
    def __init__(self, class_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weight is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weight.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ---------------------------------------------------------
# 1. 커스텀 데이터셋 클래스 정의
# ---------------------------------------------------------
class VetRAGDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, model_type="bert"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]
        document = item["document"]
        label = item["label"]

        if self.model_type == "t5":
            # T5 형식: "query: {query} document: {document}"
            text = f"query: {query} document: {document}"
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )
        else:
            # BERT 형식: [CLS] Query [SEP] Document [SEP]
            encoding = self.tokenizer(
                query, 
                document, 
                truncation=True, 
                max_length=self.max_length, 
                padding="max_length"
            )
        
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# ---------------------------------------------------------
# 2. 평가 메트릭 계산 함수
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0, average='binary')
    recall = recall_score(labels, predictions, zero_division=0, average='binary')
    f1 = f1_score(labels, predictions, zero_division=0, average='binary')
    
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

# ---------------------------------------------------------
# 3. 학습 실행 메인 함수
# ---------------------------------------------------------
def train_cross_encoder(config_path="config.yaml", dataset_path="curated_dataset.json"):
    """
    Cross-Encoder 모델을 학습하는 메인 함수.
    BERT류와 T5류 모델을 모두 지원합니다.
    
    Args:
        config_path: 설정 파일 경로
        dataset_path: 학습 데이터셋 경로
    """
    # 설정 파일 로드
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    train_config = config.get("training", {})

    # Allow TOKEN fallback for OPENAI_API_KEY if not already set
    if not os.getenv("OPENAI_API_KEY") and os.getenv("TOKEN"):
        os.environ["OPENAI_API_KEY"] = os.getenv("TOKEN")
    
    # 모델 타입 확인 (bert 또는 t5)
    model_type = train_config.get("model_type", "bert").lower()
    if model_type not in ["bert", "t5"]:
        raise ValueError(f"Unsupported model_type: {model_type}. Must be 'bert' or 't5'")
    
    # 데이터셋 로드
    print(f"[INFO] Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please run curation mode first.")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    # 데이터셋 통계 출력
    num_positive = sum(1 for item in raw_data if item["label"] == 1)
    num_negative = sum(1 for item in raw_data if item["label"] == 0)
    print(f"[INFO] Original Dataset Statistics:")
    print(f"   Total: {len(raw_data)}")
    print(f"   Positive: {num_positive} ({num_positive/len(raw_data)*100:.2f}%)")
    print(f"   Negative: {num_negative} ({num_negative/len(raw_data)*100:.2f}%)")
    
    # 데이터 분할 (Train 80%, Val 10%, Test 10%)
    train_data, temp_data = train_test_split(raw_data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # 분할 후 통계
    train_pos = sum(1 for item in train_data if item["label"] == 1)
    train_neg = sum(1 for item in train_data if item["label"] == 0)
    print(f"\n[INFO] Dataset Split:")
    print(f"   Train: {len(train_data)} (Positive: {train_pos}, Negative: {train_neg})")
    print(f"   Val: {len(val_data)}")
    print(f"   Test: {len(test_data)}")
    
    # 클래스 가중치 계산 (불균형 해결)
    if train_pos > 0 and train_neg > 0:
        pos_weight = train_neg / train_pos
        print(f"   Class weight (pos/neg): 1.0 / {pos_weight:.2f}")
    else:
        pos_weight = 1.0
    
    # 모델 및 토크나이저 준비
    model_name = train_config.get("base_model", "madatnlp/km-bert")
    print(f"[INFO] Initializing {model_type.upper()} model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if model_type == "t5":
        # T5 모델의 경우: 일부 T5 모델은 SequenceClassification을 지원하지만,
        # 대부분은 Seq2Seq 구조이므로 encoder만 사용하여 classification head 추가
        # 여기서는 간단히 SequenceClassification으로 시도하고, 실패 시 경고
        try:
            # 일부 T5 기반 모델은 SequenceClassification을 지원할 수 있음
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            print("[SUCCESS] T5 모델을 SequenceClassification으로 로드했습니다.")
        except Exception as e:
            print(f"[WARNING] T5 모델을 SequenceClassification으로 로드할 수 없습니다: {e}")
            print("[WARNING] T5는 Seq2Seq 구조이므로 classification 태스크에 직접 사용하기 어렵습니다.")
            print("[WARNING] BERT류 모델 사용을 권장합니다. 또는 T5 encoder를 활용한 커스텀 모델이 필요합니다.")
            raise ValueError(f"T5 모델 {model_name}을 classification에 사용할 수 없습니다. BERT류 모델을 사용하세요.")
    else:
        # BERT류 모델 (BERT, RoBERTa, ELECTRA 등)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 데이터셋 객체 생성
    max_length = train_config.get("max_length", 512)
    train_dataset = VetRAGDataset(train_data, tokenizer, max_length=max_length, model_type=model_type)
    val_dataset = VetRAGDataset(val_data, tokenizer, max_length=max_length, model_type=model_type)
    test_dataset = VetRAGDataset(test_data, tokenizer, max_length=max_length, model_type=model_type)
    
    # 학습 인자 설정
    output_dir = train_config.get("output_dir", "./results")
    
    training_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=float(train_config.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_config.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(train_config.get("per_device_eval_batch_size", 16)),
        learning_rate=float(train_config.get("learning_rate", 2e-5)),
        warmup_steps=int(train_config.get("warmup_steps", 500)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        logging_dir=train_config.get("logging_dir", "./logs"),
        logging_steps=int(train_config.get("logging_steps", 10)),
        evaluation_strategy=train_config.get("eval_strategy", "epoch"),
        save_strategy=train_config.get("save_strategy", "epoch"),
        load_best_model_at_end=bool(train_config.get("load_best_model_at_end", True)),
        metric_for_best_model=train_config.get("metric_for_best_model", "f1"),
        save_total_limit=int(train_config.get("save_total_limit", 1)),
        fp16=bool(torch.cuda.is_available() and train_config.get("fp16", True)),
        report_to=train_config.get("report_to", "none"),  # "tensorboard" 등으로 설정 가능
    )

    # TrainingArguments가 지원하지 않는 키를 자동으로 필터링하여 버전에 따른 TypeError 방지
    valid_args = inspect.signature(TrainingArguments.__init__).parameters.keys()
    filtered_kwargs = {}
    dropped_keys = set()
    for key, value in training_kwargs.items():
        if key in valid_args:
            filtered_kwargs[key] = value
        else:
            print(f"[WARNING] TrainingArguments does not accept '{key}' in this transformers version. Dropping this option.")
            dropped_keys.add(key)

    evaluation_supported = ("evaluation_strategy" in training_kwargs and "evaluation_strategy" not in dropped_keys)
    evaluation_value = filtered_kwargs.get("evaluation_strategy", "no")

    if (not evaluation_supported) or (str(evaluation_value).lower() == "no"):
        if filtered_kwargs.get("load_best_model_at_end"):
            print("[WARNING] load_best_model_at_end requires evaluation. Disabling because evaluation strategy is unavailable.")
            filtered_kwargs["load_best_model_at_end"] = False

    training_args = TrainingArguments(**filtered_kwargs)
    
    # 클래스 가중치 설정 (불균형 데이터셋 해결)
    train_pos = sum(1 for item in train_data if item["label"] == 1)
    train_neg = sum(1 for item in train_data if item["label"] == 0)
    
    if train_pos > 0 and train_neg > 0:
        # Positive 클래스에 더 높은 가중치 부여
        class_weight = torch.tensor([1.0, train_neg / train_pos])
        print(f"   Using class weights: {class_weight.tolist()}")
    else:
        class_weight = None
    
    # 트레이너 초기화 (가중치 적용)
    if class_weight is not None:
        trainer = WeightedTrainer(
            class_weight=class_weight,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer)
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            data_collator=DataCollatorWithPadding(tokenizer)
        )
    
    # 학습 시작
    print("[INFO] Starting training...")
    trainer.train()
    
    # 테스트셋 평가
    print("\n" + "="*60)
    print("[INFO] Evaluating on test set...")
    print("="*60)
    test_results = trainer.evaluate(test_dataset)
    
    # 상세 평가 리포트 출력
    print("\n[INFO] Test Set Results:")
    for metric, value in test_results.items():
        if metric != "eval_loss":
            print(f"   {metric}: {value:.4f}")
    
    # 예측 및 분류 리포트
    print("\n[INFO] Detailed Classification Report:")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    print(classification_report(true_labels, pred_labels, target_names=["Negative", "Positive"]))
    
    # 혼동 행렬
    cm = confusion_matrix(true_labels, pred_labels)
    print("\n[INFO] Confusion Matrix:")
    print(f"   True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
    print(f"   False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")
    
    # 최종 모델 저장
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"\n[SUCCESS] Model saved to {final_model_path}")
    
    # 평가 결과 저장
    results_path = os.path.join(output_dir, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Test results saved to {results_path}")
    
    return trainer, test_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Cross-Encoder for VetRAG")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="curated_dataset.json", help="Path to dataset file")
    
    args = parser.parse_args()
    
    train_cross_encoder(config_path=args.config, dataset_path=args.dataset)