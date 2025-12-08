# 🏥 Vet RAG Project: 수의학 전문 질의응답 시스템

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Required](https://img.shields.io/badge/GPU-Required-green.svg)](https://www.nvidia.com/ko-kr/)

## 📋 프로젝트 개요

**수의학 도메인 특화 RAG(Retrieval-Augmented Generation) 시스템**으로, 반려동물 보호자의 질문에 대해 의학적으로 정확하고 이해하기 쉬운 답변을 제공합니다.

### 핵심 특징
- ✅ **고품질 데이터 큐레이션**: LLM + 그래프 알고리즘 기반 자동 학습 데이터 생성
- ✅ **RAG² 통합**: OpenAI API를 활용한 쿼리 확장으로 검색 성능 향상
- ✅ **4단계 추론 파이프라인**: Rationale → Retrieval → Reranking → Generation
- ✅ **GPU 메모리 최적화**: RTX 4060 (8GB VRAM)에서 안정적 구동
- ✅ **구조화된 답변**: 진단/평가, 조치사항, 주의사항, 근거 요약 형식

---

## 🚀 시스템 아키텍처

### 1️⃣ 데이터 큐레이션 파이프라인
```
원본 데이터 (문서 + 질문)
    ↓
【Phase 1】 Retrieval (KmBERT Bi-Encoder)
    → Top-50 후보 문서 검색
    ↓
【Phase 2】 LLM Scoring (GPT-2/Qwen)
    → Perplexity 기반 적합성 평가
    ↓
【Phase 3】 Graph Refinement (LightGCN)
    → k-NN 그래프 + 점수 전파로 품질 보정
    ↓
【Phase 4】 Auto-Labeling
    → Top-5 문서: label=1, 나머지: label=0
    ↓
curated_dataset.json (학습 데이터)
```

### 2️⃣ 모델 학습 (Cross-Encoder Training)
```
curated_dataset.json
    ↓
Base Model: madatnlp/km-bert
    ↓
Binary Classification Training
    → 질문-문서 연관성 판단
    ↓
results/final_model/ (학습된 Reranker)
```

### 3️⃣ 추론 파이프라인
```
사용자 질문
    ↓
【Step 0】 Rationale Generation (OpenAI gpt-4o-mini)
    → "강아지가 노란 토를 해요"
    → "구토, 공복토, 담즙 역류, 위장관 질환"
    ↓
【Step 1】 Retrieval (KmBERT Bi-Encoder)
    → Top-50 문서 검색 (코사인 유사도)
    ↓
【Step 2】 Reranking (Fine-tuned Cross-Encoder)
    → Top-3 문서로 정밀 재순위화
    ↓
【Step 3】 Answer Generation (OpenAI gpt-4o-mini)
    → 구조화된 답변 생성
    ↓
최종 답변 (4가지 항목)
    1. 핵심 진단/평가
    2. 추가 조치
    3. 주의사항
    4. 근거 요약
```

---

## 💻 시스템 요구 사항

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| **GPU** | NVIDIA RTX 3060 (8GB VRAM) | RTX 4060 이상 (8GB+ VRAM) |
| **RAM** | 16GB | 32GB |
| **Storage** | 20GB 여유 공간 | 50GB+ SSD |
| **OS** | Windows 10/11 | Windows 11 / Linux |
| **Python** | 3.8+ | 3.10+ |

### 추가 요구사항
- **OpenAI API Key**: Rationale Generation 및 Answer Generation에 필요
- **Hugging Face Token**: 모델 다운로드에 필요 (무료)

---

## 📂 프로젝트 구조

```
catholic_retriver/
├── 📁 data/                              # 원본 데이터
│   ├── TS_말뭉치데이터_내과/              # 수의학 문서 (JSON)
│   ├── TS_말뭉치데이터_안과/
│   ├── TS_말뭉치데이터_외과/
│   ├── TS_말뭉치데이터_치과/
│   ├── TS_말뭉치데이터_피부과/
│   └── Training/02.라벨링데이터/         # 질의응답 데이터 (ZIP)
│
├── 📁 vet_rag_project/                   # 메인 프로젝트
│   ├── 📄 config.yaml                   # 시스템 설정 파일 ⚙️
│   ├── 📄 main_curation.py              # 실행 진입점 🚀
│   ├── 📄 requirements.txt              # Python 패키지 목록
│   ├── 📄 .env                          # API 키 설정 (생성 필요)
│   │
│   ├── 📁 src/                          # 핵심 모듈
│   │   ├── 📄 curator.py               # 데이터 큐레이션 (5-Phase)
│   │   ├── 📄 data_loader.py           # 데이터 로딩 및 청킹
│   │   ├── 📄 embedding.py             # KmBERT 임베딩
│   │   ├── 📄 graph_refiner.py         # LightGCN 그래프 전파
│   │   ├── 📄 llm_scorer.py            # PPL 기반 평가
│   │   ├── 📄 module_augment.py        # 프롬프트 생성
│   │   ├── 📄 rag_pipeline.py          # 추론 파이프라인 (핵심!)
│   │   └── 📄 trainer.py               # Cross-Encoder 학습
│   │
│   ├── 📁 cache/                        # 캐시 데이터 (자동 생성)
│   │   └── doc_embeddings.npy          # 문서 임베딩 캐시
│   │
│   ├── 📁 results/                      # 학습 결과 (자동 생성)
│   │   └── final_model/                # Fine-tuned Reranker
│   │
│   ├── 📁 logs/                         # 학습 로그 (자동 생성)
│   └── 📄 curated_dataset.json         # 큐레이션 결과 (자동 생성)
│
└── 📄 README.md                         # 이 문서
```

### 핵심 모듈 설명

| 파일명 | 역할 | 주요 기능 |
|--------|------|----------|
| **config.yaml** | 설정 관리 | 모델 경로, 하이퍼파라미터, API 설정 |
| **main_curation.py** | 실행 진입점 | `--mode curate/train/inference/all` |
| **rag_pipeline.py** | 추론 엔진 | Rationale → Retrieval → Rerank → Generation |
| **curator.py** | 큐레이션 | Retrieval → Scoring → Graph → Labeling |
| **module_augment.py** | 프롬프트 | 구조화된 답변 형식 생성 |

---

## 📊 데이터 형식

### 1. 문서 데이터 (Knowledge Base)

**위치**: `data/TS_말뭉치데이터_내과/` 등

**형식**: JSON 파일 (각 문서당 1개 파일)
```json
{
  "title": "개(2판) - 심장 질환",
  "department": "내과",
  "disease": "심장사상충증은 모기에 의해 전파되는 기생충 질환으로..."
}
```

**필수 필드**:
- `title`: 문서 제목
- `department`: 진료과 (내과, 외과, 안과, 치과, 피부과)
- `disease`: 본문 내용

### 2. 질의응답 데이터 (Training Queries)

**위치**: `data/Training/02.라벨링데이터/TL_질의응답데이터_내과.zip`

**형식**: ZIP 내부에 JSON 파일들
```json
{
  "question": "강아지가 기침을 해요. 심장사상충일까요?",
  "answer": "기침은 심장사상충의 주요 증상 중 하나입니다..."
}
```

**필수 필드**:
- `question`: 사용자 질문
- `answer`: 정답 (큐레이션 시 직접 사용되지 않음, 참고용)

---

## 🛠️ 설치 및 실행 가이드

### Step 0️⃣: 환경 변수 설정 (필수)

#### 1. **OpenAI API Key 발급** (Rationale & Answer Generation용)
```bash
# https://platform.openai.com/api-keys 접속
# API Key 생성 및 복사
```

#### 2. **Hugging Face Token 발급** (모델 다운로드용)
```bash
# https://huggingface.co/settings/tokens 접속
# "New token" 클릭 → "Read" 권한으로 생성
```

#### 3. **`.env` 파일 생성**
```bash
cd vet_rag_project

# .env 파일 생성 및 편집
notepad .env  # Windows
# 또는
nano .env     # Linux
```

**`.env` 파일 내용**:
```bash
# OpenAI API Key (gpt-4o-mini 사용)
TOKEN=sk-proj-your_actual_openai_api_key_here

# Hugging Face Token (모델 다운로드)
HF_TOKEN=hf_your_actual_huggingface_token_here

# Weights & Biases 비활성화 (선택)
WANDB_DISABLED=true
```

---

### Step 1️⃣: Python 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv .venv

# 가상환경 활성화
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 패키지 설치
cd vet_rag_project
pip install -r requirements.txt

# GPU 가속 (CUDA 지원)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Step 2️⃣: 데이터 큐레이션 (학습 데이터 생성)

```bash
python main_curation.py --mode curate
```

**처리 과정**:
1. `data/` 폴더의 문서 로딩
2. KmBERT로 문서 임베딩 생성 (캐시됨)
3. 600개 질문에 대해 Top-50 검색
4. LLM Scoring (PPL 평가)
5. Graph Refinement (LightGCN)
6. `curated_dataset.json` 생성

**예상 시간**: 약 30-60분 (GPU 성능에 따라 다름)

---

### Step 3️⃣: Cross-Encoder 학습

```bash
python main_curation.py --mode train --curated_data curated_dataset.json
```

**학습 설정** (`config.yaml`에서 조정 가능):
- Epochs: 3
- Batch Size: 16
- Learning Rate: 2e-5
- 결과 모델: `results/final_model/`

**예상 시간**: 약 10-30분

---

## 🤗 Hugging Face 모델 사용법

본 프로젝트의 Fine-tuned Cross-Encoder는 Hugging Face에 공개되어 있습니다.

### 모델 정보
- **모델명**: [JOhyeongi/vet-kmbert-cross-encoder](https://huggingface.co/JOhyeongi/vet-kmbert-cross-encoder)
- **베이스 모델**: madatnlp/km-bert
- **태스크**: Binary Classification (질문-문서 연관성 판단)
- **언어**: 한국어

### 사용 예제

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 모델 다운로드 및 로드
model = AutoModelForSequenceClassification.from_pretrained(
    "JOhyeongi/vet-kmbert-cross-encoder"
)
tokenizer = AutoTokenizer.from_pretrained(
    "JOhyeongi/vet-kmbert-cross-encoder"
)

# 추론
query = "강아지가 구토를 해요"
document = "구토의 원인은..."
inputs = tokenizer([[query, document]], return_tensors="pt", max_length=512)
score = model(**inputs).logits.softmax(dim=1)[0][1].item()

print(f"연관성 점수: {score:.4f}")
```

---

### Step 4️⃣: 추론 테스트

```bash
python main_curation.py --mode inference \
    --model_path results/final_model \
    --query "강아지가 노란 토를 해요. 어떻게 해야 하나요?"
```

**출력 예시**:
```
========================================
STEP 0: Rationale Generation (RAG²)
========================================
[INFO] [Rationale] Extracted keywords: ['구토', '공복토', '담즙 역류', '위장관 질환']
[INFO] [Rationale] Expanded Query: 강아지가 노란 토를 해요 [SEP] 구토, 공복토, 담즙 역류...

========================================
STEP 1: Retrieval
========================================
[SUCCESS] Retrieved 50 candidates.

========================================
STEP 2: Reranking
========================================
[SUCCESS] Reranked top 3 documents.
   [1] Score: 0.9812 | 공복토는 위산과 담즙이 섞여...
   [2] Score: 0.9543 | 담즙 역류는 십이지장의 내용물이...
   [3] Score: 0.8901 | 위장관 질환의 주요 증상...

========================================
STEP 3: Answer Generation
========================================

1. **핵심 진단/평가**: 
   노란색 토는 공복토 또는 담즙 역류의 가능성이 높습니다...

2. **추가 조치**: 
   - 식사 간격을 좁혀주세요 (하루 2-3회 소량 급여)
   - 증상이 지속되면 동물병원 방문 필요

3. **주의사항**: 
   - 구토 횟수, 색상, 시간대를 기록하세요
   - 혈액이 섞이거나 검은색이면 즉시 병원

4. **근거 요약**:
   - 공복토는 위산과 담즙이 섞여 노란색으로 보입니다
   - 식사 간격을 줄이면 증상 개선 가능
```

---

### Step 5️⃣: 전체 파이프라인 실행 (한 번에)

```bash
python main_curation.py --mode all
```

큐레이션 → 학습 → 추론을 순차적으로 실행합니다.

---

## ⚙️ 설정 파일 (config.yaml)

### 주요 설정 항목

```yaml
# ========================================
# 1. Knowledge Base (문서 데이터 경로)
# ========================================
knowledge_base:
  directories:
    - "..\\data\\TS_말뭉치데이터_내과"
    - "..\\data\\TS_말뭉치데이터_안과"
    - "..\\data\\TS_말뭉치데이터_외과"
    - "..\\data\\TS_말뭉치데이터_치과"
    - "..\\data\\TS_말뭉치데이터_피부과"

# ========================================
# 2. Retrieval (검색 모델)
# ========================================
retrieval:
  model_name: "madatnlp/km-bert"  # 한국어 Bi-Encoder
  top_k: 50                        # 1차 검색 문서 수

# ========================================
# 3. LLM Scorer (큐레이션용, 가벼운 모델)
# ========================================
llm_scorer:
  model_name: "gpt2"               # PPL 평가용 (빠름)
  alpha_high_std: 0.7              # 고품질 필터링 threshold
  beta_low_std: 0.7                # 저품질 필터링 threshold

# ========================================
# 4. Answer Generation (추론용, OpenAI API)
# ========================================
llm:
  model_name: "gpt-4o-mini"        # OpenAI API 모델
  api_key: null                    # .env의 TOKEN 사용
  temperature: 0.7                 # 창의성 조절 (0.0~1.0)
  max_tokens: 512                  # 최대 생성 토큰 수

# ========================================
# 5. Rationale Generation (쿼리 확장, OpenAI API)
# ========================================
rationale_gen:
  enabled: true                    # RAG² 활성화 (권장)
  model_name: "gpt-4o-mini"        # OpenAI API 모델
  temperature: 0.1                 # 낮게 (일관성 있는 키워드)
  top_p: 0.9
  max_tokens: 128                  # 키워드만 필요
  prompt_template: |               # 커스텀 프롬프트 (선택)
    [역할] 수의학 검색어 확장 전문가
    [입력 질문] {query}
    [목표] 핵심 의학 키워드 3~8개 추출...

# ========================================
# 6. Graph Refinement (LightGCN)
# ========================================
graph:
  k_neighbors: 5                   # k-NN 그래프
  lambda_propagation: 0.3          # 전파 강도
  propagation_steps: 3             # 전파 레이어 수

# ========================================
# 7. Training (Cross-Encoder 학습)
# ========================================
training:
  model_type: "bert"               # "bert" 또는 "t5"
  output_dir: "./results"
  num_train_epochs: 3
  per_device_train_batch_size: 16  # GPU 메모리 부족 시 8로 변경
  learning_rate: 2e-5
  save_total_limit: 1              # 최신 체크포인트만 유지
  logging_steps: 10
  eval_strategy: "epoch"
  load_best_model_at_end: true
```

### 설정 변경 가이드

| 변경하고 싶은 것 | 수정할 설정 | 값 |
|-----------------|------------|-----|
| **Rationale 비활성화** | `rationale_gen.enabled` | `false` |
| **OpenAI → 로컬 모델** | `llm.model_name` | `"Qwen/Qwen2.5-7B-Instruct"` |
| **GPU 메모리 부족** | `training.per_device_train_batch_size` | `8` (또는 `4`) |
| **검색 문서 수 증가** | `retrieval.top_k` | `100` |
| **학습 에폭 증가** | `training.num_train_epochs` | `5` |

---

## 🔧 핵심 기술 및 최적화

### 1️⃣ RAG² (Rationale-Augmented Generation)
```python
# 사용자 질문을 의학 전문 용어로 자동 확장
Original Query: "강아지가 노란 토를 해요"
    ↓ [OpenAI gpt-4o-mini]
Expanded Query: "강아지가 노란 토를 해요 [SEP] 구토, 공복토, 담즙 역류, 위장관 질환"
    ↓
검색 성능 향상 (+15~25%)
```

**효과**:
- 일상 언어 → 의학 용어 자동 변환
- 동의어/관련어 확장으로 검색 커버리지 증가
- Zero-shot (추가 학습 불필요)

### 2️⃣ 5-Phase 데이터 큐레이션
```
Phase 1: Retrieval (KmBERT)
    → Top-50 후보 문서
Phase 2: LLM Scoring (GPT-2 PPL)
    → 적합성 점수 (0~1)
Phase 3: Graph Refinement (LightGCN)
    → k-NN 그래프 + 점수 전파
Phase 4: Auto-Labeling
    → Top-5: label=1, 나머지: label=0
Phase 5: Validation
    → 품질 검증 및 저장
```

**효과**:
- 수동 라벨링 불필요 (100% 자동화)
- 그래프 알고리즘으로 노이즈 제거
- 고품질 학습 데이터 생성

### 3️⃣ GPU 메모리 최적화
- ✅ **순차적 모델 로딩**: 큐레이션 시 모델을 하나씩 로드/언로드
- ✅ **중간 결과 캐싱**: 임베딩, PPL 점수를 `cache/`에 저장
- ✅ **OpenAI API 활용**: 로컬 LLM 대신 API 사용으로 VRAM 절약
- ✅ **Gradient Checkpointing**: 학습 시 메모리 사용량 감소

**결과**: RTX 4060 (8GB VRAM)에서 안정적 구동

### 4️⃣ 구조화된 답변 생성
```
사용자 질문 → 4가지 항목으로 구조화된 답변

1. 핵심 진단/평가
   → 증상의 의학적 의미

2. 추가 조치
   → 즉시 실행 가능한 행동 가이드

3. 주의사항
   → 위험 신호 및 모니터링 포인트

4. 근거 요약
   → 답변의 출처 (참고 자료에서 추출)
```

**효과**:
- 보호자가 이해하기 쉬운 형식
- Hallucination 방지 (근거 요약 필수)
- 일관된 답변 품질

---

## 📈 성능 벤치마크

| 지표 | Baseline<br>(KmBERT만) | +Reranking | +RAG²<br>(최종) |
|------|----------------------|-----------|---------------|
| **Top-3 정확도** | 42% | 68% | 79% |
| **응답 시간** | 0.8초 | 2.1초 | 2.5초 |
| **GPU 메모리** | 2.1GB | 3.4GB | 3.6GB |

**테스트 환경**: RTX 4060 (8GB), 180개 평가 질문

### 주요 개선 사항
- 🎯 **검색 정확도**: Cross-Encoder로 Top-3 정확도 +37%p
- 🚀 **추론 속도**: Bi-Encoder (빠른 검색) + Cross-Encoder (정밀 재순위화)
- 💾 **메모리 효율**: OpenAI API 활용으로 8GB VRAM에서 구동
- 🎓 **쿼리 확장**: RAG²로 검색 성능 +11%p 추가 향상

---

## 🐛 문제 해결 (Troubleshooting)

### ❌ CUDA Out of Memory

**증상**: `RuntimeError: CUDA out of memory`

**해결 방법**:
```yaml
# config.yaml 수정
training:
  per_device_train_batch_size: 8  # 16 → 8로 감소
  gradient_accumulation_steps: 2  # 추가
```

또는 GPU 메모리 정리:
```python
import torch
torch.cuda.empty_cache()
```

---

### ❌ OpenAI API Key 오류

**증상**: `InvalidAPIKey` 또는 `Authentication failed`

**해결 방법**:
1. `.env` 파일에 `TOKEN=sk-proj-...` 형식으로 저장 확인
2. API Key가 유효한지 확인: https://platform.openai.com/api-keys
3. API 사용량 제한 확인: https://platform.openai.com/usage

---

### ❌ Hugging Face Token 오류

**증상**: `401 Unauthorized` 또는 `Access denied`

**해결 방법**:
```bash
# .env 파일 확인
HF_TOKEN=hf_your_token_here  # 형식 확인

# 또는 환경변수로 설정
export HF_TOKEN=hf_your_token_here  # Linux/Mac
$env:HF_TOKEN="hf_your_token_here"  # Windows PowerShell
```

---

### ❌ 캐시 파일 손상

**증상**: `ValueError: could not load embeddings` 또는 오래된 캐시

**해결 방법**:
```bash
# Windows
Remove-Item -Recurse -Force cache
python main_curation.py --mode curate

# Linux/Mac
rm -rf cache
python main_curation.py --mode curate
```

---

### ❌ 느린 추론 속도

**원인**: Rationale Generation이 활성화되어 있음

**해결 방법** (빠른 추론이 필요한 경우):
```yaml
# config.yaml 수정
rationale_gen:
  enabled: false  # RAG² 비활성화
```

**효과**:
- 응답 시간: 2.5초 → 1.2초
- 정확도 하락: 79% → 68% (약 -11%p)

---

### ❌ 모델 다운로드 실패

**증상**: `Connection timeout` 또는 `HTTP 503`

**해결 방법**:
```bash
# 1. 인터넷 연결 확인
ping huggingface.co

# 2. 프록시 설정 (필요시)
export HF_HUB_ENABLE_HF_TRANSFER=1

# 3. 모델 수동 다운로드
huggingface-cli download madatnlp/km-bert
```

---

### ⚠️ 경고: Weights & Biases

**증상**: `wandb` 로그인 요청

**해결 방법**:
```bash
# .env 파일에 추가
WANDB_DISABLED=true
```

---

### 프로젝트 구조 설명
```
핵심 파이프라인 흐름:

1. 데이터 큐레이션 (main_curation.py --mode curate)
   └─> curator.py (5-Phase)
       ├─> data_loader.py (문서/질문 로딩)
       ├─> embedding.py (KmBERT 임베딩)
       ├─> llm_scorer.py (PPL 평가)
       └─> graph_refiner.py (LightGCN 전파)

2. 모델 학습 (main_curation.py --mode train)
   └─> trainer.py (Cross-Encoder 학습)

3. 추론 (main_curation.py --mode inference)
   └─> rag_pipeline.py
       ├─> generate_rationale() (OpenAI API)
       ├─> retrieve() (KmBERT)
       ├─> rerank() (Fine-tuned Cross-Encoder)
       └─> generate() (OpenAI API)
```

---

---

**최종 업데이트**: 2025년 12월 8일
