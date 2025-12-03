# Vet RAG Project: 수의학 전문 질의응답 시스템

이 프로젝트는 수의학 도메인에 특화된 RAG(Retrieval-Augmented Generation) 시스템입니다.
고품질의 데이터 큐레이션 파이프라인을 통해 학습 데이터를 생성하고, 이를 바탕으로 검색 모델을 미세 조정(Fine-tuning)하여 정확한 답변을 제공합니다.

## 🚀 주요 기능 (추론 파이프라인)

이 저장소는 **추론(Inference) 전용**입니다. 팀원들은 프롬프트를 수정하여 실험할 수 있습니다.

### 추론 파이프라인 (4단계)
0. **Rationale Generation (RAG^2)**: Generator 모델을 사용하여 모호한 질문을 검색에 적합한 의학적 근거로 변환.
1. **Retrieval (Top-50)**: Bi-Encoder로 빠르게 후보 문서 검색.
2. **Reranking (Top-3)**: 학습된 Cross-Encoder로 정밀 재순위화.
3. **Generation**: Qwen 2.5 (4-bit Quantized) LLM으로 최종 답변 생성.

### 프롬프트 수정
팀원들이 실험할 수 있는 주요 파일:
- **`src/module_augment.py`**: `build_prompt()` 함수를 수정하여 프롬프트 형식 변경
- **`config.yaml`**: 모델 설정, 파라미터 조정

## 💻 시스템 요구 사항

- **GPU**: VRAM 8GB 이상 (RTX 3060/4060 이상 권장)
- **RAM**: 16GB 이상
- **OS**: Windows/Linux (Windows의 경우 WSL2 권장)
- **Python**: 3.8 이상

## 📂 프로젝트 구조

```
catholic_retriver/
├── data/                       # 원본 데이터 (문서, 질의응답)
│   ├── TS_말뭉치데이터_내과/    # 문서 데이터 (JSON 형식)
│   └── Training/               # 질의응답 데이터 (ZIP 형식)
├── exper/                      # 실험 및 테스트 코드
├── vet_rag_project/            # 메인 프로젝트 디렉토리
│   ├── config.yaml             # 설정 파일 (모델, 파라미터 등)
│   ├── main_curation.py        # 추론 실행 스크립트
│   ├── requirements.txt        # 의존성 패키지 목록
│   ├── upload_model.py         # 모델 업로드 스크립트 (참고용)
│   └── src/
│       ├── data_loader.py      # 데이터 로딩 및 청킹
│       ├── embedding.py        # KmBERT 임베딩 모듈
│       ├── module_augment.py   # 프롬프트 생성 모듈 ⭐ (팀원들이 수정할 부분)
│       └── rag_pipeline.py     # 전체 추론 파이프라인 (Rationale->Retrieval->Rerank->Gen)
└── README.md                   # 프로젝트 설명서
```

## 📊 데이터 준비

### 문서 데이터
`data/TS_말뭉치데이터_내과/` 폴더에 다음 형식의 JSON 파일을 준비하세요:
```json
{
  "title": "문서 제목",
  "department": "내과",
  "disease": "본문 내용..."
}
```

### 질의응답 데이터
`data/Training/02.라벨링데이터/TL_질의응답데이터_내과.zip` 형식으로 준비하세요.
ZIP 파일 내부에는 다음 형식의 JSON 파일들이 포함되어야 합니다:
```json
{
  "question": "질문 내용",
  "answer": "답변 내용"
}
```

## 🛠️ 설치 및 실행

### 0. 환경 변수 설정 (필수)

#### OpenAI API 키 설정 (Answer Generation용)

1. **`.env` 파일 생성**:
   ```bash
   cd vet_rag_project
   # .env.example이 있다면 복사, 없다면 새로 생성
   ```

2. **`.env` 파일 편집**:
   ```bash
   # .env 파일 내용
   OPENAI_API_KEY=sk-your-openai-api-key-here
   WANDB_DISABLED=true
   ```
   
   > **참고**: OpenAI API 키는 https://platform.openai.com/api-keys 에서 발급받을 수 있습니다.

#### Hugging Face 토큰 설정 (모델 업로드용, 선택사항)

모델을 Hugging Face에 업로드하려면 토큰이 필요합니다. (팀원들은 토큰 없이 다운로드 가능)

1. **Hugging Face 토큰 발급**:
   - https://huggingface.co/settings/tokens 접속
   - "New token" 클릭 → **"Write"** 권한으로 생성 (업로드용)
   - 토큰 복사

2. **`.env` 파일에 추가** (선택사항):
   ```bash
   HF_TOKEN=hf_your_actual_token_here
   ```
   
   > **참고**: 모델 다운로드만 하는 경우 토큰이 필요 없습니다. (Public 리포지토리)

### 1. 환경 설정
```bash
cd vet_rag_project
pip install -r requirements.txt

# 4-bit 양자화를 위한 추가 설치 (Windows)
pip install bitsandbytes accelerate
```

### 2. 추론 (Inference)

#### 명령줄 모드
```bash
python main_curation.py --mode inference --query "강아지가 구토를 해요. 어떻게 해야 하나요?"
```

#### 대화형 모드
```bash
python main_curation.py --mode inference
```
- 질문을 입력하면 답변을 생성합니다.
- `quit` 또는 `exit`로 종료합니다.

**참고:**
- 첫 실행 시 Hugging Face에서 모델을 자동으로 다운로드합니다 (토큰 불필요).
- RTX 4060 등 일반 GPU에서도 구동 가능하도록 Qwen 모델을 4-bit로 로드합니다.
- 프롬프트 수정: `src/module_augment.py`의 `build_prompt()` 함수를 수정하세요.

## 📤 모델 공유 (Hugging Face)

### 모델 업로드 (작성자만 수행)

학습된 모델을 Hugging Face에 공개 리포지토리로 업로드하여 팀원들과 공유할 수 있습니다.

1. **`upload_model.py` 파일 수정**:
   ```python
   # upload_model.py 파일을 열어서 아래 설정을 수정하세요
   MY_TOKEN = "hf_your_actual_token_here"  # Hugging Face Write 토큰
   REPO_ID = "your-username/your-model-name"  # 예: "gildong/vet-rag-reranker"
   MODEL_DIR = "./results/bert_top25percent/final_model"  # 업로드할 모델 경로
   ```

2. **모델 업로드 실행**:
   ```bash
   python upload_model.py
   ```
   
   - 모델이 Hugging Face에 공개 리포지토리로 업로드됩니다.
   - 팀원들은 **토큰 없이** 모델을 다운로드할 수 있습니다.

3. **`config.yaml` 설정** (업로드 후):
   ```yaml
   training:
     huggingface_repo_id: "your-username/your-model-name"  # 주석 해제하고 리포지토리 이름 입력
   ```

### 모델 다운로드 및 사용 (팀원용)

팀원들은 코드를 실행하면 자동으로 Hugging Face에서 모델을 다운로드합니다.

1. **코드 클론 및 설치**:
   ```bash
   git clone <repository-url>
   cd vet_rag_project
   pip install -r requirements.txt
   ```

2. **환경 변수 설정** (`.env` 파일):
   ```bash
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```

3. **`config.yaml` 설정**:
   ```yaml
   training:
     huggingface_repo_id: "your-username/your-model-name"  # 업로드된 모델 리포지토리 이름
   ```

4. **코드 실행**:
   ```bash
   python main_curation.py --mode inference --query "강아지가 구토를 해요"
   ```
   
   - 첫 실행 시 Hugging Face에서 모델을 자동으로 다운로드합니다.
   - 이후 실행 시에는 캐시된 모델을 사용합니다.
   - **토큰 없이** 다운로드 가능합니다 (Public 리포지토리).

> **💡 장점**:
> - 팀원들은 Hugging Face 회원가입 불필요
> - 모델 업데이트 시 자동으로 새 버전 다운로드
> - 로컬 캐싱으로 빠른 재실행

## ⚙️ 설정 (config.yaml)

주요 설정 항목:

```yaml
# 검색 모델 설정
retrieval:
  model_name: "madatnlp/km-bert"
  top_k: 50

# 큐레이션용 LLM (가벼운 모델 권장)
llm_scorer:
  model_name: "gpt2"
  alpha_high_std: 0.7
  beta_low_std: 0.7

# 추론용 LLM (고성능 모델)
# Rationale Generation과 Answer Generation에 공통으로 사용됩니다.
llm:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  quantization: "bnb_4bit"
  gpu_memory_utilization: 0.90
  max_model_len: 4096
  dtype: "float16"

# Rationale Generation 설정 (RAG^2-style Query Enhancement)
rationale_gen:
  enabled: true  # false로 설정하면 Rationale Generation을 건너뜁니다
  temperature: 0.3  # 낮게 설정하여 일관성 있는 출력
  top_p: 0.9
  max_tokens: 256  # Rationale은 간결하게

# 학습 설정
training:
  output_dir: "./results/bert_top25percent"
  num_train_epochs: 3
  per_device_train_batch_size: 16
  learning_rate: 2e-5
  save_total_limit: 1
  # Hugging Face 모델 저장소 (선택사항)
  # huggingface_repo_id: "your-username/your-model-name"  # 모델 업로드 후 주석 해제
```

## 🔧 주요 기능 및 최적화

### GPU 메모리 최적화
- **순차적 모델 로딩**: 큐레이션 시 KmBERT → LLM Scorer → Cross-Encoder 순으로 로드/언로드
- **중간 결과 캐싱**: 임베딩, PPL 점수를 `cache/` 폴더에 저장하여 재사용
- **4-bit 양자화**: `bitsandbytes`를 사용하여 Qwen 7B 모델을 VRAM 8GB 환경에서 구동
- **모델 재사용**: Generator 모델(LLM)을 Rationale Generation과 Answer Generation에 공통으로 사용하여 메모리 효율성 향상

### 데이터 품질 관리
- **5단계 큐레이션**: Retrieval → LLM Scoring → Graph Refinement → Labeling → Validation
- **자동 라벨링**: 상위 5개 문서는 `label=1`, 나머지는 `label=0`으로 자동 할당
- **점수 기반 필터링**: PPL 점수와 그래프 전파 점수를 결합하여 최종 적합성 평가

## 📈 성능 및 결과

- **검색 정확도**: Cross-Encoder 재순위화를 통해 Top-3 정확도 대폭 향상
- **메모리 효율성**: RTX 4060 (8GB VRAM)에서 안정적으로 구동
- **추론 속도**: Bi-Encoder (빠른 검색) + Cross-Encoder (정밀 재순위화) 조합으로 최적화
- **RAG^2 통합**: Rationale Generation을 통해 모호한 질문을 검색에 적합한 쿼리로 자동 변환

## 🐛 문제 해결 (Troubleshooting)

### CUDA Out of Memory 에러
```bash
# config.yaml에서 batch_size 줄이기
per_device_train_batch_size: 8  # 16 → 8로 변경
```

### vLLM 설치 실패 (Windows)
- Windows에서는 `vllm` 설치가 어려울 수 있습니다.
- 자동으로 `transformers` + `bitsandbytes`로 fallback됩니다.

### 캐시 초기화
```bash
# cache 폴더 삭제 후 재실행
Remove-Item -Recurse -Force cache
python main_curation.py --mode curate
```

## 📝 라이선스

이 프로젝트는 연구 및 교육 목적으로 제작되었습니다.

## 🙏 감사의 말

- **KmBERT**: 한국어 임베딩 모델 제공
- **Qwen Team**: 고성능 한국어 LLM 제공
- **Hugging Face**: 모델 및 라이브러리 생태계 지원
#   c a t h o l i c _ r e t r e i v a l 
 
 
