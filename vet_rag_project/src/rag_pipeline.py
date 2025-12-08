import torch
import numpy as np
import os
import json
import yaml
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# Try importing OpenAI
try:
    from langchain_openai import ChatOpenAI
    USE_OPENAI = True
except ImportError:
    print("[WARNING] langchain_openai not found. Please install: pip install langchain-openai")
    USE_OPENAI = False

# Try importing vLLM
try:
    from vllm import LLM, SamplingParams
    USE_VLLM = True
except ImportError:
    print("[WARNING] vLLM not found. Falling back to Transformers.")
    USE_VLLM = False
    from transformers import AutoModelForCausalLM

from .embedding import KmBertEmbedder
from .module_augment import build_prompt
from .data_loader import load_documents_from_dirs

class VetRAGPipeline:
    def __init__(self, config_path="config.yaml", doc_dir=None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Initializing VetRAG Pipeline on {self.device}...")
        
        # Rationale Generation 설정 (RAG^2-style Query Enhancement)
        rationale_config = self.config.get("rationale_gen", {})
        self.use_rationale = rationale_config.get("enabled", True)
        
        # 1. Load Retrieval Model (Bi-Encoder)
        retrieval_model_name = self.config["retrieval"]["model_name"]
        print(f"[1/3] Loading Retrieval Model: {retrieval_model_name}")
        self.embedder = KmBertEmbedder(model_name=retrieval_model_name, device=self.device)
        
        # 2. Load Reranking Model (Cross-Encoder)
        rerank_model_path = f"{self.config['training']['output_dir']}/final_model"
        print(f"[2/3] Loading Reranking Model: {rerank_model_path}")
        try:
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_path).to(self.device)
            self.rerank_model.eval()
        except Exception as e:
            print(f"[WARNING] Failed to load reranking model from {rerank_model_path}. Using base model instead. Error: {e}")
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(retrieval_model_name)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(retrieval_model_name, num_labels=2).to(self.device)
        
        # 3-1. Load Answer Generation Model (LLM)
        llm_config = self.config.get("llm", {})
        llm_model_name = llm_config.get("model_name", "gpt2")
        print(f"[3/4] Loading Answer Generation Model: {llm_model_name}")
        
        # OpenAI API 모델인지 확인
        if llm_model_name.startswith("gpt-") or "openai" in llm_model_name.lower():
            if not USE_OPENAI:
                raise ImportError("OpenAI API를 사용하려면 langchain-openai가 필요합니다: pip install langchain-openai")
            self._init_openai_llm(llm_model_name, llm_config, is_answer_gen=True)
        elif USE_VLLM:
            try:
                self.llm = LLM(
                    model=llm_model_name,
                    quantization=llm_config.get("quantization"),
                    gpu_memory_utilization=llm_config.get("gpu_memory_utilization", 0.9),
                    max_model_len=llm_config.get("max_model_len", 4096),
                    dtype=llm_config.get("dtype", "float16"),
                    trust_remote_code=True
                )
                self.sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=512,
                    repetition_penalty=1.2
                )
                print("[SUCCESS] Answer Generation Model (vLLM) loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to load vLLM: {e}. Falling back to Transformers.")
                self.use_vllm_fallback(llm_model_name, is_answer_gen=True)
        else:
            self.use_vllm_fallback(llm_model_name, is_answer_gen=True)
        
        # 3-2. Rationale Generation Model 설정 (지연 로딩: 필요할 때만 로드)
        rationale_model_name = rationale_config.get("model_name", None)
        if rationale_model_name and self.use_rationale:
            print(f"[4/4] Rationale Generation Model: {rationale_model_name} (will be loaded on-demand)")
            self.rationale_model_name = rationale_model_name
            # OpenAI API 모델인 경우 별도 초기화 불필요 (지연 로딩)
            if rationale_model_name.startswith("gpt-") or "openai" in rationale_model_name.lower():
                self.rationale_llm = None  # 지연 로딩
            else:
                self.rationale_llm_tokenizer = None
                self.rationale_llm_model = None
        else:
            # Rationale 모델이 설정되지 않았거나 비활성화된 경우, Answer Generation 모델 재사용
            print(f"[4/4] Rationale Generation Model not specified. Using Answer Generation Model.")
            self.rationale_model_name = llm_model_name
            if hasattr(self, 'llm') and isinstance(self.llm, ChatOpenAI):
                self.rationale_llm = self.llm
            else:
                self.rationale_llm_tokenizer = self.llm_tokenizer
                self.rationale_llm_model = self.llm_model
        
        # Rationale Generation 프롬프트 템플릿 (키워드 확장 중심)
        self.rationale_prompt_template = rationale_config.get(
            "prompt_template",
            """[역할]
당신은 수의학 RAG 시스템의 검색 성능을 높이기 위한 '검색어 확장 전문가'입니다.

[입력 질문]
{query}

[목표]
- 위 질문에 대해, 수의학 논문/전문 서적에 등장할 법한 핵심 의학 키워드만 뽑아냅니다.
- 사용자의 일상 표현을, 문서에 등장할 수의학 전문 용어/관련 질환명/증상명/검사명으로 바꿉니다.

[출력 형식 규칙]
- 한국어 키워드만 사용합니다.
- 한글, 숫자, 공백만 사용하고 따옴표("), 화살표(->, →), 괄호(), 책 제목, 연도(2018, 2020 등)는 절대 쓰지 않습니다.
- 문장으로 쓰지 말고 키워드만 콤마(,)로 구분해 한 줄로 출력합니다.
- 3~8개의 짧은 키워드만 생성합니다.
- 예시처럼, 증상 + 관련 질환/장기/증상군 조합을 간단히 섞어 줍니다.

[예시]
- 질문: 강아지가 노란 토를 해요.
- 가능한 출력: 구토, 공복토, 담즙 역류, 위장관 질환, 소화기 질환
- 잘못된 출력: 강아지가 노란 토를 합니다, 동물병원의사들이추천하는반려견건강백서 2018

[최종 출력]
위 규칙을 만족하는 키워드 목록만 콤마로 구분해 출력하세요.
키워드:"""
        )
        
        # 4. Load Documents & Embeddings
        self.documents = []
        self.doc_embeddings = None
        
        kb_dirs = None
        if doc_dir is not None:
            kb_dirs = doc_dir
        else:
            kb_dirs = self.config.get("knowledge_base", {}).get("directories")
        if kb_dirs:
            print(f"[INFO] Starting to load knowledge base from {len(kb_dirs) if isinstance(kb_dirs, list) else 1} directory(ies)...")
            self.load_knowledge_base(kb_dirs)
            print(f"[SUCCESS] Knowledge base loaded: {len(self.documents)} documents.")
        else:
            print("[WARNING] No knowledge base directories provided. Retrieval will not work until documents are loaded.")
    
    def _get_openai_api_key(self) -> Optional[str]:
        """OpenAI API 키를 환경변수나 .env 파일에서 가져옵니다."""
        # 1차: 환경변수
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("TOKEN")
        if api_key:
            return api_key
        
        # 2차: .env 파일에서 직접 파싱
        env_paths = [
            Path(__file__).resolve().parent.parent / ".env",
            Path.cwd() / ".env",
        ]
        for env_path in env_paths:
            if env_path.exists():
                try:
                    text = env_path.read_text(encoding="utf-8")
                    for line in text.splitlines():
                        stripped = line.strip().lstrip("\ufeff")
                        if stripped.startswith("#") or not stripped:
                            continue
                        if stripped.upper().startswith("TOKEN") and "=" in stripped:
                            api_key = stripped.split("=", 1)[1].strip().strip('"').strip("'")
                            if api_key:
                                return api_key
                except Exception as e:
                    print(f"[WARNING] Failed to read .env from {env_path}: {e}")
        
        return None
    
    def _init_openai_llm(self, model_name: str, config: Dict, is_answer_gen: bool = True):
        """OpenAI API를 사용하는 LLM을 초기화합니다."""
        api_key = self._get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API 키를 찾을 수 없습니다. OPENAI_API_KEY 또는 TOKEN 환경변수를 설정하거나 .env 파일에 TOKEN=... 을 추가하세요.")
        
        temperature = config.get("temperature", 0.7 if is_answer_gen else 0.1)
        max_tokens = config.get("max_tokens", 512 if is_answer_gen else 128)
        
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        
        if is_answer_gen:
            self.llm = llm
            print(f"[SUCCESS] Answer Generation Model (OpenAI API: {model_name}) initialized.")
        else:
            self.rationale_llm = llm
            print(f"[SUCCESS] Rationale Generation Model (OpenAI API: {model_name}) initialized.")
            
    def use_vllm_fallback(self, model_name, is_answer_gen=True):
        """
        Transformers로 모델 로드 (vLLM 실패 시 또는 의료 특화 모델용)
        
        Args:
            model_name: 모델 이름
            is_answer_gen: True면 Answer Generation용, False면 Rationale Generation용
        """
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        print(f"[INFO] Loading {model_name} with BitsAndBytes 4-bit quantization...")
        
        # Answer 모델이 이미 해제되었으므로 GPU에 충분한 공간이 있음
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        device_map = "auto"  # GPU에 로드
        
        print(f"[INFO] Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"[INFO] Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            device_map=device_map,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )
        model.eval()
        print(f"[INFO] Model loaded and set to eval mode.")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if is_answer_gen:
            self.llm_tokenizer = tokenizer
            self.llm_model = model
            print(f"[SUCCESS] Answer Generation model ready.")
        else:
            self.rationale_llm_tokenizer = tokenizer
            self.rationale_llm_model = model
            print(f"[SUCCESS] Rationale Generation model ready.")
            
    def load_knowledge_base(self, doc_dirs):
        if isinstance(doc_dirs, str):
            doc_dirs = [doc_dirs]
        print(f"[INFO] Loading documents from {len(doc_dirs)} directory(ies)...")
        self.documents = load_documents_from_dirs(doc_dirs)
        
        # Load or compute embeddings
        cache_path = "cache/doc_embeddings.npy"
        if os.path.exists(cache_path):
            print("[INFO] Loading cached document embeddings...")
            self.doc_embeddings = np.load(cache_path)
            
            # Verify shape match
            if len(self.documents) != self.doc_embeddings.shape[0]:
                print(f"[WARNING] Document count ({len(self.documents)}) does not match embedding count ({self.doc_embeddings.shape[0]}). Recomputing...")
                self.doc_embeddings = self.embedder.encode(self.documents)
                np.save(cache_path, self.doc_embeddings)
        else:
            print("[INFO] Computing document embeddings (this may take a while)...")
            self.doc_embeddings = self.embedder.encode(self.documents)
            os.makedirs("cache", exist_ok=True)
            np.save(cache_path, self.doc_embeddings)
            
    def _load_rationale_model(self):
        """
        Rationale Generation 모델을 지연 로딩합니다.
        필요할 때만 로드하고 사용 후 메모리에서 해제합니다.
        OpenAI API 모델인 경우 별도 해제 불필요.
        """
        # OpenAI API 모델인 경우
        if self.rationale_model_name and (self.rationale_model_name.startswith("gpt-") or "openai" in self.rationale_model_name.lower()):
            if self.rationale_llm is None:
                rationale_config = self.config.get("rationale_gen", {})
                self._init_openai_llm(self.rationale_model_name, rationale_config, is_answer_gen=False)
            return
        
        # 로컬 모델인 경우
        if hasattr(self, 'rationale_llm_model') and self.rationale_llm_model is not None:
            return  # 이미 로드됨
        
        if not self.rationale_model_name or self.rationale_model_name == self.config.get("llm", {}).get("model_name"):
            # Answer Generation 모델 재사용
            if hasattr(self, 'llm') and isinstance(self.llm, ChatOpenAI):
                self.rationale_llm = self.llm
            else:
                self.rationale_llm_tokenizer = self.llm_tokenizer
                self.rationale_llm_model = self.llm_model
            return
        
        # Answer 모델을 먼저 메모리에서 해제 (GPU 메모리 확보)
        print(f"[INFO] Unloading Answer Generation model to free GPU memory...")
        if hasattr(self, 'llm_model') and self.llm_model is not None:
            del self.llm_model
            del self.llm_tokenizer
            self.llm_model = None
            self.llm_tokenizer = None
            torch.cuda.empty_cache()
            print(f"[SUCCESS] Answer Generation model unloaded.")
        
        print(f"[INFO] Loading Rationale Generation Model: {self.rationale_model_name} (on-demand)...")
        self.use_vllm_fallback(self.rationale_model_name, is_answer_gen=False)
        print(f"[SUCCESS] Rationale Generation model loaded.")
    
    def _unload_rationale_model(self):
        """
        Rationale Generation 모델을 메모리에서 해제합니다.
        Answer 모델 재로딩은 generate 메서드에서 필요할 때 수행합니다.
        OpenAI API 모델인 경우 해제 불필요.
        """
        # OpenAI API 모델인 경우 해제 불필요
        if hasattr(self, 'rationale_llm') and isinstance(self.rationale_llm, ChatOpenAI):
            return
        
        if hasattr(self, 'rationale_llm_model') and self.rationale_llm_model is not None:
            if not hasattr(self, 'llm_model') or self.rationale_llm_model != self.llm_model:
                print(f"[INFO] Unloading Rationale Generation model from GPU memory...")
                del self.rationale_llm_model
                del self.rationale_llm_tokenizer
                self.rationale_llm_model = None
                self.rationale_llm_tokenizer = None
                # 강력한 메모리 정리
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[SUCCESS] Rationale Generation model unloaded.")
    
    def generate_rationale(self, query: str) -> str:
        """
        RAG^2-style Rationale Generation (Query Expansion)
        의료 특화 모델을 사용하여 키워드 확장: 원본 질문 + 의학 전문 용어 키워드
        모델은 필요할 때만 로드하고 사용 후 해제합니다.
        """
        prompt = self.rationale_prompt_template.format(query=query)
        
        # Rationale 모델 지연 로딩
        self._load_rationale_model()
        
        try:
            # OpenAI API 모델인 경우
            if hasattr(self, 'rationale_llm') and isinstance(self.rationale_llm, ChatOpenAI):
                messages = [{"role": "user", "content": prompt}]
                response = self.rationale_llm.invoke(messages)
                rationale_text = response.content.strip()
            else:
                # 로컬 모델인 경우
                rationale_tokenizer = self.rationale_llm_tokenizer
                rationale_model = self.rationale_llm_model
                
                inputs = rationale_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                ).to(self.device)
                
                input_length = inputs.input_ids.shape[1]
                
                with torch.no_grad():
                    outputs = rationale_model.generate(
                        **inputs,
                        max_new_tokens=self.config.get("rationale_gen", {}).get("max_tokens", 128),
                        do_sample=True,
                        temperature=self.config.get("rationale_gen", {}).get("temperature", 0.1),
                        top_p=self.config.get("rationale_gen", {}).get("top_p", 0.9),
                        repetition_penalty=1.2,
                        pad_token_id=rationale_tokenizer.eos_token_id,
                        eos_token_id=rationale_tokenizer.eos_token_id
                    )
                
                # 생성된 부분만 추출 (입력 프롬프트 제외)
                generated_ids = outputs[0][input_length:]
                rationale_text = rationale_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Rationale 모델 해제 (메모리 절약)
                if hasattr(self, 'llm_model') and rationale_model != self.llm_model:
                    self._unload_rationale_model()
            
            # 키워드 파싱: 콤마로 구분된 키워드 추출
            keywords = []
            if rationale_text:
                # "키워드:" 같은 접두사 제거
                rationale_text = rationale_text.split(":", 1)[-1].strip()
                # 줄바꿈/비표준 공백 제거
                rationale_text = rationale_text.replace("\n", " ").replace("\t", " ")
                # 콤마로 분리하고 공백 제거
                raw_keywords = [kw.strip() for kw in rationale_text.split(",") if kw.strip()]
                cleaned_keywords = []
                for kw in raw_keywords:
                    # 따옴표, 화살표, 괄호, 특수문자 제거
                    for ch in ['"', "'", "“", "”", "‘", "’", "→", "->", "(", ")", "[", "]"]:
                        kw = kw.replace(ch, " ")
                    kw = kw.strip()
                    # 너무 긴 항목 제거 (문장이 아닌 키워드만)
                    if not kw or len(kw) > 25:
                        continue
                    cleaned_keywords.append(kw)
                keywords = cleaned_keywords
            
            # 빈 응답 체크 또는 키워드가 없으면 원본 질문만 사용
            if not keywords:
                print(f"[WARNING] [Rationale] No keywords extracted. Using original query only.")
                return query
            
            # Query Expansion: 원본 질문 + 키워드 결합
            expanded_query = f"{query} [SEP] {', '.join(keywords)}"
            print(f"[INFO] [Rationale] Extracted keywords: {keywords}")

            # 나중에 분석/저장을 위해 마지막 Rationale 정보 보관
            self.last_rationale = {
                "original_query": query,
                "expanded_query": expanded_query,
                "keywords": keywords,
            }
            return expanded_query
            
        except Exception as e:
            print(f"[WARNING] [Rationale] Error during generation: {e}. Using original query.")
            # 에러 발생 시에도 모델 해제 (로컬 모델인 경우만)
            if hasattr(self, 'rationale_llm_model') and hasattr(self, 'llm_model'):
                if self.rationale_llm_model != self.llm_model:
                    self._unload_rationale_model()
            # 실패 시에도 최소 정보는 기록
            self.last_rationale = {
                "original_query": query,
                "expanded_query": query,
                "keywords": [],
            }
            return query
    
    def retrieve(self, query: str, top_n: int = 100, use_rationale: bool = None) -> List[Dict]:
        """
        문서 검색 수행
        
        Args:
            query: 검색 쿼리
            top_n: 반환할 상위 문서 수
            use_rationale: Rationale 사용 여부 (None이면 config 설정 따름)
        """
        # Step 0: Rationale Generation (RAG^2-style)
        if use_rationale is None:
            use_rationale = self.use_rationale
        
        if use_rationale:
            print(f"[INFO] [Rationale] Original Query: {query}")
            expanded_query = self.generate_rationale(query)
            print(f"[INFO] [Rationale] Expanded Query: {expanded_query[:150]}...")
            search_query = expanded_query
        else:
            search_query = query
        
        # Step 1: Retrieval (코사인 유사도 사용)
        q_emb = self.embedder.encode(search_query)
        
        # 임베딩 정규화 (L2 norm)
        q_emb_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8)
        doc_embs_norm = self.doc_embeddings / (np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # 코사인 유사도 계산 (정규화된 벡터의 내적 = 코사인 유사도)
        sims = np.dot(doc_embs_norm, q_emb_norm.T).flatten()
        top_indices = np.argsort(sims)[::-1][:top_n]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                "chunk_id": int(idx),
                "chunk_text": self.documents[idx],
                "score": float(sims[idx])  # 이제 -1~1 범위의 코사인 유사도
            })
        return candidates
        
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5, min_score: float = 0.5) -> List[Dict]:
        """
        Rerank candidates using cross-encoder model.
        
        Args:
            query: 검색 쿼리
            candidates: 검색된 후보 문서 리스트
            top_k: 반환할 상위 문서 수
            min_score: 최소 reranking 점수 (이 점수 미만은 제외)
        """
        if not candidates:
            return []
        
        # Step 2: Reranking
        pairs = [[query, doc["chunk_text"]] for doc in candidates]
        
        inputs = self.rerank_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.rerank_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1].cpu().numpy()
            
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])
        
        # 점수 기준으로 정렬하고 최소 점수 이상만 필터링
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        reranked = [doc for doc in reranked if doc["rerank_score"] >= min_score]
        
        # 상위 k개만 반환
        return reranked[:top_k]
        
    def _ensure_answer_model_loaded(self):
        """
        Answer Generation 모델이 로드되어 있는지 확인하고, 없으면 다시 로드합니다.
        """
        llm_config = self.config.get("llm", {})
        llm_model_name = llm_config.get("model_name", "gpt2")
        
        # OpenAI API 모델인 경우
        if llm_model_name.startswith("gpt-") or "openai" in llm_model_name.lower():
            if not hasattr(self, 'llm') or self.llm is None:
                print(f"[INFO] Answer Generation model not loaded. Initializing OpenAI API...")
                self._init_openai_llm(llm_model_name, llm_config, is_answer_gen=True)
            return
        
        # 로컬 모델인 경우
        if not hasattr(self, 'llm_model') or self.llm_model is None or (hasattr(self, 'llm_tokenizer') and self.llm_tokenizer is None):
            print(f"[INFO] Answer Generation model not loaded. Reloading...")
            # 추가 메모리 정리
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            self.use_vllm_fallback(llm_model_name, is_answer_gen=True)
            print(f"[SUCCESS] Answer Generation model reloaded.")
    
    def generate(self, query: str, context_docs: List[Dict]) -> str:
        # Step 3: Generation
        # Answer 모델이 로드되어 있는지 확인
        self._ensure_answer_model_loaded()
        
        prompt = build_prompt(query, context_docs)
        
        # OpenAI API 모델인 경우
        if hasattr(self, 'llm') and isinstance(self.llm, ChatOpenAI):
            # build_prompt는 이제 시스템 역할 없이 순수 작업 지시만 반환
            # 시스템 프롬프트는 여기서 정의
            system_content = (
                "당신은 반려동물 관련 의학 정보를 전문적으로 설명하는 수의사 AI입니다.\n"
                "\n"
                "【핵심 규칙】\n"
                "1. 답변은 반드시 100% 한국어로만 작성하세요.\n"
                "2. 제공된 [참고 자료]의 내용만을 근거로 답변하세요.\n"
                "3. [참고 자료]에 없는 의학 지식이나 약물 정보를 추측하거나 만들어내지 마세요.\n"
                "4. 정보가 불충분할 경우 '제공된 정보만으로는 판단하기 어렵습니다'라고 명시하세요.\n"
                "5. 전문 용어는 보호자가 이해하기 쉽게 풀어서 설명하세요.\n"
                "\n"
                "【답변 형식】\n"
                "사용자가 요청한 답변 형식(핵심 진단/평가, 추가 조치, 주의사항, 근거 요약)을 반드시 준수하세요."
            )
            # build_prompt의 결과는 [참고 자료] + [질문] + [답변 지침]
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            response = self.llm.invoke(messages)
            return response.content.strip()
        elif USE_VLLM and hasattr(self, 'llm') and not isinstance(self.llm, ChatOpenAI):
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()
            return response
        else:
            # Llama2/Qwen 모델의 경우 채팅 템플릿 사용 시도
            try:
                # Qwen2.5 채팅 템플릿 사용
                if hasattr(self.llm_tokenizer, 'apply_chat_template'):
                    system_content = (
                        "당신은 반려동물 관련 의학 정보를 전문적으로 설명하는 수의사 AI입니다.\n"
                        "\n"
                        "【핵심 규칙】\n"
                        "1. 답변은 반드시 100% 한국어로만 작성하세요.\n"
                        "2. 제공된 [참고 자료]의 내용만을 근거로 답변하세요.\n"
                        "3. [참고 자료]에 없는 의학 지식이나 약물 정보를 추측하거나 만들어내지 마세요.\n"
                        "4. 정보가 불충분할 경우 '제공된 정보만으로는 판단하기 어렵습니다'라고 명시하세요.\n"
                        "5. 전문 용어는 보호자가 이해하기 쉽게 풀어서 설명하세요.\n"
                        "\n"
                        "【답변 형식】\n"
                        "사용자가 요청한 답변 형식(핵심 진단/평가, 추가 조치, 주의사항, 근거 요약)을 반드시 준수하세요."
                    )
                    messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt}
                    ]
                    formatted_prompt = self.llm_tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # 채팅 템플릿이 없는 경우
                    system_msg = (
                        "당신은 반려동물 관련 의학 정보를 전문적으로 설명하는 수의사 AI입니다.\n"
                        "\n"
                        "【핵심 규칙】\n"
                        "1. 답변은 반드시 100% 한국어로만 작성하세요.\n"
                        "2. 제공된 [참고 자료]의 내용만을 근거로 답변하세요.\n"
                        "3. [참고 자료]에 없는 의학 지식이나 약물 정보를 추측하거나 만들어내지 마세요.\n"
                        "4. 정보가 불충분할 경우 '제공된 정보만으로는 판단하기 어렵습니다'라고 명시하세요.\n"
                        "5. 전문 용어는 보호자가 이해하기 쉽게 풀어서 설명하세요.\n"
                        "\n"
                        "【답변 형식】\n"
                        "사용자가 요청한 답변 형식(핵심 진단/평가, 추가 조치, 주의사항, 근거 요약)을 반드시 준수하세요."
                    )
                    formatted_prompt = f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{prompt} [/INST]"
            except Exception as e:
                print(f"[WARNING] Failed to apply chat template: {e}. Using raw prompt.")
                formatted_prompt = prompt
            
            inputs = self.llm_tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            input_length = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.2,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id
                )
            
            # 생성된 부분만 추출 (입력 프롬프트 제외)
            generated_ids = outputs[0][input_length:]
            response = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # 추가 정리: 프롬프트가 여전히 포함되어 있으면 제거
            # 새 프롬프트 형식에 맞춰 필터링 업데이트
            if "### [참고 자료" in response or "### [사용자 질문" in response:
                # 답변 내용만 추출 (프롬프트 지침 제거)
                if "### [답변 작성 지침]" in response:
                    response = response.split("### [답변 작성 지침]", 1)[0].strip()
                # 시스템 메시지가 포함되어 있으면 제거
                if "당신은 반려동물" in response:
                    lines = response.split("\n")
                    filtered_lines = []
                    for line in lines:
                        if "당신은" not in line and "반려동물" not in line and "【" not in line:
                            filtered_lines.append(line)
                    response = "\n".join(filtered_lines).strip()
            
            return response

    def run(self, query: str, use_rationale: bool = None) -> str:
        """
        전체 RAG 파이프라인 실행
        
        Args:
            query: 사용자 질문
            use_rationale: Rationale 사용 여부 (None이면 config 설정 따름)
        """
        print(f"\n[INFO] Original Query: {query}")
        
        # 0. Rationale Generation (RAG^2-style)
        if use_rationale is None:
            use_rationale = self.use_rationale
        
        if use_rationale:
            print(f"\n{'='*60}")
            print("STEP 0: Rationale Generation (RAG^2)")
            print(f"{'='*60}")
            refined_query = self.generate_rationale(query)
            print(f"[SUCCESS] Rationale generated. Using refined query for retrieval.")
        else:
            refined_query = query
        
        # 1. Retrieve
        print(f"\n{'='*60}")
        print("STEP 1: Retrieval")
        print(f"{'='*60}")
        candidates = self.retrieve(query, top_n=50, use_rationale=use_rationale)
        print(f"[SUCCESS] Retrieved {len(candidates)} candidates.")
        
        # 2. Rerank
        print(f"\n{'='*60}")
        print("STEP 2: Reranking")
        print(f"{'='*60}")
        top_docs = self.rerank(query, candidates, top_k=3)
        print(f"[SUCCESS] Reranked top {len(top_docs)} documents.")
        for i, doc in enumerate(top_docs):
            print(f"   [{i+1}] Score: {doc['rerank_score']:.4f} | {doc['chunk_text'][:50]}...")
            
        # 3. Generate
        print(f"\n{'='*60}")
        print("STEP 3: Answer Generation")
        print(f"{'='*60}")
        answer = self.generate(query, top_docs)
        
        print("\n" + "="*60)
        print("[INFO] Final Answer:")
        print("="*60)
        print(answer)
        print("="*60)
        return answer
