# -*- coding: utf-8 -*-
"""
[파일 역할]
이 스크립트는 RAG (Retrieval-Augmented Generation) 파이프라인의 '인덱싱' 단계를 수행합니다.
'./data' 폴더 내의 모든 JSON 파일들을 읽어와 텍스트를 추출하고,
지정된 방법(ChunkingMethod)에 따라 텍스트를 의미 있는 단위(청크)로 분할합니다.
분할된 각 청크는 임베딩 모델을 통해 벡터로 변환되며,
최종적으로 FAISS 벡터 인덱스와 각 청크의 정보를 담은 청크맵 파일로 저장됩니다.

[실행 방법]
1. 이 스크립트를 실행하기 전에 `requirements.txt`의 모든 라이브러리를 설치해야 합니다.
   $ pip install -r requirements.txt
2. 스크립트 하단의 `Config` 클래스에서 원하는 청킹 방법, 모델 등을 설정할 수 있습니다.
3. 터미널에서 아래 명령어를 실행하여 인덱싱을 시작합니다.
   $ python vet_rag_project/build_index.py

[생성 파일]
- db/vector_store.index: FAISS 벡터 데이터베이스 파일
- db/chunks_map.json: 각 벡터에 해당하는 원본 텍스트와 메타데이터를 담은 JSON 파일
"""
import os
import sys
import json
import glob
from enum import Enum
from pathlib import Path
from typing import List, Dict

import torch

# 프로젝트 루트를 sys.path에 추가하여 다른 모듈(exper)을 임포트할 수 있도록 함
sys.path.append(str(Path(__file__).resolve().parent.parent))

import faiss
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# --- LangChain Splitters ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# --- 로컬 청킹 모듈 ---
from exper.chunker_kss_sentence_grouping import group_sentences_by_char_limit


# ==================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 설정 (CONFIG) ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==================================================
class ChunkingMethod(Enum):
    """사용할 청킹 방법을 정의합니다."""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    KSS_GROUPING = "kss_grouping"


class Config:
    """스크립트의 모든 설정을 관리합니다."""
    # --- 1. 청킹 방법 선택 ---
    # ChunkingMethod.RECURSIVE: 고정 크기로 자르는 가장 기본적인 방법
    # ChunkingMethod.KSS_GROUPING: 문장 단위로 자른 뒤, 글자 수 제한에 맞춰 문장 그룹으로 묶는 방법
    # ChunkingMethod.SEMANTIC: 의미적 유사도를 기반으로 청크를 나누는 방법 (OpenAI API 키 필요)
    CHUNKING_METHOD: ChunkingMethod = ChunkingMethod.RECURSIVE

    # --- 2. 청킹 세부 설정 ---
    RECURSIVE_CHUNK_SIZE: int = 1200
    RECURSIVE_CHUNK_OVERLAP: int = 200
    KSS_GROUPING_MAX_CHARS: int = 800
    SEMANTIC_BREAKPOINT_THRESHOLD: int = 95

    # --- 3. 임베딩 모델 설정 ---
    # BAAI/bge-m3: 한국어 성능이 우수하고 긴 문맥을 잘 처리하는 공개 모델
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    # RTX 4060 8GB 환경에서 OOM과 과도한 지연을 피하기 위해 배치 크기를 줄인다.
    EMBEDDING_BATCH_SIZE: int = 32

    # --- 4. 데이터 경로 설정 ---
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DB_DIR: Path = BASE_DIR / "vet_rag_project" / "db"
    
    # 최종 결과물 경로
    INDEX_PATH: Path = DB_DIR / "vector_store.index"
    MAP_PATH: Path = DB_DIR / "chunks_map.json"

# ==================================================
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 설정 (CONFIG) ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ==================================================


def load_documents_from_json(root_dir: Path) -> List[Document]:
    """JSON 파일들을 읽어 LangChain의 Document 객체 리스트를 생성합니다."""
    print(f"[LOAD] '{root_dir}' 디렉토리에서 데이터 로드를 시작합니다.")
    json_files = glob.glob(str(root_dir / '**' / '*.json'), recursive=True)
    documents = []
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
                page_content = data.get('disease')
                if page_content and isinstance(page_content, str):
                    metadata = {
                        'source': str(Path(file_path).relative_to(root_dir.parent)),
                        'title': data.get('title', 'N/A'),
                        'author': data.get('author', 'N/A'),
                        'publisher': data.get('publisher', 'N/A'),
                        'department': data.get('department', 'N/A')
                    }
                    documents.append(Document(page_content=page_content, metadata=metadata))
        except Exception as e:
            print(f"  - 파일 처리 중 오류 발생 {file_path}: {e}")
    print(f"  └ 총 {len(documents)}개의 문서를 로드했습니다.")
    return documents


def chunk_documents(docs: List[Document], config: Config) -> List[Document]:
    """설정에 따라 적절한 청킹 메소드를 호출하여 문서를 분할합니다."""
    print(f"[CHUNK] '{config.CHUNKING_METHOD.value}' 방법으로 청킹을 시작합니다.")
    
    if config.CHUNKING_METHOD == ChunkingMethod.RECURSIVE:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n", "\n\n", "\n", ". ", " "],
            chunk_size=config.RECURSIVE_CHUNK_SIZE,
            chunk_overlap=config.RECURSIVE_CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = splitter.split_documents(docs)

    elif config.CHUNKING_METHOD == ChunkingMethod.KSS_GROUPING:
        chunks = group_sentences_by_char_limit(docs, max_chars=config.KSS_GROUPING_MAX_CHARS)

    elif config.CHUNKING_METHOD == ChunkingMethod.SEMANTIC:
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Semantic Chunking을 사용하려면 OPENAI_API_KEY 환경변수가 필요합니다.")
        
        def _clean_text(text: str) -> str:
            lines = text.split('\n')
            cleaned_lines = [line.strip() for line in lines if line.strip() and not line.strip().isdigit()]
            return '\n'.join(cleaned_lines)

        cleaned_docs = [Document(page_content=_clean_text(doc.page_content), metadata=doc.metadata) for doc in docs]
        
        embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        splitter = SemanticChunker(
            embeddings=embed_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=config.SEMANTIC_BREAKPOINT_THRESHOLD
        )
        chunks = splitter.split_documents(cleaned_docs)

    else:
        raise NotImplementedError(f"지원하지 않는 청킹 방법입니다: {config.CHUNKING_METHOD}")
        
    print(f"  └ 총 {len(chunks)}개의 청크가 생성되었습니다.")
    return chunks


def create_and_save_index(chunks: List[Document], config: Config):
    """텍스트 청크를 임베딩하고 FAISS 인덱스와 청크맵을 파일로 저장합니다."""
    if not chunks:
        raise ValueError("청킹된 데이터가 비어 있어 인덱스를 생성할 수 없습니다.")

    config.DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[EMBED] 임베딩 모델 로드: {config.EMBEDDING_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    
    texts_to_embed = [c.page_content for c in chunks]
    print(f"  - 임베딩 시작 (문서 개수: {len(texts_to_embed)}, 배치 크기: {config.EMBEDDING_BATCH_SIZE})")
    
    try:
        embeddings = model.encode(
            texts_to_embed,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n[ERROR] 임베딩 과정이 사용자의 중단(KeyboardInterrupt)으로 종료되었습니다.")
        raise
    
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    print(f"  - 임베딩 완료 (벡터 차원: {dim})")

    print(f"[SAVE] FAISS 인덱스 및 청크맵 저장을 시작합니다.")
    # FAISS 인덱스 생성 및 저장
    index = faiss.IndexFlatIP(dim)  # 내적(Inner Product)을 유사도 척도로 사용
    index.add(embeddings)
    faiss.write_index(index, str(config.INDEX_PATH))
    print(f"  - FAISS 인덱스 저장 완료: {config.INDEX_PATH}")

    # 청크맵(chunk_id -> chunk) 저장
    chunk_map = []
    for i, chunk in enumerate(chunks):
        chunk_map.append({
            "chunk_id": i,
            "chunk_text": chunk.page_content,
            "metadata": chunk.metadata
        })

    with open(config.MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=4)
    print(f"  - 청크맵 저장 완료: {config.MAP_PATH}")


def main():
    """전체 인덱싱 파이프라인을 실행합니다."""
    print("="*50)
    print("인덱스 빌드 파이프라인을 시작합니다.")
    print(f"선택된 청킹 방법: {Config.CHUNKING_METHOD.value}")
    print("="*50)

    # 1. 데이터 로드
    documents = load_documents_from_json(Config.DATA_DIR)
    
    # 2. 청킹
    chunked_documents = chunk_documents(documents, Config)
    
    # 3. 임베딩 및 저장
    create_and_save_index(chunked_documents, Config)
    
    print("="*50)
    print("모든 작업이 성공적으로 완료되었습니다!")
    print(f"  - 인덱스 파일: {Config.INDEX_PATH}")
    print(f"  - 청크맵 파일: {Config.MAP_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()
