# -*- coding: utf-8 -*-
"""
[파일 역할]
이 스크립트는 `build_index.py`를 통해 생성된 벡터 DB를 사용하여,
실시간 사용자 쿼리에 대한 답변을 생성하는 RAG(Retrieval-Augmented Generation) 애플리케이션입니다.
전체 파이프라인은 아래와 같은 순서로 진행됩니다.
1. 사용자 쿼리 입력
2. 쿼리 재작성 (현재는 원본 유지)
3. 벡터 DB 검색 (Retrieval)
4. 프롬프트 증강 (Augmentation)
5. LLM(GPT) 호출 및 답변 생성

[실행 방법]
1. `build_index.py`를 먼저 실행하여 `db/vector_store.index`와 `db/chunks_map.json`을 생성해야 합니다.
2. 프로젝트 루트 디렉토리(`.env` 파일이 위치할 곳)에 `.env` 파일을 만들고 아래와 같이 OpenAI API 키를 추가합니다.
   OPENAI_API_KEY="sk-..."
3. 터미널에서 아래 명령어를 실행합니다.
   $ python vet_rag_project/app.py
"""
import os
import json
import faiss
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# ==================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 설정 (CONFIG) ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==================================================
class Config:
    """스크립트의 모든 설정을 관리합니다."""
    # --- 1. 경로 설정 ---
    BASE_DIR = Path(__file__).resolve().parent
    DB_DIR: Path = BASE_DIR / "db"
    INDEX_PATH: Path = DB_DIR / "vector_store.index"
    MAP_PATH: Path = DB_DIR / "chunks_map.json"

    # --- 2. 모델 설정 ---
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    LLM_MODEL: str = "gpt-4o-mini"  # OpenAI의 LLM 모델

    # --- 3. 검색 설정 ---
    SEARCH_TOP_K: int = 3  # 검색 결과로 가져올 청크 개수

# ==================================================
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 설정 (CONFIG) ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ==================================================

class RAGPipeline:
    """
    RAG 파이프라인의 전체 과정을 관리하는 클래스.
    """
    def __init__(self, config: Config):
        self.config = config
        print("[INFO] RAG 파이프라인을 초기화합니다...")
        
        # .env 파일에서 환경 변수 로드
        load_dotenv()
        
        # OpenAI API 키 확인 및 클라이언트 초기화
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
        self.llm_client = OpenAI()
        print("[INFO] OpenAI 클라이언트 초기화 완료.")

        # DB 파일 존재 여부 확인
        if not self.config.INDEX_PATH.exists() or not self.config.MAP_PATH.exists():
            raise FileNotFoundError(
                f"DB 파일({self.config.INDEX_PATH} 또는 {self.config.MAP_PATH})을 찾을 수 없습니다. "
                f"`build_index.py`를 먼저 실행하여 DB를 생성하세요."
            )
            
        # 임베딩 모델 로드
        self.embed_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        print(f"[INFO] 임베딩 모델 로드 완료: {self.config.EMBEDDING_MODEL}")

        # FAISS 인덱스 로드
        self.index = faiss.read_index(str(self.config.INDEX_PATH))
        print(f"[INFO] FAISS 인덱스 로드 완료: {self.config.INDEX_PATH}")

        # 청크맵 로드
        with open(self.config.MAP_PATH, "r", encoding="utf-8") as f:
            self.chunk_map = json.load(f)
        print(f"[INFO] 청크맵 로드 완료: {self.config.MAP_PATH}")

    def _rewrite_query(self, user_query: str) -> str:
        """
        사용자 쿼리를 재작성합니다. (향후 확장 포인트)
        """
        print(f"\n[STEP 1] 쿼리 재작성 (현재는 원본 통과)")
        rewritten_query = user_query
        print(f"  - 원본/결과: {rewritten_query}")
        return rewritten_query

    def _search(self, query: str) -> List[Dict]:
        """
        쿼리를 사용하여 DB에서 관련 문서를 검색합니다.
        """
        print(f"\n[STEP 2] 벡터 DB 검색")
        query_embedding = self.embed_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        
        distances, indices = self.index.search(query_embedding, self.config.SEARCH_TOP_K)
        
        retrieved_chunks = [self.chunk_map[i] for i in indices[0]]
        print(f"  - 검색된 청크 ID: {indices[0]}")
        return retrieved_chunks

    def _augment_prompt(self, user_query: str, context_chunks: List[Dict]) -> str:
        """
        검색된 컨텍스트와 메타데이터를 사용하여 LLM에 보낼 프롬프트를 구성합니다.
        """
        print(f"\n[STEP 3] 프롬프트 증강 (검색된 내용 + 메타데이터 포함)")
        
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            metadata = chunk['metadata']
            context_text = (
                f"--- 컨텍스트 {i+1} ---\n"
                f"출처: {metadata.get('source', 'N/A')}\n"
                f"제목: {metadata.get('title', 'N/A')}\n"
                f"진료과: {metadata.get('department', 'N/A')}\n"
                f"내용: {chunk['chunk_text']}"
            )
            context_parts.append(context_text)
        
        context_str = "\n\n".join(context_parts)
        
        prompt = f"""
        당신은 반려동물 지식에 대해 답변하는 전문 AI 어시스턴트입니다.
        아래 제공된 '컨텍스트' 정보를 바탕으로 사용자의 '질문'에 대해 답변해 주세요.
        답변은 반드시 컨텍스트에 근거해야 하며, 각 컨텍스트의 출처(source)를 인용하여 신뢰도를 높여주세요.
        만약 컨텍스트에 질문과 관련된 내용이 없다면, "제공된 정보만으로는 답변하기 어렵습니다."라고 솔직하게 답변하세요.

        [컨텍스트]
        {context_str}

        [질문]
        {user_query}

        [답변]
        """
        print("  - 생성된 프롬프트 템플릿 (일부):")
        print(prompt[:500] + "...")
        return prompt

    def _call_llm(self, augmented_prompt: str) -> str:
        """
        증강된 프롬프트를 OpenAI LLM에 보내 답변을 생성합니다.
        """
        print(f"\n[STEP 4] LLM({self.config.LLM_MODEL}) 호출")
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "당신은 지식이 풍부한 반려동물 전문가입니다."},
                    {"role": "user", "content": augmented_prompt},
                ],
                temperature=0.3,
            )
            final_answer = response.choices[0].message.content
            print(f"  - LLM 답변 생성 완료.")
            return final_answer
        except Exception as e:
            print(f"  - LLM 호출 중 오류 발생: {e}")
            return "LLM을 호출하는 데 실패했습니다. API 키 또는 네트워크 상태를 확인하세요."

    def run(self, user_query: str):
        """
        전체 RAG 파이프라인을 실행합니다.
        """
        rewritten_query = self._rewrite_query(user_query)
        context_chunks = self._search(rewritten_query)
        augmented_prompt = self._augment_prompt(user_query, context_chunks)
        final_answer = self._call_llm(augmented_prompt)
        return final_answer


def main():
    """
    메인 실행 함수: 파이프라인을 초기화하고 테스트 쿼리를 실행합니다.
    """
    print("="*50)
    print("RAG 쿼리 테스트를 시작합니다.")
    print("="*50)
    
    try:
        pipeline = RAGPipeline(Config())
        
        # 여기에 테스트하고 싶은 질문을 입력하세요.
        user_query = "강아지가 자꾸 토해요. 원인이 뭘까요?"
        
        print(f"\n사용자 질문: {user_query}")
        print("-" * 50)
        
        final_answer = pipeline.run(user_query)
        
        print("\n" + "="*50)
        print("✅ 최종 생성 답변:")
        print(final_answer)
        print("="*50)

    except FileNotFoundError as e:
        print(f"\n[오류] {e}")
        print("[알림] `build_index.py`를 실행하여 먼저 데이터베이스를 생성해야 합니다.")
    except Exception as e:
        print(f"\n[오류] 예상치 못한 오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
