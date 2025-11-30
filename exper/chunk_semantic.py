
import json
import os
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# --- 설정 ---
TARGET_JSON_FILE = '/Users/test/Assign/자연어처리/rag/data/TS_말뭉치데이터_내과/0a1fbe96-e151-11ef-a39c-00155dced605.json'
OUTPUT_FILE = 'chunk_result_semantic.txt'

# --- 헬퍼 함수 ---

def load_document_from_file(file_path):
    """지정된 단일 JSON 파일에서 Document 객체를 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            page_content = data.get('disease')
            if page_content and isinstance(page_content, str):
                metadata = {
                    'source': file_path,
                    'title': data.get('title', 'N/A'),
                    'author': data.get('author', 'N/A'),
                    'publisher': data.get('publisher', 'N/A'),
                    'department': data.get('department', 'N/A')
                }
                print(f"문서를 성공적으로 로드했습니다: {file_path}")
                return Document(page_content=page_content, metadata=metadata)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
    except Exception as e:
        print(f"파일 처리 중 오류 발생 {file_path}: {e}")
    return None

def save_chunks_to_file(chunks, filename):
    """청크 리스트를 파일에 저장합니다."""
    with open(filename, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i+1} ---\n")
            if isinstance(chunk, Document):
                f.write(f"[METADATA]: {chunk.metadata}\n\n")
                f.write(chunk.page_content)
            else:
                f.write(chunk)
            f.write("\n\n")
    print(f"'{filename}'에 총 {len(chunks)}개의 청크를 저장했습니다.")

def _clean_text(text: str) -> str:
    """텍스트에서 불필요한 줄바꿈과 숫자 라인을 제거합니다."""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.isdigit(): continue
        if len(line) > 0:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

# --- 메인 로직 ---

print("--- 방법 3: Semantic Chunker 시작 ---")

# 0. API 키 확인
if 'OPENAI_API_KEY' not in os.environ or not os.environ['OPENAI_API_KEY']:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! 경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.      !!!")
    print("!!! Semantic Chunking을 실행할 수 없습니다. 스크립트를 종료합니다. !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
else:
    # 1. 데이터 로드
    doc = load_document_from_file(TARGET_JSON_FILE)

    if doc:
        # 2. Semantic 청킹 수행 (노트북 설정값 적용)
        
        # 2.1. 텍스트 전처리
        cleaned_doc = Document(page_content=_clean_text(doc.page_content), metadata=doc.metadata)
        
        # 2.2. Semantic Splitter 정의
        chunking_embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
        semantic_splitter = SemanticChunker(
            embeddings=chunking_embed_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95  # 민감도 설정
        )
        
        # 2.3. 청킹 실행
        semantic_chunks = semantic_splitter.split_documents([cleaned_doc])
        
        # 3. 결과 저장
        save_chunks_to_file(semantic_chunks, OUTPUT_FILE)

print("--- 방법 3: 완료 ---")
