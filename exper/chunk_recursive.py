
import json
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 설정 ---
TARGET_JSON_FILE = '/Users/test/Assign/자연어처리/rag/data/TS_말뭉치데이터_내과/0a1fbe96-e151-11ef-a39c-00155dced605.json'
OUTPUT_FILE = 'chunk_result_recursive.txt'

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

# --- 메인 로직 ---

print("--- 방법 1: Recursive Character Text Splitter 시작 ---")

# 1. 데이터 로드
doc = load_document_from_file(TARGET_JSON_FILE)

if doc:
    # 2. 청킹 수행 (노트북의 고급 설정값 적용)
    recursive_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    recursive_chunks = recursive_splitter.split_documents([doc])
    
    # 3. 결과 저장
    save_chunks_to_file(recursive_chunks, OUTPUT_FILE)

print("--- 방법 1: 완료 ---")
