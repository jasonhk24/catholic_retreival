
import json
import os
import kss
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 설정 ---
TARGET_JSON_FILE = '/Users/test/Assign/자연어처리/rag/data/TS_말뭉치데이터_내과/0a1fbe96-e151-11ef-a39c-00155dced605.json'
OUTPUT_FILE = 'chunk_result_kss.txt'

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

print("--- 방법 2: KSS (Korean Sentence Splitter) 시작 ---")

# 1. 데이터 로드
doc = load_document_from_file(TARGET_JSON_FILE)

if doc:
    # 2. KSS 청킹 수행 (성능 개선 로직 포함)
    kss_chunks = []
    
    # KSS 과부하 방지를 위한 사전 분할기
    pre_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=0,
        separators=["\n\n", "\n", ". "]
    )

    # 2.1. 문서를 kss가 처리하기 좋은 크기로 대략적으로 나눔
    smaller_docs = pre_splitter.split_documents([doc])
    
    # 2.2. 작게 나뉜 각 문서에 대해 kss 실행
    for small_doc in smaller_docs:
        try:
            sentences = kss.split_sentences(small_doc.page_content)
            for sentence in sentences:
                # 2.3. 각 문장에 원본 메타데이터를 붙여 새 Document 생성
                kss_chunks.append(Document(page_content=sentence, metadata=doc.metadata))
        except Exception as e:
            print(f"KSS 처리 중 오류: {e}")

    # 3. 결과 저장
    if kss_chunks:
        save_chunks_to_file(kss_chunks, OUTPUT_FILE)
    else:
        print("처리할 청크가 없습니다.")

print("--- 방법 2: 완료 ---")
