import json
import os
import glob
import random
import zipfile
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Chunk text using RecursiveCharacterTextSplitter.
    """
    if not text:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Wrap text in Document object as splitter expects documents or list of strings
    # split_text returns list of strings
    chunks = splitter.split_text(text)
    return chunks

def load_documents(data_dir: str) -> List[str]:
    """
    Load documents from JSON files in the specified directory.
    Assumes 'disease' field contains the text.
    """
    print(f"[INFO] Loading documents from {data_dir}...")
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    documents = []
    
    for file_path in json_files:
        try:
            # 일부 JSON 파일에 UTF-8 BOM이 포함되어 있으므로 utf-8-sig로 읽어준다.
            with open(file_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except UnicodeDecodeError:
            # 예외적으로 다른 인코딩일 경우를 대비한 안전장치
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        text = data.get("disease", "")
        if not text:
            continue

        text = text.replace("\n", " ").strip()
        chunks = chunk_text(text)
        documents.extend(chunks)
            
    print(f"[SUCCESS] Loaded {len(documents)} document chunks from {len(json_files)} files.")
    return documents

def load_documents_from_dirs(directories: List[str]) -> List[str]:
    """
    Load and aggregate documents from multiple directories.
    """
    all_documents = []
    valid_dirs = 0
    for directory in directories or []:
        if not directory:
            continue
        if not os.path.exists(directory):
            print(f"[WARNING] Document directory {directory} not found. Skipping...")
            continue
        valid_dirs += 1
        docs = load_documents(directory)
        all_documents.extend(docs)
    print(f"[INFO] Total aggregated chunks: {len(all_documents)} from {valid_dirs} directories.")
    return all_documents

def extract_main_question(query: str) -> str:
    """
    질문에서 핵심 질문만 추출합니다.
    여러 질문이 합쳐진 경우, 첫 번째 주요 질문만 반환합니다.
    """
    if not query:
        return ""
    
    # 물음표로 분리
    parts = query.split("?")
    
    # 첫 번째 질문 부분 찾기 (주요 증상/상황 설명 + 첫 번째 질문)
    if len(parts) > 1:
        # 첫 번째 질문까지 포함
        main_query = parts[0] + "?"
        
        # 너무 짧으면 (10자 미만) 다음 질문도 포함
        if len(main_query.strip()) < 10 and len(parts) > 1:
            main_query = parts[0] + "?" + parts[1] + "?"
        
        # 핵심 질문만 추출 (너무 긴 경우 앞부분만)
        # 일반적으로 첫 200자 정도가 핵심 질문
        if len(main_query) > 300:
            # 문장 단위로 자르기
            sentences = main_query.split(".")
            main_query = ".".join(sentences[:3])  # 처음 3개 문장만
            if not main_query.endswith("?"):
                main_query += "?"
    else:
        # 물음표가 없는 경우, 처음 200자만
        main_query = query[:200] if len(query) > 200 else query
    
    return main_query.strip()


def load_queries(zip_path: str, sample_ratio: float = 0.9, extract_main_only: bool = True) -> List[str]:
    """
    Load queries directly from a ZIP file containing JSONs.
    Assumes 'qa' -> 'input' field contains the query.
    
    Args:
        zip_path: ZIP 파일 경로
        sample_ratio: 샘플링 비율
        extract_main_only: True면 핵심 질문만 추출 (여러 질문이 합쳐진 경우 첫 번째만)
    """
    print(f"[INFO] Loading queries from ZIP: {zip_path}...")
    queries = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            file_list = [f for f in z.namelist() if f.endswith('.json')]
            total_files = len(file_list)
            print(f"   Found {total_files} JSON files in ZIP.")
            
            if sample_ratio < 1.0:
                sample_size = int(total_files * sample_ratio)
                print(f"   Sampling {sample_size} files ({sample_ratio*100}%)...")
                file_list = random.sample(file_list, sample_size)
            
            print("   Reading files from ZIP...")
            for i, filename in enumerate(file_list):
                if i % 1000 == 0:
                    print(f"   Processed {i}/{len(file_list)} files...", end="\r")
                
                try:
                    with z.open(filename) as f:
                        data = json.load(f)
                        query = data.get("qa", {}).get("input", "")
                        if query:
                            if extract_main_only:
                                query = extract_main_question(query)
                            queries.append(query)
                except Exception as e:
                    print(f"Error reading {filename} in ZIP: {e}")
                    
    except Exception as e:
        print(f"[ERROR] Error opening ZIP file {zip_path}: {e}")
        return []

    print(f"\n[SUCCESS] Loaded {len(queries)} queries.")
    return queries

if __name__ == "__main__":
    # Test
    pass
