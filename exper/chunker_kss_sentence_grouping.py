# -*- coding: utf-8 -*-
"""
KSS로 문장 분리 후, 지정된 글자 수에 맞춰 문장들을 다시 묶어주는 청킹 모듈
"""
import kss
from langchain_core.documents import Document
from typing import List
import json
from pathlib import Path

def group_sentences_by_char_limit(docs: List[Document], max_chars: int = 500) -> List[Document]:
    """
    LangChain Document 리스트를 입력받아, KSS로 문장을 분리한 뒤
    지정된 글자 수(max_chars)에 가깝게 문장들을 다시 묶어 새로운 Document 리스트를 반환합니다.
    """
    print(f"[KSS-Grouping] KSS 글자 수 기반 청킹 시작 (목표 청크 크기: {max_chars}자)")
    final_chunks = []
    for doc in docs:
        try:
            sentences = kss.split_sentences(doc.page_content)
        except Exception as e:
            print(f"KSS 처리 중 오류 발생 (원본 문서 통째로 반환): {e}")
            final_chunks.append(doc)
            continue
        current_chunk_sentences = []
        current_chunk_chars = 0
        for sentence in sentences:
            sentence_len = len(sentence)
            if current_chunk_chars + sentence_len > max_chars and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                final_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
                current_chunk_sentences = [sentence]
                current_chunk_chars = sentence_len
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_chars += sentence_len
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            final_chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
    print(f"[KSS-Grouping] 총 {len(final_chunks)}개의 청크 생성 완료.")
    return final_chunks

def save_chunks_to_file(chunks: List[Document], filename: str):
    """청크 리스트를 텍스트 파일에 저장합니다."""
    output_path = Path(__file__).resolve().parent.parent / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"--- CHUNK {i+1} (글자 수: {len(chunk.page_content)}) ---\n")
            f.write(f"[METADATA]: {chunk.metadata}\n\n")
            f.write(chunk.page_content)
            f.write("\n\n")
    print(f"'{output_path}'에 총 {len(chunks)}개의 청크를 저장했습니다.")

# ==================================================
# 이 파일이 직접 실행될 때를 위한 테스트 코드
# ==================================================
if __name__ == '__main__':
    print("### KSS 문장 그룹핑 청커 실제 데이터 테스트 (파일 저장) ###")
    try:
        base_dir = Path(__file__).resolve().parent.parent
        target_file = base_dir / "data/TS_말뭉치데이터_내과/0a1fbe96-e151-11ef-a39c-00155dced605.json"
        with open(target_file, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        page_content = data.get('disease')
        if not page_content:
            raise ValueError("파일에서 'disease' 내용을 찾을 수 없습니다.")
        metadata = {
            'source': str(target_file.relative_to(base_dir)),
            'title': data.get('title', 'N/A'),
            'department': data.get('department', 'N/A')
        }
        real_doc = Document(page_content=page_content, metadata=metadata)
        print(f"'{target_file.name}' 파일 로드 성공.")
    except Exception as e:
        print(f"테스트 파일 로드 중 오류 발생: {e}")
        real_doc = None

    if real_doc:
        TARGET_CHUNK_SIZE = 500
        grouped_chunks = group_sentences_by_char_limit(
            docs=[real_doc],
            max_chars=TARGET_CHUNK_SIZE
        )
        output_filename = "real_data_kss_grouping_result.txt"
        save_chunks_to_file(grouped_chunks, output_filename)