# module_augment.py
# [담당자] Person 3 — 증강 리드
# 목적: 사용자 질문(query) + 검색된 문서(context)를 결합해
#       LLM이 이해하기 좋은 프롬프트(prompt)로 가공
#
# [입출력 계약]
#   - build_prompt(query: str, context_docs: list[dict]) -> str
#
# 입력(context_docs) 예시:
#   [
#     {
#       "chunk_id": 0,
#       "chunk_text": "강아지는 생후 6주부터 예방접종을 시작해야 합니다.",
#       "meta": {"title": "개(2판)...", "department": "내과"},
#       "score": 0.87
#     }, ...
#   ]
#
# 출력(str):
#   LLM API에 전달할 완성된 프롬프트 문자열
#
# [핵심 역할]
#   - Retrieval 단계 결과(context)를 문맥 손실 없이 압축/정리
#   - Prompt 형식 통일 (근거→질문→답변 구조)
#   - Few-shot / CoT / Role Prompting 등 실험적 개선 가능
#
# [To-Do]
#   1. Few-shot 예시 삽입 포맷 추가
#   2. 체계적 CoT 유도 (예: "생각 과정을 단계별로 설명하세요")
#   3. context 포맷팅 규칙 개선 (길이 제한, 중복 제거 등)
#   4. prompt 길이 제약(4096 tokens 이하) 검증 로직 추가

from typing import List, Dict

# =========================
# 1) 핵심 함수
# =========================
def build_prompt(query: str, context_docs: List[Dict]) -> str:
    """
    검색된 문서(context)와 질문(query)을 결합하여 User Message 내용을 생성합니다.

    [변경 사항]
    - 기존의 `시스템 역할(수의사 AI 등)` 정의를 제거했습니다. (generate 함수에서 처리)
    - 오직 [근거 자료] + [질문] + [구체적 답변 포맷]만 구성합니다.
    """

    # 1. 검색된 근거(context) 조합 (중복 제거 로직 유지)
    if not context_docs:
        print("[augment] 경고: 검색된 문서가 없습니다. 빈 context를 사용합니다.")

    unique_contexts = []
    seen = set()

    for doc in context_docs:
        text = doc.get("chunk_text", "").strip()
        if text and text not in seen:
            seen.add(text)
            unique_contexts.append(text)

    # 문맥이 너무 길어질 경우를 대비해 구분자 추가
    context_str = "\n\n——\n\n".join(unique_contexts)

    # 2. User Prompt 구성
    # 시스템 역할(페르소나)는 제거하고, '작업 지시'만 남김
    prompt_content = f"""
### [참고 자료 (Context)]
{context_str}

### [사용자 질문 (Query)]
{query}

### [답변 작성 지침]
위 [참고 자료]를 바탕으로 [사용자 질문]에 대해 답변해주세요.
다음 형식을 반드시 준수해야 합니다.

1. **핵심 진단/평가**: 질문에 대한 의학적/상식적 평가
2. **추가 조치**: 구체적인 행동 가이드 (필요 시 '병원 방문 필요' 명시)
3. **주의사항**: 추가로 확인해야 할 증상이나 위험 요소
4. **근거 요약**: 위 답변의 출처가 되는 [참고 자료]의 내용을 1~2줄로 요약 (Bullet point 사용)

***주의사항:***
- [참고 자료]에 없는 내용은 절대 지어내지 말고, 정보가 부족하면 "제공된 정보만으로는 판단하기 어렵습니다"라고 명시하세요.
- 전문 용어는 사용하되, 보호자가 이해하기 쉽게 풀어써주세요.
"""
    
    print("[augment] 프롬프트 생성 완료")
    return prompt_content.strip()


# =========================
# 2) 모듈 단독 테스트
# =========================
if __name__ == "__main__":
    test_query = "강아지 예방접종 언제부터 해야 해?"
    test_context = [
        {
            "chunk_id": 0,
            "chunk_text": "강아지는 생후 6주부터 예방접종을 시작해야 합니다. 종합백신, 코로나 장염, 켄넬코프, 광견병 예방접종이 필요합니다.",
            "meta": {"title": "개(2판)", "department": "내과"},
            "score": 0.91,
        },
        {
            "chunk_id": 1,
            "chunk_text": "새끼 강아지의 사회화 시기는 생후 3주에서 12주 사이가 매우 중요합니다.",
            "meta": {"title": "개(2판)", "department": "행동의학"},
            "score": 0.74,
        },
    ]

    prompt = build_prompt(test_query, test_context)
    print("\n--- 생성된 프롬프트 ---\n")
    print(prompt)