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
    검색된 문서 리스트(context_docs)를 받아 LLM용 프롬프트를 생성합니다.

    구성 요소:
      [시스템 역할 설명]
      [근거(context)]
      [질문(query)]
      [답변(LLM 생성영역)]

    주의:
      - context_docs는 retrieve() 출력 그대로 사용 가능해야 함
      - 각 청크는 chunk_text 외에도 meta/title 등이 포함될 수 있음
    """

    if not context_docs:
        print("[augment] 경고: 검색된 문서가 비어 있습니다. 빈 context로 프롬프트를 생성합니다.")

    # 1. 검색된 근거(context) 조합
    #   - 중복 문장 제거
    #   - 길이 제한 또는 상위 N개만 선택 (과도한 토큰 방지)
    unique_contexts = []
    seen = set()
    for doc in context_docs:
        text = doc.get("chunk_text", "").strip()
        if text and text not in seen:
            seen.add(text)
            unique_contexts.append(text)
    context_str = "\n\n".join(unique_contexts)

    # 2. 시스템 프롬프트 정의 (역할 지시)
    #   - 기본은 "수의사 전문가"로 설정하지만,
    #     도메인 확장 시 config로 교체 가능.
    system_prompt = (
        "당신은 반려동물 관련 의학 정보를 전문적으로 설명하는 수의사 AI입니다.\n"
        "반드시 아래 '근거'에 포함된 내용만을 활용해 질문에 답변하세요.\n"
        "\n"
        "【중요 규칙 - 반드시 준수하세요】\n"
        "1. 답변은 반드시 한국어로만 작성하세요. 중국어, 영어, 일본어 등 다른 언어를 절대 사용하지 마세요.\n"
        "2. 문맥에 관련 내용이 없거나 근거가 충분하지 않으면 반드시 '정보 부족'이라고 답하세요.\n"
        "3. 문맥에 없는 의학 지식이나 약물 정보를 새로 추측하거나 만들어내지 마세요.\n"
        "4. 근거가 충분하지 않으면 '제공된 정보만으로는 답변하기 어렵습니다.' 또는 '정보 부족'이라고 명확히 밝히세요.\n"
        "5. 최종 답에는 반드시 주요 근거 요약을 bullet 형태로 포함하세요.\n"
        "\n"
        "【언어 규칙】\n"
        "- 답변은 100% 한국어로만 작성하세요.\n"
        "- 다른 언어를 섞으면 안 됩니다.\n"
        "- 중국어, 영어, 일본어 등 어떤 다른 언어도 사용하지 마세요."
    )

    # 3. 사용자 입력 + 근거 조합
    user_prompt = f"""[근거]
{context_str}

[질문]
{query}

[답변]
위 '근거'만을 토대로 다음 형식에 맞게 답변하세요.
1. 핵심 진단/평가
2. 권장 조치(필요 시 '즉시 병원 방문' 여부 포함)
3. 주의할 증상 또는 추가 확인 사항
4. 근거 요약: 제공된 문맥에서 가져온 핵심 한두 줄을 bullet로 제시
근거가 부족하면 1번 항목에서 '제공된 정보만으로는 답변하기 어렵습니다.'라고 명시하세요.
"""

    # 4. 최종 프롬프트 결합
    #   - 모델에 따라 system/user 메시지를 분리할 수도 있지만,
    #     여기서는 단일 문자열 기반 파이프라인(App → Model) 구조로 통일.
    final_prompt = f"{system_prompt}\n\n{user_prompt}"

    print("[augment] 프롬프트 생성 완료")
    return final_prompt


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