# module_model.py
# [담당자] Person 4 — 모델 리드
# 목적: 증강된 프롬프트(prompt)를 받아 LLM(OpenAI API)을 호출하여 답변을 생성
#
# [입출력 계약]
#   - generate_answer(prompt: str) -> str
#
# [핵심 불변식]
#   - 입력은 module_augment.build_prompt()의 결과(완성된 문자열)
#   - 출력은 LLM이 생성한 자연어 답변 텍스트 1개
#   - API 키는 반드시 환경변수 OPENAI_API_KEY로 관리
#
# [To-Do]
#   1. Mock 응답 → 실제 API 호출로 전환
#   2. 예외처리 및 재시도(backoff) 로직 추가
#   3. rate limit 대응(429) 및 timeout 처리
#   4. 모델명/파라미터 통합 관리(config 또는 .env)

import os
import time
from typing import Optional
from openai import OpenAI, APIError, RateLimitError

# =========================
# 초기화
# =========================
# 안전한 API 키 로딩
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SDK 클라이언트 초기화
_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    _client = OpenAI(api_key=OPENAI_API_KEY)
    print("[model:init] OpenAI 클라이언트 초기화 완료")
else:
    print("[model:init] 경고: OPENAI_API_KEY가 설정되지 않았어요. Mock 모드로 실행합니다.")


# =========================
# 1) LLM 호출 함수
# =========================
def generate_answer(prompt: str) -> str:
    """
    LLM API를 호출하여 주어진 프롬프트(prompt)에 대한 답변을 생성합니다.

    입력:
      - prompt: module_augment.build_prompt()에서 생성된 완성된 문자열

    출력:
      - answer(str): 모델이 생성한 자연어 답변

    실패/예외:
      - API 키 누락 → Mock 응답
      - API 에러/타임아웃 → 예외 메시지 출력 후 Fallback 응답
    """

    # --- 1. Mock 모드 (API 키 없을 때) ---
    if not _client:
        print("[model] Mock 모드로 응답을 생성합니다.")
        return (
            "이것은 Mock 응답이에요. 실제 OpenAI API 호출이 필요합니다.\n"
            "예시: 강아지는 생후 6~8주부터 예방접종을 시작하는 것이 좋아요."
        )

    # --- 2. 실제 API 호출 ---
    try:
        print("[model] OpenAI API 호출 중 ...")
        response = _client.chat.completions.create(
            model="gpt-4o-mini",  # 또는 gpt-4o, gpt-4-turbo 등
            messages=[
                {"role": "system", "content": "당신은 반려동물에 대한 전문 지식을 가진 수의사입니다."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,  # 답변 일관성 높이기 위해 낮게 설정
            max_tokens=600,
            timeout=30,       # 네트워크 타임아웃(초)
        )

        # --- 3. 응답 파싱 ---
        answer = response.choices[0].message.content.strip()
        print("[model] 답변 생성 완료")
        return answer

    except RateLimitError:
        print("[model] 경고: Rate Limit 초과. 3초 후 재시도합니다.")
        time.sleep(3)
        return generate_answer(prompt)

    except APIError as e:
        print(f"[model] API 오류 발생: {e}")
        return "죄송합니다. 현재 서버 응답이 지연되고 있습니다."

    except Exception as e:
        print(f"[model] 예외 발생: {e}")
        return "답변을 생성하는 중 문제가 발생했습니다."


# =========================
# 2) 모듈 단독 테스트
# =========================
if __name__ == "__main__":
    test_prompt = """[근거]
강아지는 생후 6주부터 예방접종을 시작해야 합니다.
종합백신(DHPPL), 코로나 장염, 켄넬코프, 광견병 예방접종이 필요합니다.

[질문]
강아지 예방접종 언제부터 해야 해?

[답변]
위 '근거'를 바탕으로 답변을 작성해 주세요.
"""
    result = generate_answer(test_prompt)
    print("\n--- 생성된 답변 ---")
    print(result)