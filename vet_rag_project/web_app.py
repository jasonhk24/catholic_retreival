# -*- coding: utf-8 -*-
import gradio as gr
from app import RAGPipeline, Config
import os

# --- 전역 설정 및 모델 로드 ---
try:
    rag_pipeline = RAGPipeline(Config())
    print("[INFO] WebApp: RAG 파이프라인이 성공적으로 초기화되었습니다.")
except Exception as e:
    rag_pipeline = None
    print(f"[ERROR] WebApp: RAG 파이프라인 초기화 실패: {e}")

# --- Gradio 인터페이스 함수 ---
def ask_ai_consultant(message: str, history: list) -> str:
    """
    history: 이전 대화 기록 (type="messages"일 경우 딕셔너리 리스트)
    """
    if rag_pipeline is None:
        return "죄송합니다. RAG 파이프라인이 초기화되지 않았습니다."
    
    print(f"\n[Gradio] 사용자 질문: {message}")
    answer = rag_pipeline.run(message)
    print(f"[Gradio] AI 답변: {answer}")
    
    return answer

# --- Gradio UI 구성 ---
# theme=gr.themes.Soft()는 Gradio 5.x 이상에서 작동 (6.0 버그 시 삭제 가능)
with gr.Blocks(theme=gr.themes.Soft(), title="반려동물 AI 상담소") as web_app:
    
    # [수정 1] 안정적인 이미지 URL 사용 및 스타일 개선
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/320px-Cat_August_2010-4.jpg" 
                 style="width: 150px; height: 150px; object-fit: cover; border-radius: 50%; margin: auto; border: 3px solid #ffb347;">
            <h1 style="font-size: 28px; margin-top: 10px;">🐶 반려동물 AI 상담소 🐱</h1>
            <p style="font-size: 16px; color: #666;">
                반려동물의 건강 상태나 궁금한 점을 물어봐주세요.<br>
                <span style="font-size: 12px; color: #999;">(전공 서적 기반으로 답변합니다)</span>
            </p>
        </div>
        """
    )
    
    # [수정 2] type="messages" 추가하여 경고 메시지 제거
    gr.ChatInterface(
        fn=ask_ai_consultant,
        type="messages",  # 이 옵션이 경고를 없앱니다 (최신 방식)
        chatbot=gr.Chatbot(height=500, type="messages"),
        textbox=gr.Textbox(placeholder="예: 강아지가 자꾸 토해요. 원인이 뭘까요?", container=False, scale=7),
        title=None,
        examples=["강아지가 갑자기 다리를 절어요", "고양이 피부에 각질이 생겼어요", "노견이 먹으면 좋은 영양제는?"],
        cache_examples=False,
    )

if __name__ == "__main__":
    print("[INFO] Gradio 웹 앱을 시작합니다.")
    web_app.launch()