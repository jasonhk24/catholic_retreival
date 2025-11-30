"""
Hugging Face에 모델을 업로드하는 스크립트

사용법:
1. .env 파일에 HF_TOKEN 설정 (또는 환경 변수로 설정)
2. 아래 REPO_ID를 수정한 후 실행:
   python upload_model.py
"""

from huggingface_hub import login, upload_folder, HfApi
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# ============================================
# 사용자 설정 (반드시 수정하세요!)
# ============================================

# 1. Hugging Face 토큰 (.env 파일에서 자동으로 읽어옴)
#    .env 파일에 HF_TOKEN=hf_your_token_here 형식으로 저장
MY_TOKEN = os.getenv("HF_TOKEN")

# 2. 저장소 이름 (내아이디/프로젝트명)
#    예: "gildong/vet-rag-reranker" 또는 "myusername/catholic-retriever-model"
REPO_ID = "JOhyeongi/top25bert"  # 여기에 실제 저장소 이름 입력 (대소문자 정확히!)

# 3. 업로드할 모델 경로
#    config.yaml의 training.output_dir에 따라 다를 수 있습니다
MODEL_DIR = "./results/bert_top25percent/final_model"  # 업로드할 모델 폴더 경로

# ============================================
# 업로드 실행
# ============================================

def upload_model():
    """모델을 Hugging Face에 업로드합니다."""
    
    # 토큰 확인
    if not MY_TOKEN:
        raise ValueError(
            "❌ 오류: Hugging Face 토큰을 찾을 수 없습니다!\n"
            "   .env 파일에 HF_TOKEN=hf_your_token_here 형식으로 저장하세요.\n"
            "   토큰 발급: https://huggingface.co/settings/tokens"
        )
    
    # 저장소 이름 확인
    if REPO_ID == "User/My-Project-Model":
        raise ValueError(
            "❌ 오류: REPO_ID를 실제 저장소 이름으로 변경해주세요!\n"
            "   예: 'your-username/vet-rag-reranker'"
        )
    
    # 모델 디렉토리 확인
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ 오류: 모델 디렉토리를 찾을 수 없습니다: {MODEL_DIR}\n"
            f"   먼저 모델을 학습시켜야 합니다: python main_curation.py --mode train"
        )
    
    # 필수 파일 확인
    required_files = ["config.json", "tokenizer_config.json", "vocab.txt"]
    missing_files = [f for f in required_files if not (model_path / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"❌ 오류: 모델 디렉토리에 필수 파일이 없습니다: {missing_files}\n"
            f"   모델 디렉토리: {MODEL_DIR}"
        )
    
    print("="*60)
    print("🚀 Hugging Face 모델 업로드 시작")
    print("="*60)
    print(f"📦 저장소: {REPO_ID}")
    print(f"📁 모델 경로: {MODEL_DIR}")
    print(f"🔓 공개 설정: Public (누구나 다운로드 가능)")
    print("="*60)
    
    # 1. Hugging Face 로그인 및 사용자 정보 확인
    print(f"\n[1/3] Hugging Face 로그인 중...")
    current_username = None
    repo_username = REPO_ID.split("/")[0]
    
    try:
        # .env 파일에서 읽은 토큰을 명시적으로 전달
        login(token=MY_TOKEN)
        
        # 현재 로그인한 사용자 정보 확인
        api = HfApi(token=MY_TOKEN)
        user_info = api.whoami(token=MY_TOKEN)
        current_username = user_info.get("name", "알 수 없음")
        
        print(f"✅ 로그인 완료")
        print(f"   현재 사용자: {current_username}")
        
        # REPO_ID의 사용자명과 일치하는지 확인
        if repo_username.lower() != current_username.lower():
            print(f"\n⚠️  경고: REPO_ID의 사용자명('{repo_username}')과 현재 로그인한 사용자명('{current_username}')이 일치하지 않습니다!")
            print(f"   REPO_ID를 '{current_username}/top25bert' 형식으로 변경하거나,")
            print(f"   '{repo_username}' 계정으로 로그인하세요.")
            print(f"\n   계속 진행하시겠습니까? (권장: REPO_ID 수정)")
            
    except Exception as e:
        print(f"❌ 로그인 실패: {e}")
        print(f"   .env 파일에 HF_TOKEN=hf_your_token_here 형식으로 저장하세요.")
        print(f"   또는 토큰이 유효한지 확인하세요: https://huggingface.co/settings/tokens")
        raise
    
    # 2. 공개 리포지토리 생성
    print(f"\n[2/3] '{REPO_ID}' 리포지토리를 생성(확인) 중...")
    try:
        api.create_repo(
            repo_id=REPO_ID, 
            token=MY_TOKEN,
            private=False,  # 공개 리포지토리
            exist_ok=True,
            repo_type="model"
        )
        print(f"✅ 리포지토리 준비 완료: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            print(f"\n❌ 리포지토리 생성 실패: 권한 없음")
            print(f"   가능한 원인:")
            print(f"   1. REPO_ID의 사용자명이 현재 로그인한 계정과 일치하지 않음")
            if current_username:
                print(f"      현재 로그인: {current_username}")
                print(f"      REPO_ID 사용자명: {repo_username}")
                print(f"\n   해결 방법:")
                print(f"   → REPO_ID를 '{current_username}/top25bert' 형식으로 변경하세요")
            else:
                print(f"      REPO_ID 사용자명: {repo_username}")
            print(f"   2. 사용자명의 대소문자가 정확하지 않음 (Hugging Face는 대소문자를 구분합니다)")
            print(f"   3. 토큰이 해당 계정의 것이 아님")
            print(f"\n   추가 확인:")
            print(f"   - Hugging Face 웹사이트에서 정확한 사용자명 확인: https://huggingface.co/settings")
            print(f"   - 올바른 계정의 토큰을 사용하세요")
        else:
            print(f"❌ 리포지토리 생성 실패: {e}")
        raise
    
    # 3. 모델 폴더 업로드
    print(f"\n[3/3] 모델 폴더 업로드 중... (시간이 걸릴 수 있습니다)")
    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=REPO_ID,
            repo_type="model",
            token=MY_TOKEN
        )
        print(f"✅ 업로드 완료!")
    except Exception as e:
        print(f"❌ 업로드 실패: {e}")
        raise
    
    print("\n" + "="*60)
    print("✅ 업로드 완료!")
    print("="*60)
    print(f"📦 저장소 주소: https://huggingface.co/{REPO_ID}")
    print("\n💡 이제 팀원들은 토큰 없이 모델을 다운로드할 수 있습니다!")
    print("   코드 실행 시 자동으로 모델이 다운로드됩니다.")
    print("="*60)

if __name__ == "__main__":
    upload_model()
