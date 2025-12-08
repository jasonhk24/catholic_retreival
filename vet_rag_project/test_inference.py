"""
추론 파이프라인 테스트 스크립트

Fine-tuned Cross-Encoder를 사용한 전체 RAG 파이프라인 테스트
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 설정 로드
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("="*60)
print("🧪 추론 파이프라인 테스트")
print("="*60)

# 1. 환경 변수 확인
print("\n[1/5] 환경 변수 확인...")
hf_token = os.getenv("HF_TOKEN")
openai_key = os.getenv("TOKEN")
print(f"  ✓ HF_TOKEN: {'설정됨' if hf_token else '❌ 없음'}")
print(f"  ✓ OPENAI API KEY: {'설정됨' if openai_key else '❌ 없음'}")

# 2. 모델 경로 확인
print("\n[2/5] 학습된 모델 확인...")
model_path = Path("results/bert_top25percent")
final_model_path = Path("results/final_model")

if model_path.exists():
    print(f"  ✓ 모델 발견: {model_path}")
    # 모델 파일 확인
    model_files = list(model_path.glob("*"))
    print(f"  ✓ 파일 수: {len(model_files)}")
    print(f"  ✓ 주요 파일: {[f.name for f in model_files[:5]]}")
elif final_model_path.exists():
    print(f"  ✓ 모델 발견: {final_model_path}")
    model_path = final_model_path
else:
    print("  ❌ 학습된 모델을 찾을 수 없습니다.")
    print("  먼저 'python main_curation.py --mode train'을 실행하세요.")
    exit(1)

# 3. 데이터 확인
print("\n[3/5] 데이터 확인...")
data_dirs = config["knowledge_base"]["directories"]
doc_count = 0
for data_dir in data_dirs:
    data_path = Path(data_dir)
    if data_path.exists():
        files = list(data_path.glob("*.json"))
        doc_count += len(files)
        print(f"  ✓ {data_path.name}: {len(files)}개 문서")
    else:
        print(f"  ⚠ {data_path.name}: 경로 없음")

print(f"  ✓ 총 문서 수: {doc_count}개")

if doc_count == 0:
    print("  ❌ 문서가 없습니다. data/ 폴더를 확인하세요.")
    exit(1)

# 4. RAG 파이프라인 초기화
print("\n[4/5] RAG 파이프라인 초기화...")
try:
    from src.rag_pipeline import VetRAGPipeline
    
    print("  ✓ 모듈 import 성공")
    print("  ⏳ 파이프라인 로딩 중... (시간이 걸릴 수 있습니다)")
    
    pipeline = VetRAGPipeline(
        config_path="config.yaml",
        doc_dir=data_dirs
    )
    
    print("  ✓ 파이프라인 초기화 완료")
    print(f"  ✓ 로드된 문서 수: {len(pipeline.documents)}개")
    
except Exception as e:
    print(f"  ❌ 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. 추론 테스트
print("\n[5/5] 추론 테스트...")
test_queries = [
    "강아지가 노란 토를 해요. 어떻게 해야 하나요?",
    "고양이가 밥을 안 먹어요.",
    "강아지 예방접종은 언제부터 해야 하나요?"
]

print(f"\n총 {len(test_queries)}개 질문으로 테스트합니다.\n")

for i, query in enumerate(test_queries, 1):
    print("="*60)
    print(f"테스트 {i}/{len(test_queries)}")
    print("="*60)
    print(f"질문: {query}\n")
    
    try:
        answer = pipeline.run(query)
        print(f"\n✅ 추론 성공!")
        print("="*60)
        
        # 다음 질문으로 넘어가기 전 짧은 대기
        if i < len(test_queries):
            print("\n다음 질문으로 이동...\n")
            import time
            time.sleep(2)
            
    except Exception as e:
        print(f"\n❌ 추론 실패: {e}")
        import traceback
        traceback.print_exc()
        break

print("\n" + "="*60)
print("🎉 테스트 완료!")
print("="*60)
print("\n다음 단계:")
print("1. 결과가 만족스러우면 모델을 Hugging Face에 업로드하세요")
print("2. GitHub에 코드를 커밋하고 푸시하세요")
print("3. README.md를 업데이트하세요")
