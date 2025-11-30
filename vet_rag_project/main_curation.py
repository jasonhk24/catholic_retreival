import argparse
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.curator import DataCurator
from src.trainer import train_cross_encoder

# Load environment variables from .env file
ENV_SEARCH_PATHS = [
    Path.cwd() / ".env",
    Path(__file__).resolve().parent / ".env",
    Path(__file__).resolve().parent.parent / ".env",
    Path(__file__).resolve().parent.parent.parent / ".env",
]

env_loaded = False
for env_path in ENV_SEARCH_PATHS:
    if env_path.exists():
        load_dotenv(env_path)
        env_loaded = True
        break
if not env_loaded:
    load_dotenv()

# Allow TOKEN fallback (if user stored as TOKEN=...)
if not os.getenv("OPENAI_API_KEY") and os.getenv("TOKEN"):
    os.environ["OPENAI_API_KEY"] = os.getenv("TOKEN")

# Set environment variables (required for Hugging Face models)
os.environ.setdefault("WANDB_DISABLED", "true")

if not os.getenv("HF_TOKEN"):
    print("[WARNING] HF_TOKEN not found in environment variables.")
    print("Please create a .env file with your Hugging Face token.")
    print("See .env.example for reference.")

def main():
    parser = argparse.ArgumentParser(description="Vet RAG Project Pipeline")
    parser.add_argument("--mode", type=str, choices=["curate", "train", "inference", "all"], default="all", help="Pipeline mode")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--input_data", type=str, default="raw_data.json", help="Path to raw input data (for curation)")
    parser.add_argument("--curated_data", type=str, default="curated_dataset.json", help="Path to curated dataset (for training)")
    parser.add_argument("--model_path", type=str, default="./results/final_model", help="Path to trained model (for inference)")
    parser.add_argument("--query", type=str, help="Query for inference")
    
    args = parser.parse_args()
    
    # Load config to fetch knowledge base directories
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    doc_dirs = base_config.get("knowledge_base", {}).get("directories") or [r"..\data\TS_말뭉치데이터_내과"]
    if isinstance(doc_dirs, str):
        doc_dirs = [doc_dirs]

    # Paths to data
    query_zip = r"..\data\Training\02.라벨링데이터\TL_질의응답데이터_내과.zip"
    
    if args.mode in ["curate", "all"]:
        print("="*50)
        print("[INFO] Starting Data Curation Phase")
        print("="*50)
        # Load real data
        from src.data_loader import load_documents_from_dirs, load_queries
        
        valid_dirs = [d for d in doc_dirs if os.path.exists(d)]
        if not valid_dirs:
             print(f"[WARNING] None of the document directories {doc_dirs} were found. Please check the paths.")
             return
        if not os.path.exists(query_zip):
             print(f"[WARNING] Query ZIP file {query_zip} not found. Please check the path.")
             return

        print("[INFO] Loading data...")
        documents = load_documents_from_dirs(doc_dirs)
        # 전체 질문을 로드한 뒤 9:1로 나누고, 학습용에서 600개만 샘플링
        queries = load_queries(query_zip, sample_ratio=1.0)
        
        # Generate dummy IDs for documents
        doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        if not documents or not queries:
            print("[ERROR] Failed to load data. Exiting.")
            return
        
        # 질문을 학습/평가용으로 9:1 분할
        train_queries, eval_queries = train_test_split(
            queries, test_size=0.1, random_state=42
        )
        # 학습용 질문에서 최대 600개만 사용
        if len(train_queries) > 600:
            random.seed(42)
            train_queries = random.sample(train_queries, 600)

        print(f"[INFO] Total queries: {len(queries)} (train: {len(train_queries)}, eval: {len(eval_queries)})")
        print(f"Starting curation with {len(train_queries)} train queries and {len(documents)} documents.")
        
        curator = DataCurator(config_path=args.config)
        
        # 방법 1: 상위 25%를 긍정으로 라벨링 (RAG^2 방식) - 학습용 질문 600개만 사용
        output_path_25 = args.curated_data.replace(".json", "_top25percent.json")
        print(f"\n[INFO] Generating dataset with top 25% labeling method (train queries only)...")
        curator.run_pipeline(train_queries, documents, doc_ids, output_path=output_path_25, method="top_25_percent")

        # 평가용 질문 목록을 별도 파일로 저장 (추론/평가용)
        eval_q_path = args.curated_data.replace(".json", "_eval_queries.json")
        import json as _json
        with open(eval_q_path, "w", encoding="utf-8") as f:
            _json.dump(eval_queries, f, ensure_ascii=False, indent=2)
        print(f"\n[SUCCESS] Saved eval queries to {eval_q_path}")
        
    if args.mode in ["train", "all"]:
        print("\n" + "="*50)
        print("[INFO] Starting Model Training Phase")
        print("="*50)
        
        # 단일 데이터셋 (top25percent)만 사용
        dataset_25 = args.curated_data.replace(".json", "_top25percent.json")
        if not os.path.exists(dataset_25):
            print(f"[WARNING] Dataset {dataset_25} not found. Please run in 'curate' mode first.")
            return

        dataset_name = "top25percent"
        model_type = "bert"  # km-bert 기반 크로스 인코더

        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} on {dataset_name} dataset (km-bert)")
        print(f"{'='*60}")

        # 출력 디렉토리 설정
        output_dir = f"./results/{model_type}_{dataset_name}"

        # config 임시 수정을 위한 딕셔너리
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 모델 타입과 출력 디렉토리 설정
        config["training"]["model_type"] = model_type
        config["training"]["output_dir"] = output_dir

        # 임시 config 파일 생성
        temp_config = args.config.replace(".yaml", f"_{model_type}_{dataset_name}.yaml")
        with open(temp_config, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        try:
            train_cross_encoder(config_path=temp_config, dataset_path=dataset_25)
        finally:
            # 임시 config 파일 삭제
            if os.path.exists(temp_config):
                os.remove(temp_config)
        
    if args.mode == "inference":
        print("="*50)
        print("[INFO] Starting Inference Phase")
        print("="*50)
        
        if not args.query:
            print("[ERROR] Please provide a query for inference using --query")
            return

        from src.rag_pipeline import VetRAGPipeline
        
        pipeline = VetRAGPipeline(
            config_path=args.config,
            doc_dir=doc_dirs
        )
        
        pipeline.run(args.query)

if __name__ == "__main__":
    main()
