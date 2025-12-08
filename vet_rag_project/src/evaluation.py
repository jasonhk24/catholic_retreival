import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_recall,
    context_precision,
    faithfulness,
)
from dotenv import load_dotenv


class RagasEvaluator:
    """
    RAGAS 기반 평가 유틸리티.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        embedding_model_name: str = "text-embedding-3-small",
    ):
        # 1차: 명시적으로 전달된 api_key
        token = api_key

        # 2차: vet_rag_project/.env에서 TOKEN 키 직접 파싱
        if not token:
            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.exists():
                print(f"[INFO] .env detected for RAGAS at: {env_path}")
                text = env_path.read_text(encoding="utf-8")
                found_token_line = False
                for raw_line in text.splitlines():
                    stripped = raw_line.strip().lstrip("\ufeff")
                    # 주석 라인 무시
                    if stripped.startswith("#") or not stripped:
                        continue
                    # TOKEN= 또는 TOKEN = 형식 지원
                    if stripped.upper().startswith("TOKEN"):
                        if "=" in stripped:
                            found_token_line = True
                            token = stripped.split("=", 1)[1].strip()
                            # 따옴표 제거
                            token = token.strip('"').strip("'")
                            if token:
                                break
                print(f"[INFO] .env TOKEN line found: {found_token_line}, token extracted: {bool(token)}")
            else:
                print(f"[WARNING] .env not found at expected path: {env_path}")

        # 3차: 환경변수에서 읽기 (OPENAI_API_KEY 또는 TOKEN)
        if not token:
            env_tok = os.getenv("OPENAI_API_KEY")
            env_tok2 = os.getenv("TOKEN")
            print(f"[INFO] Env OPENAI_API_KEY present: {bool(env_tok)}, TOKEN present: {bool(env_tok2)}")
            token = env_tok or env_tok2

        if not token:
            raise ValueError(
                "OPENAI_API_KEY 또는 TOKEN 환경변수가 설정되어 있지 않습니다."
            )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            api_key=token,
        )

        self.embedding = OpenAIEmbeddings(
            model=embedding_model_name,
            openai_api_key=token,
        )

        self.metrics = [
            context_recall,
            context_precision,
            faithfulness,
            answer_relevancy,
        ]

    @staticmethod
    def _prepare_dataset(samples: List[Dict]) -> Dataset:
        processed = []
        for item in samples:
            ground_truth = item.get("ground_truth")
            if isinstance(ground_truth, list):
                ground_truth = "\n".join(gt for gt in ground_truth if gt) or None
            # RAGAS는 ground_truth가 None이거나 빈 문자열이면 answer_relevancy를 제대로 평가하지 못할 수 있음
            # ground_truth가 없으면 빈 문자열 대신 None을 전달하거나, 질문과 답변만으로 평가
            if not ground_truth or (isinstance(ground_truth, str) and not ground_truth.strip()):
                ground_truth = None
            
            processed.append(
                {
                    "question": item["question"],
                    "contexts": item.get("contexts", []),
                    "answer": item["answer"],
                    "ground_truth": ground_truth,  # RAGAS는 ground_truth 필드를 사용
                }
            )
        return Dataset.from_list(processed)

    def evaluate_batch(
        self,
        samples: List[Dict],
        output_path: str = "results/evaluation_report.json",
    ) -> Dict:
        if not samples:
            raise ValueError("평가할 샘플이 비어 있습니다.")

        dataset = self._prepare_dataset(samples)
        result = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embedding,
        )

        per_sample_df: pd.DataFrame = result.to_pandas()
        # 숫자형 컬럼(각 metric 점수)만 대상으로 평균을 계산한다.
        numeric_df = per_sample_df.select_dtypes(include="number")
        summary = {
            metric: float(numeric_df[metric].mean())
            for metric in numeric_df.columns
        }

        report = {
            "per_sample": per_sample_df.to_dict(orient="records"),
            "summary": summary,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fout:
            json.dump(report, fout, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] 평가 리포트 저장: {output_path}")
        return report

