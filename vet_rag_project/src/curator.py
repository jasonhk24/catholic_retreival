import yaml
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict

from .embedding import KmBertEmbedder
from .llm_scorer import LLMScorer
from .graph_refiner import GraphRefiner

class DataCurator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # Don't load models in __init__, load them when needed
        self.graph_refiner = GraphRefiner(
            k_neighbors=self.config["graph"]["k_neighbors"],
            lambda_val=self.config["graph"]["lambda_propagation"],
            steps=self.config["graph"]["propagation_steps"]
        )

    def run_pipeline(self, queries: List[str], documents: List[str], doc_ids: List[str], 
                     output_path: str = "curated_dataset.json", 
                     method: str = "top_k"):
        """
        Run pipeline with sequential model loading to avoid GPU memory issues.
        
        Step 1: KmBERT → Compute embeddings → Save → Unload
        Step 2: LLM → Compute PPL scores → Save → Unload  
        Step 3: Load cached data → Generate final dataset
        
        Args:
            queries: 질문 리스트
            documents: 문서 리스트
            doc_ids: 문서 ID 리스트
            output_path: 출력 파일 경로
            method: 데이터셋 생성 방법 ("top_k", "top_25_percent", "hard_negative")
                - "top_k": 기존 방식 (상위 k개를 긍정)
                - "top_25_percent": 상위 25%를 긍정으로 라벨링 (RAG^2 방식)
                - "hard_negative": 정답 5개 + Hard Negative 15개 (1:3 비율)
        """
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        doc_emb_path = os.path.join(cache_dir, "doc_embeddings.npy")
        query_emb_path = os.path.join(cache_dir, "query_embeddings.npy")
        ppl_scores_path = os.path.join(cache_dir, "ppl_scores.npy")
        
        # ========== STEP 1: KmBERT Embeddings ==========
        print("\n" + "="*60)
        print("STEP 1: Computing KmBERT Embeddings")
        print("="*60)
        
        d_embs = None
        q_embs = None

        if os.path.exists(doc_emb_path) and os.path.exists(query_emb_path):
            print(f"[INFO] Loading cached embeddings...")
            d_embs = np.load(doc_emb_path)
            q_embs = np.load(query_emb_path)

            # 캐시 검증: 문서/질문 개수와 임베딩 개수가 일치하는지 확인
            cache_ok = True
            if d_embs.shape[0] != len(documents) or d_embs.shape[0] == 0:
                print(
                    f"[WARNING] Cached document embeddings shape {d_embs.shape} "
                    f"does not match documents length {len(documents)}. Recomputing..."
                )
                cache_ok = False
            if q_embs.shape[0] != len(queries) or q_embs.shape[0] == 0:
                print(
                    f"[WARNING] Cached query embeddings shape {q_embs.shape} "
                    f"does not match queries length {len(queries)}. Recomputing..."
                )
                cache_ok = False

            if cache_ok:
                print(f"[SUCCESS] Loaded embeddings from cache")
            else:
                d_embs = None
                q_embs = None

        # 캐시가 없거나, 검증에 실패한 경우 임베딩 재계산
        if d_embs is None or q_embs is None:
            print(f"[1/2] Computing document embeddings...")
            embedder = KmBertEmbedder(model_name=self.config["retrieval"]["model_name"])
            d_embs = embedder.encode(documents)
            np.save(doc_emb_path, d_embs)
            print(f"[INFO] Saved to {doc_emb_path}")

            print(f"[2/2] Computing query embeddings...")
            q_embs = embedder.encode(queries)
            np.save(query_emb_path, q_embs)
            print(f"[INFO] Saved to {query_emb_path}")

            # Free GPU memory
            del embedder
            torch.cuda.empty_cache()
            print(f"[INFO] KmBERT unloaded, GPU memory freed")
        
        # ========== STEP 2: LLM PPL Scoring ==========
        print("\n" + "="*60)
        print("STEP 2: Computing LLM PPL Scores")
        print("="*60)
        
        # PPL 캐시는 샘플링 비율에 따라 달라지므로, 항상 재계산하거나 캐시 키를 샘플링 비율로 구분해야 함
        # 여기서는 간단히 캐시를 무시하고 재계산
        use_ppl_cache = False  # 샘플링 비율이 변경되었으므로 캐시 무시
        if use_ppl_cache and os.path.exists(ppl_scores_path):
            print(f"[INFO] Loading cached PPL scores...")
            all_ppl_data = np.load(ppl_scores_path, allow_pickle=True).item()
            print(f"[SUCCESS] Loaded from cache")
        else:
            print(f"Loading LLM scorer...")
            llm_scorer = LLMScorer(model_name=self.config["llm_scorer"]["model_name"])
            
            all_ppl_data = {}
            top_k = self.config["retrieval"]["top_k"]
            
            print(f"Computing PPL for {len(queries)} queries...")
            for i, query in enumerate(tqdm(queries)):
                # Get top-k candidates for this query
                q_emb = q_embs[i:i+1]
                sims = np.dot(d_embs, q_emb.T).flatten()
                top_indices = np.argsort(sims)[::-1][:top_k]
                candidate_docs = [documents[idx] for idx in top_indices]
                
                # Compute PPL
                ppl_diffs = llm_scorer.compute_ppl_diff(query, candidate_docs)
                normalized_ppl = llm_scorer.z_score_normalization(ppl_diffs)
                alpha = llm_scorer.adaptive_weighting(
                    ppl_diffs,
                    alpha_high=self.config["llm_scorer"]["alpha_high_std"],
                    beta_low=self.config["llm_scorer"]["beta_low_std"]
                )
                
                all_ppl_data[i] = {
                    'top_indices': top_indices,
                    'ppl_diffs': ppl_diffs,
                    'normalized_ppl': normalized_ppl,
                    'alpha': alpha
                }
            
            np.save(ppl_scores_path, all_ppl_data)
            print(f"[INFO] Saved PPL scores to {ppl_scores_path}")
            
            # Free GPU memory
            del llm_scorer
            torch.cuda.empty_cache()
            print(f"[INFO] LLM unloaded, GPU memory freed")
        
        # ========== STEP 3: Generate Final Dataset ==========
        print("\n" + "="*60)
        print("STEP 3: Generating Final Dataset")
        print("="*60)
        
        all_labeled_data = []
        top_k = self.config["retrieval"]["top_k"]
        
        for i, query in enumerate(tqdm(queries, desc="Processing queries")):
            q_emb = q_embs[i:i+1]
            sims = np.dot(d_embs, q_emb.T).flatten()
            
            # Get cached PPL data (if exists, otherwise compute on the fly)
            if i in all_ppl_data:
                ppl_data = all_ppl_data[i]
            else:
                # PPL data not found in cache, compute it
                print(f"[WARNING] PPL data for query {i} not found in cache. Computing on the fly...")
                top_k = self.config["retrieval"]["top_k"]
                top_indices = np.argsort(sims)[::-1][:top_k]
                candidate_docs = [documents[idx] for idx in top_indices]
                
                # Need to load LLM scorer temporarily
                llm_scorer = LLMScorer(model_name=self.config["llm_scorer"]["model_name"])
                ppl_diffs = llm_scorer.compute_ppl_diff(query, candidate_docs)
                normalized_ppl = llm_scorer.z_score_normalization(ppl_diffs)
                alpha = llm_scorer.adaptive_weighting(
                    ppl_diffs,
                    alpha_high=self.config["llm_scorer"]["alpha_high_std"],
                    beta_low=self.config["llm_scorer"]["beta_low_std"]
                )
                ppl_data = {
                    'top_indices': top_indices,
                    'ppl_diffs': ppl_diffs,
                    'normalized_ppl': normalized_ppl,
                    'alpha': alpha
                }
                del llm_scorer
                torch.cuda.empty_cache()
            top_indices = ppl_data['top_indices']
            normalized_ppl = ppl_data['normalized_ppl']
            alpha = ppl_data['alpha']
            
            candidate_docs = [documents[idx] for idx in top_indices]
            candidate_ids = [doc_ids[idx] for idx in top_indices]
            candidate_embs = d_embs[top_indices]
            
            # Combine scores
            sims_norm = (sims[top_indices] - sims[top_indices].min()) / (sims[top_indices].max() - sims[top_indices].min() + 1e-8)
            ppl_norm = (normalized_ppl - normalized_ppl.min()) / (normalized_ppl.max() - normalized_ppl.min() + 1e-8)
            s_init = (1 - alpha) * sims_norm + alpha * ppl_norm
            
            # Graph refinement
            adj_matrix = self.graph_refiner.build_knn_graph(candidate_embs)
            s_final = self.graph_refiner.propagate_scores(s_init, adj_matrix)
            
            # Curation - 방법에 따라 라벨링
            final_top_indices = np.argsort(s_final)[::-1]
            sorted_scores = s_final[final_top_indices]
            
            if method == "top_25_percent":
                # 방법 1: 상위 25%를 긍정으로 라벨링 (RAG^2 방식)
                num_candidates = len(final_top_indices)
                num_positive = max(1, int(num_candidates * 0.25))  # 최소 1개
                
                for rank, idx in enumerate(final_top_indices):
                    label = 1 if rank < num_positive else 0
                    all_labeled_data.append({
                        "query": query,
                        "doc_id": candidate_ids[idx],
                        "document": candidate_docs[idx],
                        "label": label,
                        "score": float(s_final[idx]),
                        "rank": rank
                    })
                    
            elif method == "hard_negative":
                # 방법 2: 정답 5개 + Hard Negative 15개 (1:3 비율)
                top_pos_k = self.config["curation"]["top_k_positive"]
                num_negative = top_pos_k * 3  # 1:3 비율
                
                # Positive 샘플 (상위 5개)
                positive_indices = final_top_indices[:top_pos_k]
                for idx in positive_indices:
                    all_labeled_data.append({
                        "query": query,
                        "doc_id": candidate_ids[idx],
                        "document": candidate_docs[idx],
                        "label": 1,
                        "score": float(s_final[idx]),
                        "rank": list(final_top_indices).index(idx)
                    })
                
                # Negative 샘플 (상위권 Hard Negative만 선택)
                # 상위 5개를 제외한 나머지 중에서 상위 15개만 선택
                negative_candidates = final_top_indices[top_pos_k:]
                selected_negatives = negative_candidates[:min(num_negative, len(negative_candidates))]
                
                for idx in selected_negatives:
                    all_labeled_data.append({
                        "query": query,
                        "doc_id": candidate_ids[idx],
                        "document": candidate_docs[idx],
                        "label": 0,
                        "score": float(s_final[idx]),
                        "rank": list(final_top_indices).index(idx)
                    })
                    
            else:
                # 기존 방식: top_k
                top_pos_k = self.config["curation"]["top_k_positive"]
                for rank, idx in enumerate(final_top_indices):
                    label = 1 if rank < top_pos_k else 0
                    all_labeled_data.append({
                        "query": query,
                        "doc_id": candidate_ids[idx],
                        "document": candidate_docs[idx],
                        "label": label,
                        "score": float(s_final[idx]),
                        "rank": rank
                    })
        
        # 데이터셋 통계 계산
        num_positive = sum(1 for item in all_labeled_data if item["label"] == 1)
        num_negative = sum(1 for item in all_labeled_data if item["label"] == 0)
        total = len(all_labeled_data)
        positive_ratio = num_positive / total if total > 0 else 0
        
        # Save final dataset
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_labeled_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Curated dataset saved to {output_path}")
        print(f"[INFO] Dataset Statistics:")
        print(f"   Total samples: {total}")
        print(f"   Positive (label=1): {num_positive} ({positive_ratio*100:.2f}%)")
        print(f"   Negative (label=0): {num_negative} ({(1-positive_ratio)*100:.2f}%)")
        print(f"   Ratio: 1:{num_negative/num_positive:.2f}" if num_positive > 0 else "   Ratio: N/A")

if __name__ == "__main__":
    curator = DataCurator()
    queries = ["강아지 예방접종 언제 하나요?"]
    docs = ["강아지 예방접종은 6주부터 시작합니다.", "고양이 예방접종은 8주부터입니다.", "사과는 맛있다."]
    ids = ["doc1", "doc2", "doc3"]
    curator.run_pipeline(queries, docs, ids, "test_curation.json")
