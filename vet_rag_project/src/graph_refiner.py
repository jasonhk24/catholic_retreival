import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple

class GraphRefiner:
    def __init__(self, k_neighbors: int = 5, lambda_val: float = 0.3, steps: int = 3):
        self.k_neighbors = k_neighbors
        self.lambda_val = lambda_val
        self.steps = steps

    def build_knn_graph(self, embeddings: np.ndarray) -> torch.Tensor:
        """
        Build a k-NN graph from embeddings.
        Returns a normalized adjacency matrix (sparse tensor or dense tensor).
        """
        n = len(embeddings)
        sim_matrix = cosine_similarity(embeddings)
        
        # Zero out self-similarity
        np.fill_diagonal(sim_matrix, 0)
        
        # Build adjacency matrix
        adj = np.zeros((n, n))
        
        for i in range(n):
            # Get top-k neighbors
            top_k_indices = np.argsort(sim_matrix[i])[::-1][:self.k_neighbors]
            for j in top_k_indices:
                adj[i, j] = sim_matrix[i, j]
                adj[j, i] = sim_matrix[i, j] # Make symmetric
        
        # Normalize (Row-stochastic)
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1 # Avoid division by zero
        adj_normalized = adj / row_sum
        
        return torch.FloatTensor(adj_normalized)

    def propagate_scores(self, initial_scores: np.ndarray, adj_matrix: torch.Tensor) -> np.ndarray:
        """
        Propagate scores using LightGCN-like propagation.
        S_final = (1 - lambda) * S_init + lambda * (A * S_init)
        Iterative version: S_(t+1) = (1 - lambda) * S_init + lambda * (A * S_t)
        """
        if isinstance(initial_scores, np.ndarray):
            scores = torch.FloatTensor(initial_scores)
        else:
            scores = initial_scores
            
        initial_tensor = scores.clone()
        
        for _ in range(self.steps):
            propagated = torch.matmul(adj_matrix, scores)
            scores = (1 - self.lambda_val) * initial_tensor + self.lambda_val * propagated
            
        return scores.numpy()

if __name__ == "__main__":
    # Test code
    refiner = GraphRefiner(k_neighbors=2)
    embs = np.random.rand(5, 10)
    scores = np.array([0.1, 0.8, 0.2, 0.9, 0.3])
    
    adj = refiner.build_knn_graph(embs)
    print(f"Adjacency Matrix:\n{adj}")
    
    final_scores = refiner.propagate_scores(scores, adj)
    print(f"Initial Scores: {scores}")
    print(f"Final Scores: {final_scores}")
