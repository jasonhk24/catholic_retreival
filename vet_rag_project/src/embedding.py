import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import numpy as np

class KmBertEmbedder:
    def __init__(self, model_name: str = "madatnlp/km-bert", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[KmBERT] Loading {model_name} on {self.device}...")
        print(f"[KmBERT] Step 1/2: Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"[KmBERT] Step 2/2: Loading model...")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"[KmBERT] Model loaded successfully.")

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        print(f"[KmBERT] Encoding {len(texts)} text(s) in batches of {batch_size}...")
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            if i % (batch_size * 10) == 0:
                print(f"[KmBERT] Progress: {i}/{len(texts)}")
            
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

        print(f"[KmBERT] Encoding complete.")
        return np.vstack(all_embeddings)

if __name__ == "__main__":
    # Test code
    embedder = KmBertEmbedder()
    emb = embedder.encode(["안녕하세요", "수의학 질문입니다."])
    print(f"Embedding shape: {emb.shape}")
