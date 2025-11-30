import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

class LLMScorer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", device: str = None):
        """
        Initialize LLM Scorer.
        Note: For this project, we might use a placeholder or a smaller model if the specified model is too large.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LLM] Loading {model_name} on {self.device}...")
        try:
            print(f"[LLM] Step 1/3: Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            print(f"[LLM] Step 2/3: Loading model (this may take several minutes)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            print(f"[LLM] Step 3/3: Setting model to eval mode...")
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"[LLM] Model loaded successfully.")
                
        except Exception as e:
            print(f"Warning: Could not load {model_name}. Using dummy scorer for testing. Error: {e}")
            self.model = None

    def calculate_ppl(self, text: str) -> float:
        """
        Calculate Perplexity (PPL) of a text.
        """
        return self.calculate_ppl_batch([text])[0]

    def calculate_ppl_batch(self, texts: List[str], batch_size: int = 4) -> List[float]:
        """
        Calculate Perplexity (PPL) for a batch of texts.
        """
        if self.model is None:
            return [np.random.rand() * 10 + 10 for _ in texts]

        all_ppls = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize
            # max_length set to avoid OOM, adjust if needed
            encodings = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            input_ids = encodings.input_ids.to(self.device)
            attention_mask = encodings.attention_mask.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Shift for loss calculation
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_mask = attention_mask[..., 1:].contiguous()
                
                # Flatten
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Reshape
                loss = loss.view(shift_labels.size())
                
                # Apply mask
                loss = loss * shift_mask
                
                # Average loss per sequence
                seq_lens = shift_mask.sum(dim=1)
                # Avoid division by zero
                seq_lens = torch.clamp(seq_lens, min=1e-8)
                seq_loss = loss.sum(dim=1) / seq_lens
                
                ppls = torch.exp(seq_loss).cpu().numpy().tolist()
                all_ppls.extend(ppls)
                
        return all_ppls

    def compute_ppl_diff(self, query: str, documents: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Compute Delta PPL = PPL(Query) - PPL(Query + Document)
        Higher Delta PPL means the document is more helpful (reduces perplexity).
        """
        ppl_query = self.calculate_ppl(query)
        
        prompts = []
        for doc in documents:
            # Simple concatenation prompt
            prompt = f"Query: {query}\nDocument: {doc}\nAnswer:"
            prompts.append(prompt)
            
        ppl_with_docs = self.calculate_ppl_batch(prompts, batch_size=batch_size)
        
        ppl_diffs = []
        for ppl_doc in ppl_with_docs:
            delta_ppl = ppl_query - ppl_doc
            ppl_diffs.append(delta_ppl)

        return np.array(ppl_diffs)

    def adaptive_weighting(self, ppl_scores: np.ndarray, alpha_high: float = 0.7, beta_low: float = 0.7) -> float:
        """
        Determine the weight (alpha) for PPL scores based on their standard deviation.
        
        Logic:
        - High Std -> LLM is confident (discriminative) -> High alpha
        - Low Std -> LLM is confused (flat scores) -> Low alpha (rely more on retrieval)
        """
        std_dev = np.std(ppl_scores)
        
        # Threshold logic (can be tuned)
        # If std is high (e.g., > 1.0), use alpha_high
        # If std is low, use (1 - beta_low) effectively giving more weight to retrieval
        
        # For simplicity, let's map std to a weight between 0.1 and 0.9
        # Sigmoid-like mapping or simple threshold
        
        if std_dev > 1.0: # Arbitrary threshold, needs tuning based on PPL scale
            return alpha_high
        else:
            return 1.0 - beta_low

    def z_score_normalization(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores using Z-score.
        """
        if len(scores) < 2:
            return scores
        return (scores - np.mean(scores)) / (np.std(scores) + 1e-8)

if __name__ == "__main__":
    scorer = LLMScorer()
    diffs = scorer.compute_ppl_diff("강아지가 배가 아파요", ["위장염일 수 있습니다.", "사과는 빨갛다."])
    print(f"PPL Diffs: {diffs}")
    norm_diffs = scorer.z_score_normalization(diffs)
    print(f"Normalized: {norm_diffs}")
    weight = scorer.adaptive_weighting(diffs)
    print(f"Adaptive Weight: {weight}")
