"""
Fast-DetectGPT Calibration Script

Analyzes the distribution of curvature scores to find optimal 
calibration parameters (mu0, sigma0, mu1, sigma1) for GPT-2.
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_samples(hc3_path: str, n_per_class: int = 50):
    """Load balanced samples from HC3"""
    samples = {"AI": [], "Human": []}
    
    if not os.path.exists(hc3_path):
        print(f"HC3 not found at {hc3_path}")
        return samples
    
    jsonl_files = [f for f in os.listdir(hc3_path) if f.endswith('.jsonl')]
    
    for filename in jsonl_files:
        filepath = os.path.join(hc3_path, filename)
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    for ans in data.get('human_answers', []):
                        if len(ans) >= 100 and len(samples["Human"]) < n_per_class:
                            samples["Human"].append(ans[:2000])
                    
                    for ans in data.get('chatgpt_answers', []):
                        if len(ans) >= 100 and len(samples["AI"]) < n_per_class:
                            samples["AI"].append(ans[:2000])
                except:
                    continue
    
    return samples


def analyze_fastdetect_scores():
    """Analyze raw curvature scores to find optimal calibration"""
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = "/home/lightdesk/Projects/AI-Text/ensemble/cache"
    
    print("Loading GPT-2 model...")
    model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    def get_curvature(text):
        """Calculate curvature for a text"""
        tokenized = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_token_type_ids=False
        ).to(device)
        
        if tokenized.input_ids.shape[1] < 10:
            return None
        
        labels = tokenized.input_ids[:, 1:]
        
        with torch.no_grad():
            logits = model(**tokenized).logits[:, :-1]
        
        # Same model for reference and scoring
        logits_ref = logits
        logits_score = logits
        
        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        
        discrepancy = (
            (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) 
            / var_ref.sum(dim=-1).sqrt()
        )
        return discrepancy.mean().item()
    
    # Load samples
    samples = load_samples("/home/lightdesk/Projects/AI-Text/HC3", n_per_class=100)
    
    if not samples["AI"] or not samples["Human"]:
        print("No samples loaded!")
        return
    
    # Collect curvatures
    ai_curvatures = []
    human_curvatures = []
    
    print("\nCalculating AI curvatures...")
    for text in tqdm(samples["AI"]):
        c = get_curvature(text)
        if c is not None:
            ai_curvatures.append(c)
    
    print("\nCalculating Human curvatures...")
    for text in tqdm(samples["Human"]):
        c = get_curvature(text)
        if c is not None:
            human_curvatures.append(c)
    
    # Analyze distributions
    print("\n" + "="*60)
    print("FAST-DETECTGPT CURVATURE ANALYSIS")
    print("="*60)
    
    print(f"\nAI Curvatures (n={len(ai_curvatures)}):")
    print(f"  Mean (mu1):   {np.mean(ai_curvatures):.4f}")
    print(f"  Std (sigma1): {np.std(ai_curvatures):.4f}")
    print(f"  Min:          {np.min(ai_curvatures):.4f}")
    print(f"  Max:          {np.max(ai_curvatures):.4f}")
    
    print(f"\nHuman Curvatures (n={len(human_curvatures)}):")
    print(f"  Mean (mu0):   {np.mean(human_curvatures):.4f}")
    print(f"  Std (sigma0): {np.std(human_curvatures):.4f}")
    print(f"  Min:          {np.min(human_curvatures):.4f}")
    print(f"  Max:          {np.max(human_curvatures):.4f}")
    
    # Optimal calibration params
    mu0 = np.mean(human_curvatures)
    sigma0 = np.std(human_curvatures)
    mu1 = np.mean(ai_curvatures)
    sigma1 = np.std(ai_curvatures)
    
    print(f"\n" + "="*60)
    print("RECOMMENDED CALIBRATION PARAMETERS FOR GPT-2:")
    print("="*60)
    print(f"""
'gpt2_gpt2': {{
    'mu0': {mu0:.4f}, 
    'sigma0': {sigma0:.4f}, 
    'mu1': {mu1:.4f}, 
    'sigma1': {sigma1:.4f}
}}
""")
    
    # Test accuracy with these params
    from scipy.stats import norm
    
    def compute_prob(x, mu0, sigma0, mu1, sigma1):
        pdf_value0 = norm.pdf(x, loc=mu0, scale=sigma0)
        pdf_value1 = norm.pdf(x, loc=mu1, scale=sigma1)
        prob = pdf_value1 / (pdf_value0 + pdf_value1 + 1e-10)
        return prob
    
    ai_correct = sum(1 for c in ai_curvatures if compute_prob(c, mu0, sigma0, mu1, sigma1) > 0.5)
    human_correct = sum(1 for c in human_curvatures if compute_prob(c, mu0, sigma0, mu1, sigma1) <= 0.5)
    
    accuracy = (ai_correct + human_correct) / (len(ai_curvatures) + len(human_curvatures))
    fpr = 1 - (human_correct / len(human_curvatures))
    
    print(f"Accuracy with calibrated params: {accuracy:.2%}")
    print(f"FPR with calibrated params: {fpr:.2%}")
    
    return {
        "mu0": mu0,
        "sigma0": sigma0,
        "mu1": mu1,
        "sigma1": sigma1
    }


if __name__ == "__main__":
    results = analyze_fastdetect_scores()
