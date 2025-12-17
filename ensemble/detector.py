"""
AI Text Detection Ensemble - Unified Detector Module (v3.0)

This module provides a self-contained 3-component ensemble detector:
1. DeTeCtive - Style clustering via contrastive learning (RoBERTa-based)
2. Binoculars - Cross-model probability divergence (GPT-2 based)
3. Fast-DetectGPT - Probability curvature smoothness (GPT-2 based)

v3.0 Enhancements:
- Domain-adaptive thresholding to reduce false positives on formal text
- Confidence-gated voting for better uncertainty handling
- Burstiness-based pre-filtering for obvious human/AI cases
- Works on latest models (GPT-4o, Claude 3.5, Gemini) via zero-shot methods

All code is consolidated here - no external dependencies on other folders.
Optimized for Modal deployment and AMD GPU compatibility.

Author: AI-Text Project
"""

import os
import pickle
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

from .core.text_embedding import TextEmbeddingModel
from .core.faiss_index import Indexer
from .core import binoculars_metrics
from .domain_analyzer import DomainAnalyzer, DomainInfo


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DetectionResult:
    """Result from a single detector"""
    score: float  # 0.0-1.0, higher = more likely AI
    prediction: str  # "AI" or "Human"
    confidence: float  # 0.0-1.0
    details: Dict


@dataclass
class EnsembleResult:
    """Final ensemble detection result with domain awareness"""
    prediction: str  # "AI", "Human", or "UNCERTAIN"
    confidence: float  # 0.0-1.0
    agreement: str  # "3/3", "2/3", "1/3", "0/3"
    ensemble_score: float  # Weighted average score
    breakdown: Dict[str, DetectionResult]
    suggested_action: str  # "ACCEPT", "REVIEW", or "REJECT"
    # Domain-adaptive fields (v3.0)
    domain: str = "INFORMAL"  # "FORMAL", "INFORMAL", "TECHNICAL"
    domain_adjusted: bool = False  # Whether thresholds were adjusted
    burstiness: float = 0.5  # Text burstiness score (higher = more human-like)


# ============================================================================
# Utility Functions
# ============================================================================

def get_device() -> str:
    """Get the best available device"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_ensemble_dir() -> str:
    """Get the ensemble directory path"""
    return os.path.dirname(os.path.abspath(__file__))


def get_models_dir() -> str:
    """Get the models directory path"""
    return os.path.join(get_ensemble_dir(), "models")


def get_database_dir() -> str:
    """Get the database directory path"""
    return os.path.join(get_ensemble_dir(), "database")


# Default model/database configuration
# Using Deepfake model which is well-calibrated and tested
DEFAULT_DETECTIVE_MODEL = "Deepfake_best.pth"
DEFAULT_DETECTIVE_DB = "deepfake_full"


def get_cache_dir() -> str:
    """Get the HuggingFace cache directory path"""
    return os.path.join(get_ensemble_dir(), "cache")


# ============================================================================
# DeTeCtive Detector
# ============================================================================

class DeTeCtiveDetector:
    """
    DeTeCtive detector using style-based KNN clustering.
    
    Exploits: AI families cluster in embedding space
    How it works: Encodes text with RoBERTa, finds k-nearest neighbors
    in a pre-built database, votes based on neighbor labels.
    """
    
    def __init__(
        self, 
        model_path: str = None, 
        database_path: str = None,
        model_name: str = "princeton-nlp/unsup-simcse-roberta-base",
        embedding_dim: int = 768,
        k: int = 10,
        device: str = None
    ):
        """
        Initialize DeTeCtive detector
        
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            database_path: Path to FAISS database directory
            model_name: Base model name from HuggingFace
            embedding_dim: Embedding dimension (768 for RoBERTa-base)
            k: Number of neighbors for KNN voting
            device: Device to use (auto-detected if None)
        
        Note: Default configuration uses Deepfake model and database
              which is well-calibrated and tested. The Deepfake dataset provides
              good coverage across domains and AI models.
        """
        self.device = device or get_device()
        self.k = k
        self.embedding_dim = embedding_dim
        
        # Use default paths if not provided
        # Priority: Deepfake full (default) > Deepfake sample > M4 (fallback)
        if model_path is None:
            default_model = os.path.join(get_models_dir(), DEFAULT_DETECTIVE_MODEL)
            if os.path.exists(default_model):
                model_path = default_model
            else:
                # Fallback to M4 if Deepfake not found
                m4_model = os.path.join(get_models_dir(), "M4_monolingual_best.pth")
                if os.path.exists(m4_model):
                    model_path = m4_model
                else:
                    raise FileNotFoundError(f"No DeTeCtive model found. Please provide model_path or place model in {get_models_dir()}")
        
        if database_path is None:
            default_db = os.path.join(get_database_dir(), DEFAULT_DETECTIVE_DB)
            sample_db = os.path.join(get_database_dir(), "deepfake_sample")
            m4_db = os.path.join(get_database_dir(), "m4_monolingual")
            if os.path.exists(default_db):
                database_path = default_db
            elif os.path.exists(sample_db):
                database_path = sample_db
            elif os.path.exists(m4_db):
                database_path = m4_db
            else:
                raise FileNotFoundError(f"No DeTeCtive database found. Please provide database_path or create database in {get_database_dir()}")
        
        print("Loading DeTeCtive...")
        
        # Load embedding model
        self.model = TextEmbeddingModel(model_name)
        if self.device == "cuda":
            self.model = self.model.cuda()
        
        # Load trained weights
        state_dict = torch.load(
            model_path, 
            map_location=self.model.model.device, 
            weights_only=False
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.tokenizer = self.model.tokenizer
        
        # Load FAISS index
        self.index = Indexer(embedding_dim, device='cpu')  # Always CPU for FAISS
        self.index.deserialize_from(database_path)
        
        # Load label dictionary
        with open(os.path.join(database_path, 'label_dict.pkl'), 'rb') as f:
            self.label_dict = pickle.load(f)
        
        print(f"DeTeCtive loaded on {self.device}. Database size: {self.index.index.ntotal}")
    
    @torch.no_grad()
    def detect(self, text: str) -> DetectionResult:
        """
        Detect if text is AI-generated
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectionResult with score, prediction, and details
        """
        # Tokenize
        encoded_text = self.tokenizer.batch_encode_plus(
            [text],
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True
        )
        
        if self.device == "cuda":
            encoded_text = {k: v.cuda() for k, v in encoded_text.items()}
        
        # Get embeddings
        embeddings = self.model(encoded_text).cpu().detach().numpy()
        
        # KNN search
        top_ids_and_scores = self.index.search_knn(embeddings, self.k)
        ids, scores = top_ids_and_scores[0]
        
        # Vote based on neighbor labels (0 = AI, 1 = Human)
        ai_count = sum(1 for id in ids if self.label_dict[int(id)] == 0)
        
        # Calculate scores
        ai_score = ai_count / self.k
        prediction = "AI" if ai_score > 0.5 else "Human"
        confidence = abs(ai_score - 0.5) * 2
        
        return DetectionResult(
            score=ai_score,
            prediction=prediction,
            confidence=confidence,
            details={
                "ai_votes": ai_count,
                "human_votes": self.k - ai_count,
                "k": self.k,
                "method": "KNN style clustering"
            }
        )


# ============================================================================
# Binoculars Detector
# ============================================================================

# Thresholds from original Binoculars paper (optimized for Falcon-7B family)
# These DO NOT WORK for GPT-2 models!
BINOCULARS_FALCON_ACCURACY_THRESHOLD = 0.9015310749276843
BINOCULARS_FALCON_FPR_THRESHOLD = 0.8536432310785527

# Calibrated thresholds for GPT-2 / GPT-2-medium pair
# Derived from 1,200-sample calibration (HC3 + RAID datasets) - 94% accuracy
# AI scores: mean=0.61, std=0.08 | Human scores: mean=0.80, std=0.07
BINOCULARS_GPT2_ACCURACY_THRESHOLD = 0.70  # 94.25% accuracy at this threshold
BINOCULARS_GPT2_FPR_THRESHOLD = 0.71       # Low FPR with 92.5% recall


class BinocularsDetector:
    """
    Binoculars detector using cross-model probability divergence.
    
    Exploits: AI text shows consistent perplexity across different models
    How it works: Compares perplexity ratio between observer and performer models.
    Human text shows higher divergence between models.
    
    NOTE: Thresholds are model-pair specific!
    - Falcon-7B pair: threshold ~0.85
    - GPT-2 pair: threshold ~0.65-0.70
    """
    
    def __init__(
        self,
        observer_model: str = "gpt2",
        performer_model: str = "gpt2-medium",
        mode: str = "low-fpr",
        use_bfloat16: bool = False,  # Disabled by default for AMD compatibility
        max_token_observed: int = 512,
        device: str = None,
        cache_dir: str = None
    ):
        """
        Initialize Binoculars detector
        
        Args:
            observer_model: Observer model name (smaller model)
            performer_model: Performer model name (larger model)
            mode: "low-fpr" (conservative) or "accuracy" (balanced)
            use_bfloat16: Use bfloat16 precision (disable for AMD GPUs)
            max_token_observed: Maximum tokens to process
            device: Device to use
            cache_dir: HuggingFace cache directory
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = device or get_device()
        cache_dir = cache_dir or get_cache_dir()
        
        # Select thresholds based on model pair
        is_gpt2_pair = "gpt2" in observer_model.lower()
        
        if is_gpt2_pair:
            # Use GPT-2 calibrated thresholds
            if mode == "low-fpr":
                self.threshold = BINOCULARS_GPT2_FPR_THRESHOLD
            elif mode == "accuracy":
                self.threshold = BINOCULARS_GPT2_ACCURACY_THRESHOLD
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            # Use original Falcon thresholds
            if mode == "low-fpr":
                self.threshold = BINOCULARS_FALCON_FPR_THRESHOLD
            elif mode == "accuracy":
                self.threshold = BINOCULARS_FALCON_ACCURACY_THRESHOLD
            else:
                raise ValueError(f"Invalid mode: {mode}")
        
        print(f"Loading Binoculars (mode={mode})...")
        
        # Determine dtype - avoid bfloat16 for AMD GPUs
        if use_bfloat16 and self.device == "cuda":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        # Load observer model
        self.observer_model = AutoModelForCausalLM.from_pretrained(
            observer_model,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map={"": self.device}
        )
        self.observer_model.eval()
        
        # Load performer model
        self.performer_model = AutoModelForCausalLM.from_pretrained(
            performer_model,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map={"": self.device}
        )
        self.performer_model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            observer_model,
            cache_dir=cache_dir
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_token_observed = max_token_observed
        print(f"Binoculars loaded on {self.device}. Threshold: {self.threshold:.4f}")
    
    def _tokenize(self, text: Union[str, List[str]]):
        """Tokenize input text"""
        batch = [text] if isinstance(text, str) else text
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if len(batch) > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False
        ).to(self.device)
        return encodings
    
    @torch.inference_mode()
    def _get_logits(self, encodings):
        """Get logits from both models"""
        observer_logits = self.observer_model(**encodings).logits
        performer_logits = self.performer_model(**encodings).logits
        if self.device != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits
    
    def compute_score(self, text: str) -> float:
        """
        Compute raw Binoculars score
        
        Returns:
            Raw score (lower = more likely AI)
        """
        encodings = self._tokenize(text)
        observer_logits, performer_logits = self._get_logits(encodings)
        
        # Calculate perplexity and cross-entropy
        ppl = binoculars_metrics.perplexity(encodings, performer_logits)
        x_ppl = binoculars_metrics.entropy(
            observer_logits,
            performer_logits,
            encodings,
            self.tokenizer.pad_token_id
        )
        
        binoculars_score = ppl / x_ppl
        return float(binoculars_score[0])
    
    def detect(self, text: str) -> DetectionResult:
        """
        Detect if text is AI-generated
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectionResult with normalized score (higher = more AI)
        """
        raw_score = self.compute_score(text)
        
        # Normalize using calibrated ranges from HC3+RAID calibration:
        # AI scores: mean=0.61, std=0.08 -> range roughly [0.45, 0.77]
        # Human scores: mean=0.80, std=0.07 -> range roughly [0.66, 0.94]
        # Use linear interpolation: scores < 0.55 -> definitely AI, > 0.88 -> definitely Human
        ai_boundary = 0.55   # Below this = high confidence AI
        human_boundary = 0.88  # Above this = high confidence Human
        
        if raw_score <= ai_boundary:
            normalized_score = 1.0  # Definitely AI
        elif raw_score >= human_boundary:
            normalized_score = 0.0  # Definitely Human  
        else:
            # Linear interpolation between boundaries
            normalized_score = 1.0 - (raw_score - ai_boundary) / (human_boundary - ai_boundary)
        
        prediction = "AI" if raw_score < self.threshold else "Human"
        confidence = abs(normalized_score - 0.5) * 2
        
        return DetectionResult(
            score=normalized_score,
            prediction=prediction,
            confidence=confidence,
            details={
                "raw_score": raw_score,
                "threshold": self.threshold,
                "method": "Cross-model perplexity divergence"
            }
        )


# ============================================================================
# Fast-DetectGPT Detector
# ============================================================================

class FastDetectGPTDetector:
    """
    Fast-DetectGPT detector using probability curvature analysis.
    
    Exploits: AI text has smooth probability curves, human text is spiky
    How it works: Calculates conditional probability curvature analytically
    without needing perturbation sampling (fast version).
    
    CRITICAL: Research shows best results with TWO DIFFERENT models:
    - Sampling/Reference model: Provides probability distribution for expected tokens
    - Scoring model: Calculates actual log probabilities of the text
    
    Using same model for both is possible but suboptimal.
    
    Research-validated calibration params (from Fast-DetectGPT paper):
    - falcon-7b (sampling) + falcon-7b-instruct (scoring): mu0=-0.0707, sigma0=0.9520, mu1=2.9306, sigma1=1.9039 [BEST]
    - gpt-j-6B (sampling) + gpt-neo-2.7B (scoring): mu0=0.2713, sigma0=0.9366, mu1=2.2334, sigma1=1.8731
    - gpt-neo-2.7B (same model): mu0=-0.2489, sigma0=0.9968, mu1=1.8983, sigma1=1.9935
    """
    
    # Calibration parameters from the Fast-DetectGPT paper and empirical testing
    # Format: 'sampling-model_scoring-model': {parameters}
    # The sampling model provides reference probability distribution
    # The scoring model calculates actual probabilities for the text
    CALIBRATION_PARAMS = {
        # OPTIMAL: Two different models (research-recommended)
        'gpt2-medium_gpt2': {
            # GPT-2-medium as sampling/reference, GPT-2 as scoring
            # Calibrated on HC3 + RAID datasets (1200 samples) - 91.58% accuracy
            'mu0': 1.5220, 'sigma0': 1.2104, 'mu1': 4.9159, 'sigma1': 1.5107
        },
        'gpt-j-6B_gpt-neo-2.7B': {
            # From Fast-DetectGPT paper - black-box setting
            'mu0': 0.2713, 'sigma0': 0.9366, 'mu1': 2.2334, 'sigma1': 1.8731
        },
        'falcon-7b_falcon-7b-instruct': {
            # From Fast-DetectGPT paper - BEST performance
            'mu0': -0.0707, 'sigma0': 0.9520, 'mu1': 2.9306, 'sigma1': 1.9039
        },
        # SUBOPTIMAL: Same model for both (fallback)
        'gpt2_gpt2': {
            # Same model - less accurate but still functional
            'mu0': 0.6557, 'sigma0': 1.1536, 'mu1': 4.7479, 'sigma1': 1.5231
        },
        'gpt-neo-2.7B_gpt-neo-2.7B': {
            'mu0': -0.2489, 'sigma0': 0.9968, 'mu1': 1.8983, 'sigma1': 1.9935
        }
    }
    
    def __init__(
        self,
        scoring_model_name: str = "gpt2",
        sampling_model_name: str = "gpt2-medium",  # Default to different model (research-recommended)
        device: str = None,
        cache_dir: str = None,
        shared_scoring_model=None,
        shared_sampling_model=None,
        shared_tokenizer=None
    ):
        """
        Initialize Fast-DetectGPT detector
        
        IMPORTANT: For best accuracy, use two DIFFERENT models:
        - Sampling model (reference): Provides expected probability distribution (larger is better)
        - Scoring model: Calculates actual log probabilities of the text
        
        Args:
            scoring_model_name: Model used to score text probabilities (e.g., gpt2)
            sampling_model_name: Reference model for curvature (e.g., gpt2-medium)
            device: Device to use
            cache_dir: HuggingFace cache directory
            shared_scoring_model: Optionally share scoring model (saves memory)
            shared_sampling_model: Optionally share sampling model (saves memory)
            shared_tokenizer: Optionally share a tokenizer instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scipy.stats import norm
        
        self.device = device or get_device()
        cache_dir = cache_dir or get_cache_dir()
        self.norm = norm
        
        self.scoring_model_name = scoring_model_name
        self.sampling_model_name = sampling_model_name
        self.use_same_model = (scoring_model_name == sampling_model_name)
        
        if self.use_same_model:
            print(f"  WARNING: Using same model for both scoring and sampling is suboptimal!")
            print(f"  Research recommends two different models for better curvature detection.")
        
        print(f"Loading Fast-DetectGPT (scoring={scoring_model_name}, sampling={sampling_model_name})...")
        
        # Load scoring model
        if shared_scoring_model is not None:
            self.scoring_model = shared_scoring_model
            self.scoring_tokenizer = shared_tokenizer
        else:
            self.scoring_model = AutoModelForCausalLM.from_pretrained(
                scoring_model_name,
                cache_dir=cache_dir
            ).to(self.device)
            self.scoring_tokenizer = AutoTokenizer.from_pretrained(
                scoring_model_name,
                cache_dir=cache_dir
            )
            if self.scoring_tokenizer.pad_token is None:
                self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
        self.scoring_model.eval()
        
        # Load sampling/reference model
        if self.use_same_model:
            self.sampling_model = self.scoring_model
            self.sampling_tokenizer = self.scoring_tokenizer
        elif shared_sampling_model is not None:
            self.sampling_model = shared_sampling_model
            # Use same tokenizer (GPT-2 family shares tokenizer)
            self.sampling_tokenizer = self.scoring_tokenizer
            self.sampling_model.eval()
        else:
            self.sampling_model = AutoModelForCausalLM.from_pretrained(
                sampling_model_name,
                cache_dir=cache_dir
            ).to(self.device)
            self.sampling_tokenizer = AutoTokenizer.from_pretrained(
                sampling_model_name,
                cache_dir=cache_dir
            )
            if self.sampling_tokenizer.pad_token is None:
                self.sampling_tokenizer.pad_token = self.sampling_tokenizer.eos_token
            self.sampling_model.eval()
        
        # Get calibration parameters
        config_key = f'{sampling_model_name}_{scoring_model_name}'
        if config_key in self.CALIBRATION_PARAMS:
            params = self.CALIBRATION_PARAMS[config_key]
        else:
            # Use conservative defaults for unknown model pairs
            print(f"  Warning: No calibration for {config_key}, using defaults")
            params = {'mu0': 0.0, 'sigma0': 1.0, 'mu1': 1.5, 'sigma1': 1.5}
        
        self.mu0 = params['mu0']
        self.sigma0 = params['sigma0']
        self.mu1 = params['mu1']
        self.sigma1 = params['sigma1']
        
        print(f"Fast-DetectGPT loaded on {self.device}")
        print(f"  Calibration: mu0={self.mu0:.3f}, sigma0={self.sigma0:.3f}, mu1={self.mu1:.3f}, sigma1={self.sigma1:.3f}")
    
    def _compute_prob_norm(self, x: float) -> float:
        """Compute probability using normal distribution calibration"""
        pdf_value0 = self.norm.pdf(x, loc=self.mu0, scale=self.sigma0)
        pdf_value1 = self.norm.pdf(x, loc=self.mu1, scale=self.sigma1)
        prob = pdf_value1 / (pdf_value0 + pdf_value1 + 1e-10)
        return prob
    
    def _get_curvature(self, logits_ref: torch.Tensor, logits_score: torch.Tensor, 
                       labels: torch.Tensor) -> float:
        """
        Calculate conditional probability curvature analytically
        
        This is the FAST version - no perturbation sampling needed.
        Uses reference model's probability distribution to estimate curvature.
        
        Args:
            logits_ref: Reference/sampling model logits
            logits_score: Scoring model logits
            labels: Token labels
            
        Returns:
            Curvature score (positive = more likely AI)
        """
        # Handle vocabulary size mismatch between models
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]
        
        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        
        # Use scoring model's log probs but reference model's probability distribution
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
    
    @torch.no_grad()
    def detect(self, text: str) -> DetectionResult:
        """
        Detect if text is AI-generated using calibrated probability curvature
        
        Args:
            text: Input text to analyze
            
        Returns:
            DetectionResult with curvature-based score
        """
        # Validate input
        if not text or len(text.strip()) < 20:
            return DetectionResult(
                score=0.5,
                prediction="INSUFFICIENT_DATA",
                confidence=0.0,
                details={"error": "Text too short"}
            )
        
        # Tokenize with scoring tokenizer
        tokenized = self.scoring_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_token_type_ids=False
        ).to(self.device)
        
        n_tokens = tokenized.input_ids.shape[1]
        if n_tokens < 10:
            return DetectionResult(
                score=0.5,
                prediction="INSUFFICIENT_DATA",
                confidence=0.0,
                details={"error": "Too few tokens"}
            )
        
        labels = tokenized.input_ids[:, 1:]
        
        # Get scoring model logits
        logits_score = self.scoring_model(**tokenized).logits[:, :-1]
        
        # Get reference model logits (same or different model)
        if self.use_same_model:
            logits_ref = logits_score
        else:
            # Tokenize with sampling tokenizer
            tokenized_ref = self.sampling_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_token_type_ids=False
            ).to(self.device)
            logits_ref = self.sampling_model(**tokenized_ref).logits[:, :-1]
        
        # Calculate curvature
        curvature = self._get_curvature(logits_ref, logits_score, labels)
        
        # Use calibrated probability computation
        # Higher curvature = more likely AI
        ai_prob = self._compute_prob_norm(curvature)
        
        # Apply confidence based on token count (more tokens = more reliable)
        # Research suggests 100+ tokens for reliable detection
        token_factor = min(1.0, n_tokens / 100.0)
        
        # Scale confidence: near 0.5 = low confidence, near 0 or 1 = high confidence
        raw_confidence = abs(ai_prob - 0.5) * 2
        confidence = raw_confidence * token_factor
        
        # Threshold at 0.5
        prediction = "AI" if ai_prob > 0.5 else "Human"
        
        return DetectionResult(
            score=ai_prob,
            prediction=prediction,
            confidence=confidence,
            details={
                "raw_curvature": curvature,
                "n_tokens": n_tokens,
                "token_factor": token_factor,
                "scoring_model": self.scoring_model_name,
                "sampling_model": self.sampling_model_name,
                "method": "Conditional probability curvature (Fast-DetectGPT)"
            }
        )


# ============================================================================
# Ensemble Detector
# ============================================================================

class EnsembleDetector:
    """
    Full 3-component ensemble detector combining all three methods (v3.0).
    
    Architecture following research papers:
    - DeTeCtive: Style-based contrastive learning (RoBERTa + KNN)
    - Binoculars: Cross-model perplexity (GPT-2 + GPT-2-medium)
    - Fast-DetectGPT: Probability curvature (GPT-2-medium sampling + GPT-2 scoring)
    
    v3.0 Enhancements (Domain-Adaptive Detection):
    - Domain detection: FORMAL (Wikipedia/academic), INFORMAL (casual), TECHNICAL
    - Adaptive thresholds per domain to reduce false positives on formal text
    - Confidence-gated voting that returns UNCERTAIN instead of forced predictions
    - Burstiness-based pre-filtering for obvious human/AI cases
    - Works on latest models (GPT-4o, Claude 3.5, Gemini) via zero-shot Binoculars
    
    Weights optimized for low false positive rate:
    - DeTeCtive: 0.45 (best out-of-domain performance)
    - Binoculars: 0.35 (strong zero-shot capability on latest models)
    - Fast-DetectGPT: 0.20 (statistical validator)
    
    Model sharing: Binoculars and Fast-DetectGPT share the same GPT-2 and GPT-2-medium
    models to save memory while still using proper two-model architecture for both.
    """
    
    def __init__(
        self,
        detective_path: str = None,
        detective_db: str = None,
        include_binoculars: bool = True,
        include_fast_detect: bool = True,
        device: str = None,
        enable_domain_adaptation: bool = True
    ):
        """
        Initialize ensemble detector
        
        Args:
            detective_path: Path to DeTeCtive model (None for default)
            detective_db: Path to DeTeCtive database (None for default)
            include_binoculars: Include Binoculars detector
            include_fast_detect: Include Fast-DetectGPT detector
            device: Device to use
            enable_domain_adaptation: Enable domain-adaptive thresholding (v3.0)
            
        Note: Binoculars and Fast-DetectGPT share GPT-2/GPT-2-medium models
        automatically when both are enabled, saving memory while maintaining
        the research-validated two-model architecture.
        """
        self.device = device or get_device()
        self.enable_domain_adaptation = enable_domain_adaptation
        
        print("=" * 60)
        print("Initializing AI Text Detection Ensemble v3.0")
        print("=" * 60)
        
        # Initialize domain analyzer (v3.0)
        if enable_domain_adaptation:
            self.domain_analyzer = DomainAnalyzer()
            print("Domain-adaptive detection: ENABLED")
        else:
            self.domain_analyzer = None
            print("Domain-adaptive detection: DISABLED")
        
        # Initialize DeTeCtive
        self.detective = DeTeCtiveDetector(
            model_path=detective_path,
            database_path=detective_db,
            device=self.device
        )
        
        # Initialize Binoculars
        self.include_binoculars = include_binoculars
        if include_binoculars:
            self.binoculars = BinocularsDetector(
                device=self.device,
                use_bfloat16=False  # Disable for AMD GPU compatibility
            )
        else:
            self.binoculars = None
        
        # Initialize Fast-DetectGPT
        # IMPORTANT: Research shows best results with TWO DIFFERENT models
        # - Sampling model: provides reference probability distribution (larger model preferred)
        # - Scoring model: calculates actual log probabilities (can be smaller)
        self.include_fast_detect = include_fast_detect
        if include_fast_detect:
            if include_binoculars:
                # Use Binoculars' models: gpt2-medium (sampling) + gpt2 (scoring)
                # This follows the research principle of using two different models
                self.fast_detect = FastDetectGPTDetector(
                    scoring_model_name="gpt2",
                    sampling_model_name="gpt2-medium",  # Different model for proper curvature
                    device=self.device,
                    shared_scoring_model=self.binoculars.observer_model,  # gpt2
                    shared_sampling_model=self.binoculars.performer_model,  # gpt2-medium
                    shared_tokenizer=self.binoculars.tokenizer
                )
            else:
                # Load models independently with proper two-model setup
                self.fast_detect = FastDetectGPTDetector(
                    scoring_model_name="gpt2",
                    sampling_model_name="gpt2-medium",  # Different model
                    device=self.device
                )
        else:
            self.fast_detect = None
            
        # Initialize Provenance Analyzers (v3.2 - Enhanced)
        # Full provenance analysis with Unicode, Dash, Repetition, and Web Search
        from .provenance_analyzers import ProvenanceAggregator
        self.provenance_aggregator = ProvenanceAggregator()
        print("Provenance analyzers: ENABLED (Unicode, Dash, Repetition)")
        if self.provenance_aggregator.web_search and self.provenance_aggregator.web_search.enabled:
            print("Web Search Provenance: ENABLED") 
        
        # Set weights based on active components
        self._set_weights()
        
        print("\n" + "=" * 60)
        print("Ensemble ready!")
        print(f"Components: {list(self.weights.keys())}")
        print(f"Weights: {self.weights}")
        print("=" * 60 + "\n")
    
    def _set_weights(self):
        """Set detector weights based on active components
        
        v3.4 UPDATE: With Deepfake-based DeTeCtive model, calibrated for ~90% accuracy.
        Deepfake model is well-tested and provides good balance between precision and recall.
        
        Performance with Deepfake model on HC3 dataset:
        - Overall accuracy: ~90% (80% human detection, 100% AI detection)
        - Good balance between false positives and false negatives
        
        Strategy: 
        - DeTeCtive: Primary detector (Deepfake model is well-calibrated)
        - Binoculars: Secondary validator (helps with out-of-distribution text)
        - Fast-DetectGPT: Tertiary validator (helps with perplexity edge cases)
        """
        if self.include_binoculars and self.include_fast_detect:
            # v3.4: Optimized weights for Deepfake model - calibrated for ~90% accuracy
            # Balanced approach: DeTeCtive primary, Binoculars strong validator for human detection
            # Tested on HC3: 88-90% accuracy with these weights
            self.weights = {
                "detective": 0.42,   # Primary detector - well-calibrated
                "binoculars": 0.38,  # Strong validator - excellent for human detection
                "fast_detect": 0.20  # Tertiary validator - helps with edge cases
            }
        elif self.include_binoculars:
            self.weights = {
                "detective": 0.65,
                "binoculars": 0.35
            }
        elif self.include_fast_detect:
            self.weights = {
                "detective": 0.75,
                "fast_detect": 0.25
            }
        else:
            self.weights = {"detective": 1.0}
    
    def detect(self, text: str, return_breakdown: bool = True, enable_provenance: bool = False) -> EnsembleResult:
        """
        Detect if text is AI-generated using domain-adaptive ensemble approach (v3.0).
        
        Key improvements:
        1. Domain detection: FORMAL (Wikipedia/academic), INFORMAL (casual), TECHNICAL
        2. Adaptive thresholds per domain to reduce false positives on formal text
        3. Confidence-gated voting that returns UNCERTAIN instead of forced predictions
        4. Burstiness-based pre-filtering for obvious human/AI cases
        
        Args:
            text: Input text to analyze
            return_breakdown: Include individual detector results
            
        Returns:
            EnsembleResult with prediction, confidence, domain info, and breakdown
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for reliable detection")
        
        # ========================================
        # STEP 1: Domain Analysis (v3.0)
        # ========================================
        domain_info = None
        domain = "INFORMAL"  # Default
        burstiness = 0.5
        domain_adjusted = False
        
        if self.enable_domain_adaptation and self.domain_analyzer:
            domain_info = self.domain_analyzer.analyze(text)
            domain = domain_info.domain
            burstiness = domain_info.burstiness
            domain_adjusted = (domain != "INFORMAL")  # Non-default domain means adjustments applied
        
        # ========================================
        # STEP 2: Get Individual Detector Results
        # ========================================
        results = {}
        results["detective"] = self.detective.detect(text)
        
        if self.include_binoculars:
            results["binoculars"] = self.binoculars.detect(text)
        
        if self.include_fast_detect:
            results["fast_detect"] = self.fast_detect.detect(text)
            
        # ========================================
        # STEP 2.5: Provenance Analysis (v3.2 - Enhanced)
        # ========================================
        provenance_data = {}
        provenance_bundle = None
        provenance_ai_adjustment = 0.0  # Adjustment to AI likelihood based on provenance
        
        if enable_provenance and self.provenance_aggregator:
            print("Running provenance analysis...")
            # Run full provenance analysis
            # Web search only if explicitly requested and available
            run_web = (
                self.provenance_aggregator.web_search and 
                self.provenance_aggregator.web_search.enabled
            )
            provenance_bundle = self.provenance_aggregator.analyze(text, run_web_search=run_web)
            
            provenance_data = {
                "unicode": provenance_bundle.unicode,
                "dash": provenance_bundle.dash,
                "repetition": provenance_bundle.repetition,
                "web_search": provenance_bundle.web_search,
                "summary": {
                    "overall_score": provenance_bundle.overall_provenance_score,
                    "human_signals": provenance_bundle.human_signals,
                    "ai_signals": provenance_bundle.ai_signals,
                    "suggests_plagiarism": provenance_bundle.suggests_plagiarism,
                    "suggests_ai": provenance_bundle.suggests_ai,
                }
            }
            
            # Calculate adjustment based on provenance signals
            # Positive = more likely AI, Negative = more likely Human
            
            # Em-dash overuse is a strong AI signal
            if provenance_bundle.dash and provenance_bundle.dash.em_dash_overuse:
                provenance_ai_adjustment += 0.08
                print(f"  Provenance: Em-dash overuse detected (+0.08 AI)")
            
            # High repetition signature suggests AI
            if provenance_bundle.repetition and provenance_bundle.repetition.ai_repetition_signature > 0.5:
                adjustment = (provenance_bundle.repetition.ai_repetition_signature - 0.5) * 0.2
                provenance_ai_adjustment += adjustment
                print(f"  Provenance: Repetition pattern (+{adjustment:.2f} AI)")
            
            # Web matches suggest human copy-paste (not AI)
            if provenance_bundle.web_search and provenance_bundle.web_search.is_likely_copied:
                provenance_ai_adjustment -= 0.15
                print(f"  Provenance: Web matches found - likely copy-paste (-0.15 AI)")
            
            # Special characters suggest human/copy-paste
            if provenance_bundle.unicode and provenance_bundle.unicode.invisible_chars_count > 0:
                provenance_ai_adjustment -= 0.05
                print(f"  Provenance: Invisible chars found (-0.05 AI)")
            
            # Smart quotes suggest word processor (human)
            if provenance_bundle.unicode and provenance_bundle.unicode.smart_quotes_count > 2:
                provenance_ai_adjustment -= 0.03
                print(f"  Provenance: Smart quotes found (-0.03 AI)")
            
            print(f"  Provenance total adjustment: {provenance_ai_adjustment:+.2f}")
        
        # ========================================
        # STEP 3: Apply Domain-Adaptive Thresholds
        # ========================================
        # This adjusts predictions based on domain-specific thresholds
        # For formal text: require stronger AI signal to reduce false positives
        
        detective_score = results["detective"].score
        detective_pred = results["detective"].prediction
        
        bino_score = None
        bino_pred = None
        bino_raw = None
        if self.include_binoculars:
            bino_raw = results["binoculars"].details.get("raw_score", 1.0)
            bino_score = results["binoculars"].score
            bino_pred = results["binoculars"].prediction
        
        curv_score = None
        fast_pred = None
        fast_score = None
        if self.include_fast_detect:
            curv_score = results["fast_detect"].details.get("raw_curvature", 0)
            fast_score = results["fast_detect"].score
            fast_pred = results["fast_detect"].prediction
        
        # Apply domain-adaptive threshold adjustments
        adjusted_predictions = {}
        if domain_info and domain in ["FORMAL", "TECHNICAL"]:
            thresholds = domain_info.thresholds
            
            # Re-evaluate predictions with domain-specific thresholds
            # DeTeCtive: Higher threshold = require more AI votes
            adj_det_threshold = thresholds.get("detective", 0.50)
            adjusted_predictions["detective"] = "AI" if detective_score > adj_det_threshold else "Human"
            
            # Binoculars: Use raw score with adjusted threshold
            if bino_raw is not None:
                adj_bino_threshold = thresholds.get("binoculars", 0.71)
                # Lower raw score = more AI-like, so we flip the comparison
                adjusted_predictions["binoculars"] = "AI" if bino_raw < adj_bino_threshold else "Human"
            
            # Fast-DetectGPT: Use score with adjusted threshold
            if fast_score is not None:
                adj_fast_threshold = thresholds.get("fast_detect", 0.50)
                adjusted_predictions["fast_detect"] = "AI" if fast_score > adj_fast_threshold else "Human"
        else:
            # Use original predictions for informal text
            adjusted_predictions["detective"] = detective_pred
            if bino_pred:
                adjusted_predictions["binoculars"] = bino_pred
            if fast_pred:
                adjusted_predictions["fast_detect"] = fast_pred
        
        # ========================================
        # STEP 4: Calculate Ensemble Metrics
        # ========================================
        # Use ADJUSTED predictions for voting
        ai_votes = sum(1 for p in adjusted_predictions.values() if p == "AI")
        total = len(adjusted_predictions)
        agreement = f"{ai_votes}/{total}"
        
        # Calculate weighted ensemble score (using original scores)
        ensemble_score = sum(
            self.weights[name] * results[name].score
            for name in self.weights
        )
        
        # ========================================
        # STEP 5: Confidence-Gated Decision Logic
        # ========================================
        # Key insight: Don't force predictions when evidence is weak
        
        # Detect if DeTeCtive might be overconfident due to database limitations
        detective_is_extreme = detective_score <= 0.1 or detective_score >= 0.9
        detective_at_max = detective_score >= 0.95
        both_others_say_human = (
            adjusted_predictions.get("binoculars") == "Human" and 
            adjusted_predictions.get("fast_detect") == "Human"
        )
        
        # FORMAL TEXT ADJUSTMENT: Be more conservative
        # Research shows formal/encyclopedic text triggers false positives
        formal_confidence_penalty = 0.0
        if domain == "FORMAL":
            formal_confidence_penalty = 0.12  # Reduce confidence for formal text AI predictions
        elif domain == "TECHNICAL":
            formal_confidence_penalty = 0.06
        
        # Calculate base human votes
        human_votes = sum(1 for p in adjusted_predictions.values() if p == "Human")
        
        # Main decision logic with confidence gating
        if ai_votes == total:
            # All detectors agree on AI - high confidence
            prediction = "AI"
            base_confidence = max(ensemble_score, 0.85)
            # Apply formal text penalty
            confidence = max(base_confidence - formal_confidence_penalty, 0.6)
            
        elif ai_votes == 0:
            # All detectors agree on Human - but check for latest model signatures
            # IMPORTANT: Binoculars/Fast-DetectGPT fail on Claude 4.5, GPT-4o outputs
            
            # Get LLM signature from domain info (if available)
            llm_sig = domain_info.llm_signature if domain_info else 0.0
            
            # HIGH LLM SIGNATURE OVERRIDE
            # If text has strong LLM patterns (markdown, code blocks, etc.),
            # override the "all Human" consensus - latest models evade detectors
            if llm_sig >= 0.35:
                # Strong LLM signature - likely AI despite detectors saying Human
                prediction = "AI"
                # Confidence based on signature strength
                confidence = min(0.55 + llm_sig * 0.5, 0.85)
            elif llm_sig >= 0.20:
                # Moderate LLM signature - uncertain
                prediction = "UNCERTAIN"
                confidence = 0.50
            else:
                # Low LLM signature - trust the detectors
                prediction = "Human"
                confidence = max(1.0 - ensemble_score, 0.80)
                
                # But verify with raw signals for edge cases
                if curv_score is not None and curv_score > 3.0:
                    # High curvature is suspicious - reduce confidence
                    prediction = "Human"
                    confidence = 0.65
                elif bino_raw is not None and bino_raw < 0.60:
                    # Very low binoculars score is suspicious
                    prediction = "Human"
                    confidence = 0.65
                
        elif ai_votes >= 2:
            # Majority says AI (2/3 or more)
            # For formal text, require additional confirmation
            if domain == "FORMAL" and ai_votes == 2:
                # In formal domain, 2/3 is not enough - need strong signals
                strong_ai_signals = 0
                if detective_score > 0.75:
                    strong_ai_signals += 1
                if curv_score is not None and curv_score > 3.5:
                    strong_ai_signals += 1
                if bino_raw is not None and bino_raw < 0.60:
                    strong_ai_signals += 1
                
                if strong_ai_signals >= 2:
                    prediction = "AI"
                    confidence = 0.65
                else:
                    # Not enough strong signals for formal text
                    prediction = "UNCERTAIN"
                    confidence = 0.50
            else:
                # Standard case: trust the majority
                prediction = "AI"
                ai_confidence = max(
                    detective_score if adjusted_predictions.get("detective") == "AI" else 0,
                    (1.0 - bino_raw) if adjusted_predictions.get("binoculars") == "AI" and bino_raw else 0,
                    min(curv_score / 5.0, 1.0) if adjusted_predictions.get("fast_detect") == "AI" and curv_score else 0
                )
                confidence = max(ai_confidence - formal_confidence_penalty, ensemble_score - formal_confidence_penalty, 0.55)
                
        else:
            # Mixed signals - careful analysis required
            # CRITICAL FIX v3.1: When statistical detectors (Binoculars + Fast-DetectGPT) 
            # BOTH say Human, trust them over DeTeCtive which has database limitations.
            # DeTeCtive is trained on specific datasets and often fails on out-of-domain text.
            
            # v3.2 UPDATE: BUT for latest models (Claude 4.5, GPT-4o), the statistical detectors
            # often fail. Check LLM signature first - if high, trust DeTeCtive over others.
            
            # Get LLM signature 
            llm_sig = domain_info.llm_signature if domain_info else 0.0
            
            # CASE 1: Both Binoculars and Fast-DetectGPT say Human (strongest signal)
            if both_others_say_human:
                # v3.3: M4-based DeTeCtive is highly accurate - trust it more
                # Statistical detectors can fail on wikihow-style AI content (Bloomz, etc.)
                
                # If DeTeCtive says AI with VERY high confidence (>=0.9), trust it
                if adjusted_predictions.get("detective") == "AI" and detective_score >= 0.9:
                    # DeTeCtive is very confident - this is likely AI that evades stats
                    prediction = "AI"
                    confidence = min(0.65 + detective_score * 0.2, 0.85)
                # v3.2: Check if this is actually AI content with markdown/code formatting
                # If DeTeCtive says AI AND LLM signature is high, trust that combination
                elif adjusted_predictions.get("detective") == "AI" and llm_sig >= 0.25:
                    # DeTeCtive sees AI patterns AND text has LLM formatting
                    # This is likely Claude/GPT-4 content evading statistical detectors
                    prediction = "AI"
                    confidence = min(0.55 + llm_sig * 0.4 + detective_score * 0.2, 0.85)
                elif llm_sig >= 0.35:
                    # Very high LLM signature alone is enough
                    prediction = "AI"
                    confidence = min(0.55 + llm_sig * 0.5, 0.80)
                # v3.4: If DeTeCtive strongly says AI (>=0.9) but both others say Human, trust Human
                # If DeTeCtive moderately says AI (0.7-0.9), check Binoculars strength
                elif adjusted_predictions.get("detective") == "AI" and detective_score >= 0.9:
                    # DeTeCtive very confident but both stats say Human - trust Human
                    # Higher confidence if Binoculars is very strong (low raw score = human)
                    if bino_raw is not None and bino_raw > 0.75:
                        prediction = "Human"
                        confidence = 0.70
                    else:
                        prediction = "Human"
                        confidence = 0.60
                elif adjusted_predictions.get("detective") == "AI" and detective_score >= 0.7:
                    # Mixed signals - DeTeCtive confident but stats disagree
                    # If Binoculars strongly says Human, trust it
                    if bino_raw is not None and bino_raw > 0.75:
                        prediction = "Human"
                        confidence = 0.65
                    else:
                        prediction = "UNCERTAIN"
                        confidence = 0.50
                elif burstiness > 0.5:
                    # High burstiness + both statistical detectors = definitely human
                    prediction = "Human"
                    confidence = min(0.75 + burstiness * 0.15, 0.90)
                elif domain in ["FORMAL", "TECHNICAL"]:
                    # Formal/technical text - trust statistical detectors
                    prediction = "Human"
                    confidence = 0.70
                elif bino_raw is not None and bino_raw > 0.70:
                    # Binoculars strongly says human
                    prediction = "Human"
                    confidence = 0.65
                elif curv_score is not None and curv_score < 2.0:
                    # Low curvature = human writing patterns
                    prediction = "Human"
                    confidence = 0.65
                else:
                    # Edge case - DeTeCtive is very high but curvature is moderate
                    if detective_score >= 0.85 and curv_score is not None and curv_score > 2.5:
                        prediction = "UNCERTAIN"
                        confidence = 0.50
                    else:
                        prediction = "Human"
                        confidence = 0.60
            
            # CASE 2: DeTeCtive at extreme (95%+) but others disagree
            elif detective_at_max and human_votes >= 1:
                # v3.4: When DeTeCtive is at max (1.0) but statistical detectors say Human,
                # trust the statistical detectors - DeTeCtive can have false positives
                if both_others_say_human:
                    # Both Binoculars and Fast-DetectGPT say Human - trust them
                    prediction = "Human"
                    # Higher confidence if both strongly say human
                    if bino_raw is not None and bino_raw > 0.70 and curv_score is not None and curv_score < 1.5:
                        confidence = 0.70
                    else:
                        confidence = 0.60
                elif bino_raw is not None and bino_raw > 0.65:
                    prediction = "Human"
                    confidence = 0.60
                elif curv_score is not None and curv_score < 2.0:
                    prediction = "Human"
                    confidence = 0.60
                else:
                    prediction = "UNCERTAIN"
                    confidence = 0.45
            
            # CASE 3: DeTeCtive moderately confident AI (60-85%) with NO supporting evidence
            elif detective_score >= 0.6 and adjusted_predictions.get("detective") == "AI":
                # Check if statistical detectors support DeTeCtive
                has_statistical_support = False
                if curv_score is not None and curv_score > 2.5:
                    has_statistical_support = True
                if bino_raw is not None and bino_raw < 0.65:
                    has_statistical_support = True
                    
                if has_statistical_support:
                    prediction = "AI"
                    confidence = max(detective_score * 0.70 - formal_confidence_penalty, 0.55)
                elif domain in ["FORMAL", "TECHNICAL"]:
                    # No statistical support + formal domain = be conservative
                    prediction = "UNCERTAIN"
                    confidence = 0.50
                elif burstiness > 0.55:
                    # High burstiness = lean human
                    prediction = "UNCERTAIN"
                    confidence = 0.50
                else:
                    # Low burstiness + moderate DeTeCtive = mild AI lean
                    prediction = "AI"
                    confidence = max(detective_score * 0.55, 0.50)
            
            # CASE: Majority says Human
            elif human_votes >= 2:
                if detective_score <= 0.4:
                    prediction = "Human"
                    confidence = max(0.65, (1 - detective_score) * 0.75)
                elif curv_score is not None and curv_score > 3.0:
                    # High curvature is concerning
                    prediction = "UNCERTAIN"
                    confidence = 0.50
                else:
                    prediction = "Human"
                    confidence = 0.60
            
            # CASE: Strong statistical signals
            elif curv_score is not None and curv_score > 3.5 and detective_score >= 0.45:
                prediction = "AI"
                confidence = max(min(curv_score / 5.0, 0.70) - formal_confidence_penalty, 0.50)
            
            # CASE: Moderate AI signals
            elif detective_score >= 0.55 and not domain == "FORMAL":
                prediction = "AI"
                confidence = max(detective_score * 0.65, 0.50)
            
            # CASE: Low DeTeCtive score (leans human)
            elif detective_score <= 0.3:
                prediction = "Human"
                confidence = max((1 - detective_score) * 0.65, 0.55)
            
            # DEFAULT: Too mixed to decide
            else:
                prediction = "UNCERTAIN"
                confidence = 0.50
        
        # ========================================
        # STEP 6: Burstiness Boost for Human Detection
        # ========================================
        # High burstiness strongly suggests human writing
        if prediction == "Human" and burstiness > 0.7:
            confidence = min(confidence + 0.10, 0.95)
        elif prediction == "UNCERTAIN" and burstiness > 0.75:
            # Very high burstiness - lean toward human
            prediction = "Human"
            confidence = 0.60
        
        # Low burstiness in informal context with Human prediction - be cautious
        if prediction == "Human" and domain == "INFORMAL" and burstiness < 0.25:
            confidence = max(confidence - 0.10, 0.50)
        
        # ========================================
        # STEP 6.5: Provenance Signal Integration (v3.2)
        # ========================================
        # Apply provenance adjustments to final decision
        if enable_provenance and provenance_bundle:
            # Apply the provenance adjustment to confidence
            if provenance_ai_adjustment > 0:
                # More likely AI - if currently Human, reduce confidence
                if prediction == "Human":
                    confidence = max(confidence - provenance_ai_adjustment, 0.45)
                    # Strong AI signals can flip Human to UNCERTAIN
                    if provenance_ai_adjustment > 0.1 and confidence < 0.55:
                        prediction = "UNCERTAIN"
                        confidence = 0.50
                elif prediction == "AI":
                    # Reinforce AI prediction
                    confidence = min(confidence + provenance_ai_adjustment * 0.5, 0.95)
            
            elif provenance_ai_adjustment < 0:
                # More likely Human (e.g., web matches found)
                if prediction == "AI":
                    confidence = max(confidence + provenance_ai_adjustment, 0.45)
                    # Strong human signals can flip AI to UNCERTAIN
                    if provenance_ai_adjustment < -0.1 and confidence < 0.55:
                        prediction = "UNCERTAIN"
                        confidence = 0.50
                elif prediction == "Human":
                    # Reinforce Human prediction
                    confidence = min(confidence - provenance_ai_adjustment * 0.5, 0.95)
            
            # Special case: Web plagiarism detected
            # If exact web matches found, this strongly suggests human copy-paste
            if provenance_bundle.suggests_plagiarism:
                if prediction == "AI" and confidence < 0.75:
                    # Downgrade AI prediction if not very confident
                    prediction = "UNCERTAIN"
                    confidence = 0.50
        
        # ========================================
        # STEP 7: Final Confidence Clamping & Action
        # ========================================
        confidence = max(min(confidence, 1.0), 0.0)
        
        # Determine suggested action
        if prediction == "UNCERTAIN" or confidence < 0.55:
            suggested_action = "REVIEW"
        elif prediction == "AI" and confidence >= 0.75:
            suggested_action = "REJECT"
        elif prediction == "Human" and confidence >= 0.70:
            suggested_action = "ACCEPT"
        else:
            suggested_action = "REVIEW"
        
        # Create result object
        breakdown = results if return_breakdown else {}
        if enable_provenance:
            breakdown["provenance"] = provenance_data

        return EnsembleResult(
            prediction=prediction,
            confidence=confidence,
            agreement=agreement,
            ensemble_score=ensemble_score,
            breakdown=breakdown,
            suggested_action=suggested_action,
            domain=domain,
            domain_adjusted=domain_adjusted,
            burstiness=burstiness
        )
    
    def batch_detect(self, texts: List[str]) -> List[EnsembleResult]:
        """Detect multiple texts"""
        return [self.detect(text) for text in texts]


# ============================================================================
# Formatting
# ============================================================================

def format_result(result: EnsembleResult, show_breakdown: bool = True) -> str:
    """Format detection result for display"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"PREDICTION: {result.prediction}")
    lines.append(f"Confidence: {result.confidence:.2%}")
    lines.append(f"Agreement: {result.agreement}")
    lines.append(f"Ensemble Score: {result.ensemble_score:.4f}")
    lines.append(f"Suggested Action: {result.suggested_action}")
    
    if show_breakdown and result.breakdown:
        lines.append("\n" + "-" * 60)
        lines.append("DETECTOR BREAKDOWN:")
        
        for name, det_result in result.breakdown.items():
            lines.append(f"\n  {name.upper()}:")
            lines.append(f"    Prediction: {det_result.prediction}")
            lines.append(f"    Score: {det_result.score:.4f}")
            lines.append(f"    Confidence: {det_result.confidence:.4f}")
            
            if 'ai_votes' in det_result.details:
                lines.append(f"    Votes: {det_result.details['ai_votes']} AI, "
                           f"{det_result.details['human_votes']} Human")
            if 'raw_curvature' in det_result.details:
                lines.append(f"    Curvature: {det_result.details['raw_curvature']:.4f}")
            if 'raw_score' in det_result.details:
                lines.append(f"    Raw Score: {det_result.details['raw_score']:.4f}")
    
    lines.append("=" * 60)
    return "\n".join(lines)
