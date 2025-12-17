"""
Domain Analyzer for AI Text Detection

This module provides domain classification and adaptive threshold calculation
to reduce false positives on formal/encyclopedic text (Wikipedia-like) while
maintaining accuracy on latest AI models (GPT-4o, Claude 3.5, Gemini).

Key Features:
- Burstiness calculation (sentence variance × vocabulary richness)
- Formality detection (academic/formal word ratio)
- Domain classification: FORMAL, INFORMAL, TECHNICAL
- Adaptive threshold profiles per domain

Research basis:
- Group-adaptive thresholds improve F1 by 8-12% (arXiv:2502.04528)
- Reduce false positives by 40-60% on formal text (arXiv:2402.13671)
"""

import re
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class DomainInfo:
    """Domain analysis result with adaptive thresholds"""
    domain: str  # "FORMAL", "INFORMAL", "TECHNICAL"
    burstiness: float  # 0.0-1.0, higher = more varied/human-like
    formality: float  # 0.0-1.0, higher = more formal
    vocabulary_richness: float  # Type-token ratio
    avg_sentence_length: float
    thresholds: Dict[str, float]  # Adjusted thresholds per detector
    confidence_boost: float  # Adjustment to apply to confidence scores
    llm_signature: float = 0.0  # 0.0-1.0, higher = more LLM-like patterns (markdown, etc.)


# Formal/academic vocabulary indicators
# These words appear more frequently in formal/encyclopedic text
FORMAL_INDICATORS = frozenset([
    # Academic markers
    'however', 'therefore', 'furthermore', 'moreover', 'consequently',
    'nevertheless', 'accordingly', 'hence', 'thus', 'whereby',
    'wherein', 'thereof', 'whereas', 'notwithstanding', 'hitherto',
    # Scientific/technical
    'methodology', 'framework', 'implementation', 'mechanism', 'paradigm',
    'hypothesis', 'empirical', 'theoretical', 'systematic', 'comprehensive',
    'significant', 'substantial', 'approximately', 'respectively', 'primarily',
    # Formal connectors
    'specifically', 'particularly', 'notably', 'essentially', 'fundamentally',
    'additionally', 'alternatively', 'subsequently', 'previously', 'ultimately',
    # Academic verbs
    'demonstrate', 'indicate', 'suggest', 'establish', 'determine',
    'analyze', 'examine', 'investigate', 'evaluate', 'assess',
    'illustrate', 'emphasize', 'facilitate', 'utilize', 'implement',
    # Noun markers
    'phenomenon', 'characteristics', 'properties', 'aspects', 'factors',
    'implications', 'applications', 'considerations', 'parameters', 'variables',
])

# Informal indicators - casual/conversational markers
INFORMAL_INDICATORS = frozenset([
    # Contractions (detected separately)
    # Casual expressions
    'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'dunno', 'lemme',
    'yeah', 'yep', 'nope', 'nah', 'huh', 'hmm', 'umm', 'ugh',
    'wow', 'whoa', 'yay', 'meh', 'lol', 'lmao', 'omg', 'wtf', 'idk',
    'btw', 'tbh', 'imo', 'imho', 'fyi', 'afaik', 'asap',
    # Casual phrases
    'stuff', 'things', 'bunch', 'tons', 'loads', 'super', 'totally',
    'literally', 'basically', 'actually', 'honestly', 'seriously',
    'pretty', 'really', 'kinda', 'sorta', 'maybe', 'probably',
    # Filler words
    'like', 'just', 'well', 'so', 'anyway', 'anyways', 'whatever',
])


class DomainAnalyzer:
    """
    Analyzes text domain and provides adaptive detection thresholds.
    
    This helps reduce false positives on formal/encyclopedic text by:
    1. Detecting text style (formal vs informal vs technical)
    2. Calculating burstiness (human text has more variance)
    3. Adjusting detection thresholds based on domain
    
    Usage:
        analyzer = DomainAnalyzer()
        info = analyzer.analyze("Your text here...")
        # Use info.thresholds for domain-adjusted detection
    """
    
    # Default thresholds (standard detection)
    DEFAULT_THRESHOLDS = {
        "detective": 0.50,
        "binoculars": 0.71,  # Calibrated for GPT-2 pair
        "fast_detect": 0.50,
    }
    
    # Domain-specific threshold adjustments
    # Higher thresholds = stricter (fewer AI predictions = fewer false positives)
    DOMAIN_THRESHOLDS = {
        "FORMAL": {
            # Stricter thresholds for formal text to reduce false positives
            "detective": 0.65,      # Require stronger AI signal
            "binoculars": 0.65,     # Lower raw score needed (more likely AI)
            "fast_detect": 0.60,    # Higher probability needed
            "confidence_boost": -0.15,  # Reduce confidence for formal text
        },
        "INFORMAL": {
            # Standard thresholds for casual text
            "detective": 0.50,
            "binoculars": 0.71,
            "fast_detect": 0.50,
            "confidence_boost": 0.0,
        },
        "TECHNICAL": {
            # Moderately strict for technical/academic
            "detective": 0.58,
            "binoculars": 0.68,
            "fast_detect": 0.55,
            "confidence_boost": -0.08,
        },
    }
    
    def __init__(self):
        # Precompile regex patterns for efficiency
        self._sentence_pattern = re.compile(r'[.!?]+')
        self._word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        self._contraction_pattern = re.compile(r"\b\w+'\w+\b")
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        # LLM signature patterns (Claude, GPT-4, etc.)
        self._md_header_pattern = re.compile(r'^#+\s+.+$', re.MULTILINE)
        self._md_bold_pattern = re.compile(r'\*\*[^*]+\*\*')
        self._md_italic_pattern = re.compile(r'(?<!\*)\*[^*]+\*(?!\*)')
        self._code_block_pattern = re.compile(r'```[\s\S]*?```')
        self._inline_code_pattern = re.compile(r'`[^`]+`')
        self._bullet_list_pattern = re.compile(r'^\s*[-*•]\s+.+$', re.MULTILINE)
        self._numbered_list_pattern = re.compile(r'^\s*\d+[.)]\s+.+$', re.MULTILINE)
        self._step_pattern = re.compile(r'\b(step\s*\d+|first|second|third|finally|lastly|in conclusion)\b', re.IGNORECASE)
    
    def analyze(self, text: str) -> DomainInfo:
        """
        Analyze text domain and return adaptive thresholds.
        
        Args:
            text: Input text to analyze
            
        Returns:
            DomainInfo with domain classification and adjusted thresholds
        """
        if not text or len(text.strip()) < 20:
            # Return neutral for very short text
            return DomainInfo(
                domain="INFORMAL",
                burstiness=0.5,
                formality=0.5,
                vocabulary_richness=0.5,
                avg_sentence_length=15.0,
                thresholds=self.DEFAULT_THRESHOLDS.copy(),
                confidence_boost=0.0
            )
        
        # Calculate features
        burstiness = self._calculate_burstiness(text)
        formality = self._calculate_formality(text)
        vocab_richness = self._calculate_vocabulary_richness(text)
        avg_sent_len = self._calculate_avg_sentence_length(text)
        llm_signature = self._calculate_llm_signature(text)
        
        # Classify domain based on features
        domain = self._classify_domain(burstiness, formality, vocab_richness, avg_sent_len, text)
        
        # Get domain-specific thresholds
        domain_config = self.DOMAIN_THRESHOLDS.get(domain, self.DOMAIN_THRESHOLDS["INFORMAL"])
        thresholds = {
            "detective": domain_config["detective"],
            "binoculars": domain_config["binoculars"],
            "fast_detect": domain_config["fast_detect"],
        }
        confidence_boost = domain_config["confidence_boost"]
        
        return DomainInfo(
            domain=domain,
            burstiness=burstiness,
            formality=formality,
            vocabulary_richness=vocab_richness,
            avg_sentence_length=avg_sent_len,
            thresholds=thresholds,
            confidence_boost=confidence_boost,
            llm_signature=llm_signature
        )
    
    def _calculate_burstiness(self, text: str) -> float:
        """
        Calculate burstiness score (sentence length variance × vocab richness).
        
        Human text tends to have higher burstiness (more varied sentence lengths
        and vocabulary). AI text tends to be more uniform.
        
        Returns:
            Float 0.0-1.0, higher = more bursty/human-like
        """
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return 0.5  # Insufficient data
        
        # Calculate sentence length variance
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(lengths) < 2:
            return 0.5
        
        mean_len = sum(lengths) / len(lengths)
        if mean_len == 0:
            return 0.5
            
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_dev = math.sqrt(variance)
        
        # Coefficient of variation (normalized variance)
        cv = std_dev / mean_len if mean_len > 0 else 0
        
        # Also factor in vocabulary richness
        vocab_richness = self._calculate_vocabulary_richness(text)
        
        # Combine: CV contributes 60%, vocab richness 40%
        # Normalize CV (typical range 0.2-1.0 for most text)
        normalized_cv = min(cv / 0.8, 1.0)
        
        burstiness = 0.6 * normalized_cv + 0.4 * vocab_richness
        return min(max(burstiness, 0.0), 1.0)
    
    def _calculate_formality(self, text: str) -> float:
        """
        Calculate formality score based on vocabulary and style markers.
        
        Returns:
            Float 0.0-1.0, higher = more formal
        """
        text_lower = text.lower()
        words = self._word_pattern.findall(text_lower)
        
        if len(words) < 5:
            return 0.5
        
        # Count formal and informal indicators
        formal_count = sum(1 for w in words if w in FORMAL_INDICATORS)
        informal_count = sum(1 for w in words if w in INFORMAL_INDICATORS)
        
        # Check for contractions (informal)
        contractions = len(self._contraction_pattern.findall(text))
        informal_count += contractions * 2  # Weight contractions
        
        # Check for emojis (informal)
        emojis = len(self._emoji_pattern.findall(text))
        informal_count += emojis * 3  # Weight emojis heavily
        
        # Check for exclamation marks (informal)
        exclamations = text.count('!')
        informal_count += min(exclamations, 5)  # Cap at 5
        
        # Check for question marks in non-formal context
        questions = text.count('?')
        
        # Calculate formal vs informal ratio
        total_markers = formal_count + informal_count
        if total_markers == 0:
            # No clear markers - use sentence structure
            avg_sent_len = self._calculate_avg_sentence_length(text)
            # Longer sentences tend to be more formal
            return min(avg_sent_len / 30, 1.0) * 0.6 + 0.2
        
        formality = formal_count / total_markers
        
        # Adjust based on additional factors
        # Complex punctuation (semicolons, colons) indicate formality
        semicolons = text.count(';') + text.count(':')
        formality += min(semicolons * 0.02, 0.1)
        
        # Parenthetical content indicates formality
        parens = text.count('(') + text.count('[')
        formality += min(parens * 0.02, 0.1)
        
        return min(max(formality, 0.0), 1.0)
    
    def _calculate_llm_signature(self, text: str) -> float:
        """
        Detect patterns characteristic of latest LLMs (Claude, GPT-4, etc.).
        
        These models often produce:
        - Markdown formatting (headers, bold, code blocks)
        - Structured lists (bullet points, numbered)
        - Step-by-step explanations
        - Consistent, polished prose with low variance
        - Certain phrase patterns ("Let me explain", "Here's", "Great question")
        
        Returns:
            Float 0.0-1.0, higher = more LLM-like patterns
        """
        score = 0.0
        text_len = len(text)
        if text_len < 50:
            return 0.0
        
        # Markdown headers (strong signal)
        md_headers = len(self._md_header_pattern.findall(text))
        if md_headers > 0:
            score += min(md_headers * 0.12, 0.35)
        
        # Bold/italic text
        bold_count = len(self._md_bold_pattern.findall(text))
        italic_count = len(self._md_italic_pattern.findall(text))
        if bold_count > 0:
            score += min(bold_count * 0.05, 0.15)
        if italic_count > 0:
            score += min(italic_count * 0.03, 0.10)
        
        # Code blocks (very strong signal)
        code_blocks = len(self._code_block_pattern.findall(text))
        if code_blocks > 0:
            score += min(code_blocks * 0.15, 0.35)
        
        # Inline code
        inline_code = len(self._inline_code_pattern.findall(text))
        if inline_code > 0:
            score += min(inline_code * 0.02, 0.10)
        
        # Bullet lists
        bullet_items = len(self._bullet_list_pattern.findall(text))
        if bullet_items >= 2:
            score += min(bullet_items * 0.04, 0.15)
        
        # Numbered lists
        numbered_items = len(self._numbered_list_pattern.findall(text))
        if numbered_items >= 2:
            score += min(numbered_items * 0.05, 0.15)
        
        # Step-by-step patterns
        step_matches = len(self._step_pattern.findall(text))
        if step_matches >= 2:
            score += min(step_matches * 0.05, 0.15)
        
        # LLM phrase patterns (Claude/GPT-4 specific)
        text_lower = text.lower()
        llm_phrases = [
            "let me explain", "here's", "let's break", "i'll walk you through",
            "great question", "that's a great", "happy to help", "i'd be happy",
            "let's explore", "let's dive", "let's look at", "to summarize",
            "in summary", "key points:", "key takeaways", "important to note",
            "it's worth noting", "keep in mind", "as I mentioned", "as noted",
            "fundamentally,", "essentially,", "basically,", "specifically,",
            "to be more precise", "to clarify", "for context", "for example,",
            "comprehensive", "robust", "seamless", "leverage", "utilize",
            "straightforward", "delve", "intricacies", "nuanced", "realm"
        ]
        phrase_count = sum(1 for phrase in llm_phrases if phrase in text_lower)
        if phrase_count >= 2:
            score += min(phrase_count * 0.04, 0.20)
        
        # Emoji-free technical content (LLMs rarely use emojis unless asked)
        if not self._emoji_pattern.search(text) and len(text) > 200:
            # Check if content is substantive (not casual)
            words = self._word_pattern.findall(text)
            if len(words) > 50:
                avg_word_len = sum(len(w) for w in words) / len(words)
                if avg_word_len > 5.5:  # Longer words = more formal/AI-like
                    score += 0.05
        
        # Perfect grammar indicator: very few contractions in long formal text
        contractions = len(self._contraction_pattern.findall(text))
        words = self._word_pattern.findall(text)
        if len(words) > 100 and contractions < 2:
            # Lack of contractions in long text is suspicious
            score += 0.05
        
        return min(max(score, 0.0), 1.0)
    
    def _calculate_vocabulary_richness(self, text: str) -> float:
        """
        Calculate type-token ratio (unique words / total words).
        
        Returns:
            Float 0.0-1.0, higher = richer vocabulary
        """
        words = self._word_pattern.findall(text.lower())
        if len(words) < 5:
            return 0.5
        
        unique_words = set(words)
        
        # Simple TTR (adjusted for text length as longer texts have lower TTR)
        ttr = len(unique_words) / len(words)
        
        # Adjust for text length (TTR naturally decreases with length)
        # Add a small bonus for longer texts with maintained diversity
        length_factor = min(len(words) / 200, 1.0)
        adjusted_ttr = ttr + (1 - ttr) * length_factor * 0.2
        
        return min(max(adjusted_ttr, 0.0), 1.0)
    
    def _calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average words per sentence."""
        sentences = self._split_sentences(text)
        if not sentences:
            return 15.0  # Default
        
        total_words = sum(len(s.split()) for s in sentences if s.strip())
        return total_words / len(sentences) if sentences else 15.0
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split on sentence-ending punctuation
        sentences = self._sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _classify_domain(
        self, 
        burstiness: float, 
        formality: float, 
        vocab_richness: float,
        avg_sent_len: float,
        text: str
    ) -> str:
        """
        Classify text into domain category.
        
        Domain categories:
        - FORMAL: Wikipedia, academic papers, encyclopedic content
        - INFORMAL: Casual conversation, social media, personal writing
        - TECHNICAL: Mixed formal/informal, technical documentation
        """
        # Strong formal indicators
        is_highly_formal = (
            formality > 0.5 and 
            burstiness < 0.35 and 
            avg_sent_len > 18
        )
        
        # Strong informal indicators
        is_highly_informal = (
            formality < 0.3 or 
            burstiness > 0.65 or
            self._has_strong_informal_markers(text)
        )
        
        # Classification logic
        if is_highly_formal:
            return "FORMAL"
        elif is_highly_informal:
            return "INFORMAL"
        elif formality > 0.4 or avg_sent_len > 20:
            return "TECHNICAL"
        else:
            return "INFORMAL"
    
    def _has_strong_informal_markers(self, text: str) -> bool:
        """Check for strong informal markers like emojis or slang."""
        # Check for emojis
        if self._emoji_pattern.search(text):
            return True
        
        # Check for multiple exclamation marks
        if '!!' in text or '???' in text:
            return True
        
        # Check for internet slang
        text_lower = text.lower()
        slang_count = sum(1 for word in ['lol', 'lmao', 'omg', 'wtf', 'idk', 'tbh'] 
                         if word in text_lower)
        return slang_count >= 2
    
    def get_prefilter_result(self, text: str) -> Tuple[str, float]:
        """
        Quick pre-filter check for obvious cases.
        
        Returns:
            Tuple of (result, confidence) where result is:
            - "LIKELY_HUMAN": High burstiness suggests human
            - "LIKELY_AI": Low burstiness suggests AI
            - "UNCERTAIN": Need full detection pipeline
        """
        info = self.analyze(text)
        
        # High burstiness + informal = likely human
        if info.burstiness > 0.75 and info.formality < 0.3:
            return ("LIKELY_HUMAN", 0.7)
        
        # Very high burstiness = likely human
        if info.burstiness > 0.85:
            return ("LIKELY_HUMAN", 0.6)
        
        # Very low burstiness + high formality = could be either
        # (formal human writing also has low burstiness)
        if info.burstiness < 0.2 and info.formality > 0.6:
            # This is the Wikipedia trap - don't pre-classify
            return ("UNCERTAIN", 0.5)
        
        # Low burstiness + moderate formality = uncertain
        if info.burstiness < 0.25:
            return ("UNCERTAIN", 0.5)
        
        return ("UNCERTAIN", 0.5)


# Convenience function for quick analysis
def analyze_domain(text: str) -> DomainInfo:
    """Quick domain analysis using default analyzer."""
    analyzer = DomainAnalyzer()
    return analyzer.analyze(text)
