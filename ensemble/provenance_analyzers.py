"""
Provenance Analyzers Module v2.0

This module provides specialized analyzers for:
1. Unicode & Special Character Fingerprinting (enhanced)
2. Dash & Hyphen Usage Analysis with AI overuse detection
3. Web Search Provenance (Chunked Exact/Semantic Matching)
4. Repetition Pattern Analysis (AI tends to be repetitive)
5. Stylometric Anomaly Detection

These analyzers provide additional signals to the ensemble detector
to help distinguish between human and AI text, and to identify
potential sources of the text (plagiarism/copy-paste detection).

Key insight: If exact web matches found → likely human copy-paste
             AI generates novel text, but has distinct patterns

Feature Flags: Default OFF for backward compatibility
"""

import re
import os
import unicodedata
import collections
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Try to import requests for web search
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available - web search disabled")

# ============================================================================
# Configuration & Feature Flags
# ============================================================================

# Default feature flags - OFF by default for backward compatibility
DEFAULT_FEATURE_FLAGS = {
    "unicode_analysis": True,      # Basic Unicode analysis
    "dash_analysis": True,         # Dash pattern analysis
    "web_provenance": False,       # Requires API key
    "repetition_analysis": True,   # Check for repetitive patterns
    "stylometric_analysis": True,  # Advanced style analysis
}

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class UnicodeResult:
    """Result from Unicode analysis"""
    has_special_chars: bool
    special_char_ratio: float
    script_mix_score: float  # 0.0 = single script, 1.0 = highly mixed
    invisible_chars_count: int
    suspicious_patterns: List[str]
    # v2.0: Additional fields
    rare_punctuation: Dict[str, int]  # Rare punctuation marks found
    emoji_count: int
    smart_quotes_count: int  # Smart quotes indicate copy-paste or word processor
    anomaly_score: float  # 0-1, higher = more anomalous
    details: Dict[str, Any]


@dataclass
class DashResult:
    """Result from Dash/Hyphen analysis with AI pattern detection"""
    dash_types_found: List[str]
    dash_counts: Dict[str, int]
    consistent_usage: bool
    primary_dash: str
    suspicious_spacing: bool  # e.g. "word - word" vs "word-word" inconsistency
    # v2.0: AI-specific patterns
    em_dash_overuse: bool  # AI tends to overuse em-dashes
    em_dash_ratio: float   # Ratio of em-dashes to total punctuation
    ai_dash_signature: float  # 0-1, higher = more AI-like dash usage
    normalized_text: str   # Text with dashes normalized (optional)
    details: Dict[str, Any]


@dataclass
class ProvenanceResult:
    """Result from Web Search Provenance"""
    found_matches: bool
    max_similarity: float
    sources: List[Dict[str, Any]]
    exact_quote_matches: int
    # v2.0: Enhanced provenance
    plagiarism_score: float  # 0-1, higher = more likely plagiarized
    unique_sources_count: int
    is_likely_copied: bool   # If True, suggests human copy-paste rather than AI
    details: Dict[str, Any]


@dataclass
class RepetitionResult:
    """Result from Repetition Analysis - AI tends to be repetitive"""
    repetition_score: float  # 0-1, higher = more repetitive (AI-like)
    repeated_phrases: List[Tuple[str, int]]  # (phrase, count)
    sentence_similarity_avg: float
    paragraph_similarity_avg: float
    ai_repetition_signature: float  # Combined signal
    details: Dict[str, Any]


@dataclass
class ProvenanceBundle:
    """Combined provenance analysis results"""
    unicode: Optional[UnicodeResult] = None
    dash: Optional[DashResult] = None
    web_search: Optional[ProvenanceResult] = None
    repetition: Optional[RepetitionResult] = None
    
    # Aggregate scores
    overall_provenance_score: float = 0.0  # 0-1, contribution to AI likelihood
    human_signals: List[str] = field(default_factory=list)  # List of human indicators
    ai_signals: List[str] = field(default_factory=list)  # List of AI indicators
    
    # Decision support
    suggests_plagiarism: bool = False  # If True, likely human copy-paste
    suggests_ai: bool = False  # If True, provenance signals point to AI
    confidence: float = 0.0  # Confidence in provenance assessment


# ============================================================================
# Unicode Analyzer (Enhanced v2.0)
# ============================================================================

class UnicodeAnalyzer:
    """
    Analyzes text for rare characters, diacritics, invisible characters,
    and script mixing patterns that might indicate obfuscation or specific
    encoding artifacts.
    
    v2.0 Enhancements:
    - Smart quote detection (indicates word processor/copy-paste)
    - Rare punctuation cataloging
    - AI-specific character patterns
    - Anomaly scoring
    """
    
    # Common invisible characters
    INVISIBLE_CHARS = {
        '\u200b': 'Zero Width Space',
        '\u200c': 'Zero Width Non-Joiner',
        '\u200d': 'Zero Width Joiner',
        '\u200e': 'Left-To-Right Mark',
        '\u200f': 'Right-To-Left Mark',
        '\u202a': 'Left-To-Right Embedding',
        '\u202b': 'Right-To-Left Embedding',
        '\u202c': 'Pop Directional Formatting',
        '\u202d': 'Left-To-Right Override',
        '\u202e': 'Right-To-Left Override',
        '\ufeff': 'Zero Width No-Break Space (BOM)',
        '\u00a0': 'Non-Breaking Space',
        '\u2060': 'Word Joiner',
        '\u2061': 'Function Application',
        '\u2062': 'Invisible Times',
        '\u2063': 'Invisible Separator',
        '\u2064': 'Invisible Plus',
        '\u180e': 'Mongolian Vowel Separator',
    }
    
    # Smart quotes and typographic characters (indicate word processor)
    SMART_QUOTES = {
        '\u2018': 'Left Single Quote',   # '
        '\u2019': 'Right Single Quote',  # '
        '\u201c': 'Left Double Quote',   # "
        '\u201d': 'Right Double Quote',  # "
        '\u201a': 'Single Low Quote',    # ‚
        '\u201e': 'Double Low Quote',    # „
        '\u2039': 'Left Angle Quote',    # ‹
        '\u203a': 'Right Angle Quote',   # ›
        '\u00ab': 'Left Guillemet',      # «
        '\u00bb': 'Right Guillemet',     # »
    }
    
    # Rare punctuation that might indicate specific origins
    RARE_PUNCTUATION = {
        '\u2026': 'Horizontal Ellipsis',  # …
        '\u2022': 'Bullet',               # •
        '\u2023': 'Triangle Bullet',      # ‣
        '\u25e6': 'White Bullet',         # ◦
        '\u2043': 'Hyphen Bullet',        # ⁃
        '\u00b7': 'Middle Dot',           # ·
        '\u2027': 'Hyphenation Point',    # ‧
        '\u00a7': 'Section Sign',         # §
        '\u00b6': 'Pilcrow',              # ¶
        '\u2020': 'Dagger',               # †
        '\u2021': 'Double Dagger',        # ‡
        '\u2032': 'Prime',                # ′
        '\u2033': 'Double Prime',         # ″
        '\u2034': 'Triple Prime',         # ‴
        '\u2030': 'Per Mille',            # ‰
        '\u2031': 'Per Ten Thousand',     # ‱
    }
    
    def __init__(self):
        # Precompile emoji pattern
        self._emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA00-\U0001FA6F"  # chess symbols etc
            "\U0001FA70-\U0001FAFF"  # symbols ext
            "\U00002600-\U000026FF"  # misc symbols
            "]+",
            flags=re.UNICODE
        )
    
    def analyze(self, text: str) -> UnicodeResult:
        """
        Analyze text for unicode anomalies.
        
        Returns:
            UnicodeResult with comprehensive character analysis
        """
        if not text:
            return UnicodeResult(
                has_special_chars=False, 
                special_char_ratio=0.0, 
                script_mix_score=0.0, 
                invisible_chars_count=0, 
                suspicious_patterns=[],
                rare_punctuation={},
                emoji_count=0,
                smart_quotes_count=0,
                anomaly_score=0.0,
                details={}
            )
            
        total_chars = len(text)
        special_chars_count = 0
        invisible_chars_count = 0
        smart_quotes_count = 0
        emoji_count = 0
        scripts = collections.defaultdict(int)
        suspicious = []
        rare_punct = {}
        invisible_found = {}
        
        # Character-level analysis
        for char in text:
            # Check for invisible characters
            if char in self.INVISIBLE_CHARS:
                invisible_chars_count += 1
                name = self.INVISIBLE_CHARS[char]
                invisible_found[name] = invisible_found.get(name, 0) + 1
                if name not in suspicious:
                    suspicious.append(f"Invisible: {name}")
            
            # Check for smart quotes
            if char in self.SMART_QUOTES:
                smart_quotes_count += 1
            
            # Check for rare punctuation
            if char in self.RARE_PUNCTUATION:
                name = self.RARE_PUNCTUATION[char]
                rare_punct[name] = rare_punct.get(name, 0) + 1
            
            # Check for non-ASCII (excluding common handled chars)
            code_point = ord(char)
            if code_point > 255:
                # Exclude already-handled smart quotes/dashes/common unicode
                common_unicode = set([
                    0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026,  # Dashes, quotes, ellipsis
                    0x2022, 0x00B7,  # Bullets
                ])
                if code_point not in common_unicode and char not in self.SMART_QUOTES:
                    special_chars_count += 1
            
            # Script detection
            try:
                script = unicodedata.name(char).split()[0]
                scripts[script] += 1
            except ValueError:
                pass  # Control characters might not have names
        
        # Count emojis
        emojis = self._emoji_pattern.findall(text)
        emoji_count = sum(len(e) for e in emojis)

        # Calculate metrics
        special_char_ratio = special_chars_count / total_chars if total_chars > 0 else 0
        
        # Script mixing score (entropy-like)
        total_script_chars = sum(scripts.values())
        if total_script_chars > 0:
            primary_script_count = max(scripts.values())
            # 0.0 if 100% one script, higher if mixed
            script_mix_score = 1.0 - (primary_script_count / total_script_chars)
        else:
            script_mix_score = 0.0
            
        # Homoglyph detection (simplified)
        # Check if we have mixed Latin and Cyrillic/Greek which is common in obfuscation
        if 'LATIN' in scripts and ('CYRILLIC' in scripts or 'GREEK' in scripts):
            suspicious.append("Potential Homoglyph Mixing (Latin + Cyrillic/Greek)")
        
        # Calculate anomaly score
        # Higher score = more anomalous (potentially obfuscated or special origin)
        anomaly_score = 0.0
        
        # Invisible characters are suspicious
        if invisible_chars_count > 0:
            anomaly_score += min(invisible_chars_count * 0.1, 0.5)
        
        # High special character ratio is anomalous
        anomaly_score += min(special_char_ratio * 2, 0.3)
        
        # Script mixing is anomalous
        if script_mix_score > 0.1:
            anomaly_score += script_mix_score * 0.2
        
        # Smart quotes indicate word processor (slight anomaly)
        if smart_quotes_count > 0:
            anomaly_score += 0.05
        
        anomaly_score = min(anomaly_score, 1.0)

        return UnicodeResult(
            has_special_chars=special_chars_count > 0 or invisible_chars_count > 0,
            special_char_ratio=special_char_ratio,
            script_mix_score=script_mix_score,
            invisible_chars_count=invisible_chars_count,
            suspicious_patterns=suspicious,
            rare_punctuation=rare_punct,
            emoji_count=emoji_count,
            smart_quotes_count=smart_quotes_count,
            anomaly_score=anomaly_score,
            details={
                "scripts": dict(scripts),
                "special_count": special_chars_count,
                "total_chars": total_chars,
                "invisible_found": invisible_found,
                "has_smart_quotes": smart_quotes_count > 0,
            }
        )


# ============================================================================
# Dash Analyzer (Enhanced v2.0)
# ============================================================================

class DashAnalyzer:
    """
    Analyzes usage of dashes and hyphens.
    
    v2.0 Enhancements:
    - AI overuse detection: AI models tend to overuse em-dashes
    - Spacing pattern analysis
    - AI signature scoring
    
    Key insight from research:
    AI models (especially Claude, GPT-4) tend to overuse em-dashes as
    a universal connector/pause. Humans use more varied punctuation.
    """
    
    DASHES = {
        '-': ('Hyphen-minus (U+002D)', 'hyphen'),
        '‐': ('Hyphen (U+2010)', 'hyphen'),
        '‑': ('Non-breaking hyphen (U+2011)', 'hyphen'),
        '‒': ('Figure dash (U+2012)', 'figure'),
        '–': ('En dash (U+2013)', 'en'),
        '—': ('Em dash (U+2014)', 'em'),
        '―': ('Horizontal bar (U+2015)', 'bar'),
    }
    
    # Typical em-dash ratio in human text (per 1000 chars)
    # Research shows AI models (especially Claude, GPT-4) heavily overuse em-dashes
    HUMAN_EM_DASH_RATIO = 0.3  # Humans rarely use em-dashes (~0.3 per 1000 chars)
    AI_EM_DASH_RATIO = 2.0     # AI uses ~2.0+ em-dashes per 1000 chars
    EM_DASH_SIGNIFICANT = 0.5  # Any ratio above this is worth noting
    
    def analyze(self, text: str, normalize: bool = False) -> DashResult:
        """
        Analyze dash usage patterns.
        
        Args:
            text: Input text
            normalize: If True, also return normalized text
        """
        if not text:
            return DashResult(
                dash_types_found=[],
                dash_counts={},
                consistent_usage=True,
                primary_dash="None",
                suspicious_spacing=False,
                em_dash_overuse=False,
                em_dash_ratio=0.0,
                ai_dash_signature=0.0,
                normalized_text="" if normalize else "",
                details={}
            )
        
        counts = {name: 0 for name, (_, _) in self.DASHES.items()}
        dash_names = {name: info[0] for name, info in self.DASHES.items()}
        found_types = []
        
        # Count occurrences
        # Check double hyphen first to avoid double counting
        double_hyphen_count = text.count('--')
        
        # Replace double hyphens to count single hyphens correctly
        temp_text = text.replace('--', '\x00\x00')  # Temporary replacement
        
        for char in self.DASHES:
            c = temp_text.count(char)
            counts[char] = c
            if c > 0:
                found_types.append(dash_names[char])
        
        if double_hyphen_count > 0:
            found_types.append('Double hyphen')
        
        # Calculate counts by type
        count_by_name = {dash_names[char]: counts[char] for char in self.DASHES}
        count_by_name['Double hyphen'] = double_hyphen_count

        # Determine primary dash style
        total_dashes = sum(counts.values()) + double_hyphen_count
        if total_dashes == 0:
            return DashResult(
                dash_types_found=[],
                dash_counts=count_by_name,
                consistent_usage=True,
                primary_dash="None",
                suspicious_spacing=False,
                em_dash_overuse=False,
                em_dash_ratio=0.0,
                ai_dash_signature=0.0,
                normalized_text=text if normalize else "",
                details={}
            )
            
        primary_dash = max(count_by_name, key=count_by_name.get)
        
        # Check consistency
        # If multiple types of long dashes are used, that's inconsistent
        long_dash_counts = [
            count_by_name['En dash (U+2013)'], 
            count_by_name['Em dash (U+2014)'], 
            double_hyphen_count
        ]
        types_used = sum(1 for c in long_dash_counts if c > 0)
        consistent_usage = types_used <= 1
        
        # Check spacing patterns for em-dashes
        suspicious_spacing = False
        spacing_details = {}
        
        em_dash_char = '—'
        if counts['—'] > 0:
            # Regex to find spacing: (space)(dash)(space), (char)(dash)(char), etc.
            spaced = len(re.findall(r' — ', text))
            unspaced = len(re.findall(r'[^ ]—[^ ]', text))
            left_spaced = len(re.findall(r' —[^ ]', text))
            right_spaced = len(re.findall(r'[^ ]— ', text))
            
            spacing_details = {
                "spaced": spaced,
                "unspaced": unspaced,
                "left_only": left_spaced,
                "right_only": right_spaced
            }
            
            # Inconsistency: Mixing spaced and unspaced
            if spaced > 0 and unspaced > 0:
                suspicious_spacing = True
        
        # ===== AI OVERUSE DETECTION =====
        text_length = len(text)
        em_dash_count = counts['—'] + double_hyphen_count  # Include -- as em-dash substitute
        
        # Calculate em-dash ratio per 1000 characters
        em_dash_ratio = (em_dash_count / text_length) * 1000 if text_length > 0 else 0
        
        # Check for overuse (AI signature)
        # Lower threshold - any significant em-dash usage is noteworthy
        em_dash_overuse = em_dash_ratio > 1.0  # More than 1.0 per 1000 chars is suspicious
        
        # Calculate AI dash signature score
        # Score based on em-dash ratio relative to expected human/AI ratios
        if em_dash_ratio <= self.HUMAN_EM_DASH_RATIO:
            ai_dash_signature = 0.0
        elif em_dash_ratio >= self.AI_EM_DASH_RATIO:
            ai_dash_signature = 1.0
        else:
            # Linear interpolation
            ai_dash_signature = (em_dash_ratio - self.HUMAN_EM_DASH_RATIO) / (self.AI_EM_DASH_RATIO - self.HUMAN_EM_DASH_RATIO)
        
        # Boost signature if we also see inconsistent usage (less common in AI)
        if not consistent_usage:
            ai_dash_signature = max(ai_dash_signature - 0.1, 0.0)
        
        # Normalize text if requested
        normalized_text = ""
        if normalize:
            normalized_text = text
            for char in ['‒', '–', '—', '―']:  # Figure, en, em, bar -> hyphen-minus
                normalized_text = normalized_text.replace(char, '-')
            normalized_text = normalized_text.replace('--', '-')
        
        return DashResult(
            dash_types_found=found_types,
            dash_counts=count_by_name,
            consistent_usage=consistent_usage,
            primary_dash=primary_dash,
            suspicious_spacing=suspicious_spacing,
            em_dash_overuse=em_dash_overuse,
            em_dash_ratio=round(em_dash_ratio, 3),
            ai_dash_signature=round(ai_dash_signature, 3),
            normalized_text=normalized_text,
            details={
                "spacing": spacing_details,
                "total_dashes": total_dashes,
                "text_length": text_length,
            }
        )


# ============================================================================
# Repetition Analyzer (NEW v2.0)
# ============================================================================

class RepetitionAnalyzer:
    """
    Analyzes text for repetitive patterns characteristic of AI.
    
    AI-generated text often shows:
    - Repeated phrases within and across paragraphs
    - Similar sentence structures
    - Repeated transition words
    - Uniform paragraph openings
    
    Human text tends to be more varied.
    """
    
    # Common AI transition words that get overused
    AI_TRANSITIONS = [
        "furthermore", "moreover", "additionally", "however", "therefore",
        "consequently", "in conclusion", "to summarize", "overall",
        "firstly", "secondly", "lastly", "finally", "in summary",
        "it is important to note", "it's worth mentioning", "notably",
        "specifically", "particularly", "essentially", "fundamentally",
    ]
    
    def __init__(self):
        self._sentence_pattern = re.compile(r'[.!?]+')
        self._word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        
    def analyze(self, text: str, ngram_size: int = 4) -> RepetitionResult:
        """
        Analyze text for repetition patterns.
        
        Args:
            text: Input text
            ngram_size: Size of n-grams to check for repetition
        """
        if not text or len(text) < 100:
            return RepetitionResult(
                repetition_score=0.0,
                repeated_phrases=[],
                sentence_similarity_avg=0.0,
                paragraph_similarity_avg=0.0,
                ai_repetition_signature=0.0,
                details={}
            )
        
        text_lower = text.lower()
        words = self._word_pattern.findall(text_lower)
        
        if len(words) < 20:
            return RepetitionResult(
                repetition_score=0.0,
                repeated_phrases=[],
                sentence_similarity_avg=0.0,
                paragraph_similarity_avg=0.0,
                ai_repetition_signature=0.0,
                details={}
            )
        
        # Find repeated n-grams
        ngrams = []
        for i in range(len(words) - ngram_size + 1):
            ngram = ' '.join(words[i:i+ngram_size])
            ngrams.append(ngram)
        
        ngram_counts = collections.Counter(ngrams)
        repeated_phrases = [(phrase, count) for phrase, count in ngram_counts.most_common(10) if count >= 2]
        
        # Calculate repetition score based on repeated n-grams
        total_ngrams = len(ngrams)
        repeated_ngram_instances = sum(count - 1 for _, count in repeated_phrases)  # Excess occurrences
        repetition_score = repeated_ngram_instances / total_ngrams if total_ngrams > 0 else 0
        
        # Check for AI transition overuse
        transition_count = 0
        for transition in self.AI_TRANSITIONS:
            transition_count += text_lower.count(transition)
        
        transition_density = transition_count / (len(words) / 100) if words else 0  # Per 100 words
        
        # Sentence similarity (simplified - based on length variance)
        sentences = [s.strip() for s in self._sentence_pattern.split(text) if s.strip()]
        sentence_similarity_avg = 0.0
        if len(sentences) >= 3:
            lengths = [len(s.split()) for s in sentences]
            mean_len = sum(lengths) / len(lengths)
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            # Higher variance = more varied = more human-like
            # Invert to get similarity (low variance = high similarity = AI-like)
            if mean_len > 0:
                cv = (variance ** 0.5) / mean_len  # Coefficient of variation
                sentence_similarity_avg = max(0, 1 - cv)  # Higher = more similar = more AI
        
        # Paragraph similarity (check opening words)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_similarity_avg = 0.0
        if len(paragraphs) >= 2:
            opening_words = []
            for p in paragraphs:
                words_in_p = self._word_pattern.findall(p.lower())
                if len(words_in_p) >= 3:
                    opening_words.append(' '.join(words_in_p[:3]))
            
            if opening_words:
                unique_openings = len(set(opening_words))
                paragraph_similarity_avg = 1 - (unique_openings / len(opening_words))
        
        # Calculate combined AI repetition signature
        ai_repetition_signature = (
            0.3 * min(repetition_score * 5, 1.0) +  # Repeated phrases
            0.3 * sentence_similarity_avg +          # Similar sentences
            0.2 * paragraph_similarity_avg +         # Similar paragraphs
            0.2 * min(transition_density / 3, 1.0)   # Transition overuse
        )
        
        return RepetitionResult(
            repetition_score=round(repetition_score, 4),
            repeated_phrases=repeated_phrases[:5],  # Top 5
            sentence_similarity_avg=round(sentence_similarity_avg, 4),
            paragraph_similarity_avg=round(paragraph_similarity_avg, 4),
            ai_repetition_signature=round(ai_repetition_signature, 4),
            details={
                "transition_count": transition_count,
                "transition_density": round(transition_density, 2),
                "total_ngrams": total_ngrams,
                "repeated_ngrams_count": len(repeated_phrases),
            }
        )


# ============================================================================
# Web Search Provenance (Enhanced v2.0)
# ============================================================================

class WebSearchProvenance:
    """
    Checks text against web sources using Google Custom Search API.
    Chunks text into sentences/segments and queries for exact matches.
    
    v2.0 Enhancements:
    - Better chunking strategy
    - Plagiarism scoring
    - Human copy-paste detection (exact web matches suggest human, not AI)
    
    Setup:
    1. Create .env file with:
       GOOGLE_API_KEY=your_api_key
       GOOGLE_CSE_ID=your_custom_search_engine_id
    
    2. Create Custom Search Engine at https://cse.google.com/
    """
    
    def __init__(self, api_key: str = None, cse_id: str = None):
        """
        Initialize web search provenance checker.
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            cse_id: Custom Search Engine ID (or set GOOGLE_CSE_ID env var)
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.cse_id = cse_id or os.environ.get("GOOGLE_CSE_ID", "")
        self.enabled = bool(self.api_key and self.cse_id)
        
        if not self.enabled:
            print("WebSearchProvenance: DISABLED (Missing GOOGLE_API_KEY or GOOGLE_CSE_ID)")
        else:
            print("WebSearchProvenance: ENABLED")
        
    def _chunk_text(self, text: str, strategy: str = "sentences") -> List[str]:
        """
        Split text into chunks for searching.
        
        Args:
            text: Input text
            strategy: "sentences" (3 sentences) or "phrases" (unique phrases)
        """
        if strategy == "sentences":
            # Split by sentence-ending punctuation
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            
            # Create chunks of 1-2 sentences (better for exact matching)
            chunks = []
            for i in range(0, len(sentences), 2):
                chunk = " ".join(sentences[i:i+2])
                if 40 <= len(chunk) <= 200:  # Optimal length for search
                    chunks.append(chunk)
            return chunks
        
        elif strategy == "phrases":
            # Extract distinctive phrases (longer noun phrases, unique expressions)
            # Simple approach: take middle portions which are often more unique
            words = text.split()
            if len(words) < 20:
                return [text] if len(text) > 40 else []
            
            # Take a few distinctive chunks
            chunks = []
            chunk_size = min(15, len(words) // 3)
            
            # Beginning, middle, and end chunks
            positions = [
                len(words) // 4,      # ~25% in
                len(words) // 2,      # ~50% in
                3 * len(words) // 4,  # ~75% in
            ]
            
            for pos in positions:
                start = max(0, pos - chunk_size // 2)
                chunk = ' '.join(words[start:start + chunk_size])
                if len(chunk) > 30:
                    chunks.append(chunk)
            
            return chunks
        
        return []

    def search(self, text: str, max_chunks: int = 3, chunk_strategy: str = "sentences") -> ProvenanceResult:
        """
        Search for text provenance.
        
        Args:
            text: Input text
            max_chunks: Max number of chunks to query (to save quota)
            chunk_strategy: "sentences" or "phrases"
            
        Returns:
            ProvenanceResult with match information
        """
        if not self.enabled:
            return ProvenanceResult(
                found_matches=False,
                max_similarity=0.0,
                sources=[],
                exact_quote_matches=0,
                plagiarism_score=0.0,
                unique_sources_count=0,
                is_likely_copied=False,
                details={"status": "disabled"}
            )
        
        if not text or len(text) < 50:
            return ProvenanceResult(
                found_matches=False,
                max_similarity=0.0,
                sources=[],
                exact_quote_matches=0,
                plagiarism_score=0.0,
                unique_sources_count=0,
                is_likely_copied=False,
                details={"status": "text_too_short"}
            )
        
        if not REQUESTS_AVAILABLE:
            return ProvenanceResult(
                found_matches=False,
                max_similarity=0.0,
                sources=[],
                exact_quote_matches=0,
                plagiarism_score=0.0,
                unique_sources_count=0,
                is_likely_copied=False,
                details={"status": "requests_not_available"}
            )
        
        chunks = self._chunk_text(text, chunk_strategy)
        if not chunks:
            return ProvenanceResult(
                found_matches=False,
                max_similarity=0.0,
                sources=[],
                exact_quote_matches=0,
                plagiarism_score=0.0,
                unique_sources_count=0,
                is_likely_copied=False,
                details={"status": "no_valid_chunks"}
            )
        
        # Select chunks to search (prioritize middle/end as intros are generic)
        if len(chunks) > max_chunks:
            # Take from middle and end
            selected = []
            if len(chunks) >= 2:
                selected.append(chunks[len(chunks) // 2])  # Middle
            selected.extend(chunks[-max_chunks + len(selected):])  # End
            selected_chunks = selected[:max_chunks]
        else:
            selected_chunks = chunks
        
        sources = []
        exact_matches = 0
        errors = []
        
        for chunk in selected_chunks:
            # Create an exact match query (quoted)
            query = f'"{chunk}"'
            
            try:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': self.api_key,
                    'cx': self.cse_id,
                    'q': query,
                    'num': 3  # Top 3 results
                }
                
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                
                if 'error' in data:
                    errors.append(f"API error: {data['error'].get('message', 'Unknown')}")
                    continue
                
                if 'items' in data and len(data['items']) > 0:
                    exact_matches += 1
                    for item in data['items']:
                        sources.append({
                            'title': item.get('title', 'Unknown'),
                            'link': item.get('link', ''),
                            'snippet': item.get('snippet', ''),
                            'chunk_matched': chunk[:50] + "..." if len(chunk) > 50 else chunk
                        })
                
                # Rate limiting
                time.sleep(0.5)
                
            except requests.Timeout:
                errors.append("Request timeout")
            except requests.ConnectionError:
                errors.append("Connection error")
            except Exception as e:
                errors.append(f"Error: {str(e)}")
        
        # Deduplicate sources
        unique_sources = []
        seen_links = set()
        for s in sources:
            link = s.get('link', '')
            if link and link not in seen_links:
                unique_sources.append(s)
                seen_links.add(link)
        
        # Calculate plagiarism score
        chunks_searched = len(selected_chunks)
        if chunks_searched > 0:
            plagiarism_score = exact_matches / chunks_searched
        else:
            plagiarism_score = 0.0
        
        # Determine if likely copied (exact matches suggest human copy-paste, not AI)
        # Key insight: AI generates novel text, humans copy-paste from web
        is_likely_copied = exact_matches >= 2 or (exact_matches >= 1 and plagiarism_score >= 0.5)
            
        return ProvenanceResult(
            found_matches=exact_matches > 0,
            max_similarity=plagiarism_score,
            sources=unique_sources[:5],  # Top 5 unique sources
            exact_quote_matches=exact_matches,
            plagiarism_score=round(plagiarism_score, 3),
            unique_sources_count=len(unique_sources),
            is_likely_copied=is_likely_copied,
            details={
                "chunks_searched": chunks_searched,
                "total_chunks_available": len(chunks),
                "errors": errors if errors else None,
            }
        )


# ============================================================================
# Provenance Bundle Aggregator
# ============================================================================

class ProvenanceAggregator:
    """
    Aggregates all provenance analyzers and provides combined signals.
    """
    
    def __init__(
        self,
        google_api_key: str = None,
        google_cse_id: str = None,
        feature_flags: Dict[str, bool] = None
    ):
        """
        Initialize all analyzers.
        
        Args:
            google_api_key: Google API key for web search
            google_cse_id: Google Custom Search Engine ID
            feature_flags: Override default feature flags
        """
        self.flags = {**DEFAULT_FEATURE_FLAGS, **(feature_flags or {})}
        
        # Initialize analyzers
        self.unicode_analyzer = UnicodeAnalyzer() if self.flags.get("unicode_analysis") else None
        self.dash_analyzer = DashAnalyzer() if self.flags.get("dash_analysis") else None
        self.repetition_analyzer = RepetitionAnalyzer() if self.flags.get("repetition_analysis") else None
        
        # Web search requires API keys
        if self.flags.get("web_provenance"):
            self.web_search = WebSearchProvenance(google_api_key, google_cse_id)
        else:
            self.web_search = None
    
    def analyze(self, text: str, run_web_search: bool = False) -> ProvenanceBundle:
        """
        Run all enabled provenance analyzers.
        
        Args:
            text: Input text
            run_web_search: Whether to run web search (may cost API quota)
            
        Returns:
            ProvenanceBundle with all results and aggregated signals
        """
        bundle = ProvenanceBundle()
        human_signals = []
        ai_signals = []
        
        # Unicode analysis
        if self.unicode_analyzer:
            bundle.unicode = self.unicode_analyzer.analyze(text)
            
            # Interpret signals
            if bundle.unicode.invisible_chars_count > 0:
                human_signals.append("Contains invisible Unicode characters (copy-paste artifact)")
            if bundle.unicode.smart_quotes_count > 3:
                human_signals.append("Contains smart quotes (word processor/copy-paste)")
            if bundle.unicode.emoji_count > 0:
                human_signals.append("Contains emojis (informal/human)")
        
        # Dash analysis
        if self.dash_analyzer:
            bundle.dash = self.dash_analyzer.analyze(text)
            
            # Interpret signals
            if bundle.dash.em_dash_overuse:
                ai_signals.append(f"Em-dash overuse ({bundle.dash.em_dash_ratio:.1f}/1000 chars) - AI pattern")
            elif bundle.dash.em_dash_ratio > 0.5:  # Significant but not overuse
                ai_signals.append(f"Em-dash usage ({bundle.dash.em_dash_ratio:.1f}/1000 chars) - common in AI")
            if bundle.dash.ai_dash_signature > 0.4:  # Lowered threshold
                ai_signals.append(f"AI-like dash usage pattern (score: {bundle.dash.ai_dash_signature:.2f})")
            if not bundle.dash.consistent_usage:
                human_signals.append("Inconsistent dash usage (human pattern)")
        
        # Repetition analysis
        if self.repetition_analyzer:
            bundle.repetition = self.repetition_analyzer.analyze(text)
            
            # Interpret signals
            if bundle.repetition.ai_repetition_signature > 0.5:
                ai_signals.append(f"High repetition score ({bundle.repetition.ai_repetition_signature:.2f})")
            if bundle.repetition.sentence_similarity_avg > 0.7:
                ai_signals.append("Very uniform sentence structure")
            if len(bundle.repetition.repeated_phrases) > 3:
                ai_signals.append(f"Multiple repeated phrases ({len(bundle.repetition.repeated_phrases)})")
        
        # Web search (only if enabled and requested)
        if self.web_search and self.web_search.enabled and run_web_search:
            bundle.web_search = self.web_search.search(text)
            
            # Interpret signals - Key insight: exact web matches suggest HUMAN copy-paste
            if bundle.web_search.is_likely_copied:
                human_signals.append(f"Found {bundle.web_search.exact_quote_matches} exact web matches - likely copy-paste")
                bundle.suggests_plagiarism = True
            elif bundle.web_search.found_matches:
                human_signals.append("Some web matches found - may contain quotes/references")
        
        # Calculate overall provenance score
        # This score indicates contribution to AI likelihood (0 = human, 1 = AI)
        ai_score = 0.0
        weight_total = 0.0
        
        if bundle.dash and bundle.dash.ai_dash_signature > 0:
            ai_score += bundle.dash.ai_dash_signature * 0.3
            weight_total += 0.3
        
        if bundle.repetition and bundle.repetition.ai_repetition_signature > 0:
            ai_score += bundle.repetition.ai_repetition_signature * 0.4
            weight_total += 0.4
        
        if bundle.unicode:
            # Anomalous unicode slightly reduces AI likelihood (copy-paste artifacts)
            if bundle.unicode.anomaly_score > 0.2:
                ai_score -= 0.1
        
        if bundle.web_search and bundle.web_search.is_likely_copied:
            # Web matches strongly suggest human copy-paste
            ai_score -= 0.3
            bundle.suggests_plagiarism = True
        
        # Normalize
        if weight_total > 0:
            ai_score = ai_score / weight_total
        
        ai_score = max(0.0, min(1.0, ai_score))
        
        bundle.overall_provenance_score = round(ai_score, 3)
        bundle.human_signals = human_signals
        bundle.ai_signals = ai_signals
        bundle.suggests_ai = ai_score > 0.5 and len(ai_signals) > len(human_signals)
        bundle.confidence = abs(ai_score - 0.5) * 2  # 0 at 0.5, 1 at extremes
        
        return bundle


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_provenance(
    text: str,
    run_web_search: bool = False,
    google_api_key: str = None,
    google_cse_id: str = None
) -> ProvenanceBundle:
    """
    Quick provenance analysis using all enabled analyzers.
    
    Args:
        text: Input text
        run_web_search: Whether to run web search
        google_api_key: Optional Google API key
        google_cse_id: Optional Google CSE ID
        
    Returns:
        ProvenanceBundle with all results
    """
    aggregator = ProvenanceAggregator(google_api_key, google_cse_id)
    return aggregator.analyze(text, run_web_search)


def get_provenance_summary(bundle: ProvenanceBundle) -> str:
    """
    Get a human-readable summary of provenance analysis.
    """
    lines = ["📊 *Provenance Analysis*", ""]
    
    # Unicode
    if bundle.unicode:
        if bundle.unicode.has_special_chars:
            lines.append(f"🔣 Special characters found: {bundle.unicode.special_char_ratio*100:.1f}%")
        if bundle.unicode.invisible_chars_count > 0:
            lines.append(f"👻 Invisible chars: {bundle.unicode.invisible_chars_count}")
        if bundle.unicode.smart_quotes_count > 0:
            lines.append(f"📝 Smart quotes: {bundle.unicode.smart_quotes_count}")
        if bundle.unicode.emoji_count > 0:
            lines.append(f"😀 Emojis: {bundle.unicode.emoji_count}")
    
    # Dash
    if bundle.dash:
        lines.append(f"➖ Dash style: {bundle.dash.primary_dash}")
        if bundle.dash.em_dash_overuse:
            lines.append(f"⚠️ Em-dash overuse detected (AI pattern)")
        if not bundle.dash.consistent_usage:
            lines.append(f"📌 Inconsistent dash usage")
    
    # Repetition
    if bundle.repetition:
        if bundle.repetition.ai_repetition_signature > 0.4:
            lines.append(f"🔄 Repetition score: {bundle.repetition.ai_repetition_signature:.2f}")
    
    # Web search
    if bundle.web_search:
        if bundle.web_search.found_matches:
            lines.append(f"🌐 Web matches: {bundle.web_search.exact_quote_matches}")
            if bundle.web_search.is_likely_copied:
                lines.append(f"📋 Likely copy-paste from web")
    
    # Summary
    lines.append("")
    lines.append("─────────────────")
    
    if bundle.human_signals:
        lines.append("👤 Human signals:")
        for sig in bundle.human_signals[:3]:
            lines.append(f"  • {sig}")
    
    if bundle.ai_signals:
        lines.append("🤖 AI signals:")
        for sig in bundle.ai_signals[:3]:
            lines.append(f"  • {sig}")
    
    return "\n".join(lines)
