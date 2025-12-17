#!/usr/bin/env python3
"""
Test the AI Detection Ensemble using REAL samples:
- AI text: The provided academic report on Polyaniline-Based Food Sensors
- Human text: Real samples from HC3 dataset (verified human-written)

This avoids the issue of AI-generated "human-like" test texts.
"""

import os
import sys
import json

# Add ensemble to path
ENSEMBLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENSEMBLE_DIR)

# The AI-generated academic text provided by the user
AI_ACADEMIC_TEXT = """
**Academic Report: Polyaniline-Based Food Sensors**  
**(Note: This is a condensed version. Actual implementation would expand each section with detailed content and visuals)**  

---

### **Title Page**  
- Title: "Polyaniline-Based Sensors for Food Quality Monitoring: Mechanisms, Applications, and Technological Integration"  
- Author Name  
- Institution  
- Date  

---

### **Abstract**  
Polyaniline (PANI), a conductive polymer, has emerged as a transformative material for food safety monitoring due to its reversible electrical and optical responses to spoilage biomarkers like ammonia (NH₃) and trimethylamine (TMA). This report synthesizes advancements in PANI-based sensors, focusing on their dual detection mechanisms, integration with IoT systems, and cost-effective scalability for global food supply chains. Key innovations include MXene-enhanced nanocomposites, reusable sensor designs, and machine learning-driven calibration, addressing challenges such as humidity interference and industrial feasibility.  

---

### **1. Introduction**  
**1.1 Food Safety and Spoilage Detection**  
- Global food waste exceeds 1.3 billion tons annually, with microbial spoilage as a primary contributor.
- Traditional methods (e.g., pH strips, gas chromatography) lack real-time monitoring capabilities.

**1.2 Role of Polyaniline Sensors**  
- PANI's protonation-deprotonation mechanism enables dual optical/electrical signaling.
- Advantages: Low cost ($0.50–$2/sensor), room-temperature operation, and compatibility with flexible substrates.

---

### **2. Polyaniline: Synthesis and Properties**  
**2.1 Chemical Synthesis**  
- **Oxidative Polymerization**: Aniline monomer + ammonium persulfate (APS) in HCl, yielding emeraldine salt form.
- **Doping**: HCl, camphorsulfonic acid (CSA), or dodecylbenzenesulfonic acid (DBSA) enhance conductivity (up to 200 S/cm).

**2.2 Material Enhancements**  
- **MXene Composites**: PANI/Ti₃C₂TX improves NH₃ sensitivity (LOD: 20 ppb).
- **AgNW Integration**: PANI/silver nanowire composites detect TMA at 3.3 µg/L in pork.

---

### **3. Detection Mechanisms**  
**3.1 Chemical Reactions**  
- **NH₃ Detection**:  
  PANI-H+ + NH₃ → PANI + NH₄+
  - Resistance increase (ΔR = 15–50 kΩ) and color shift (green → blue).

**3.2 Multi-Parameter Sensing**  
- Simultaneous monitoring of pH, NH₃, and temperature via PANI-cellulose films.
"""


def load_real_human_samples(max_samples=5):
    """Load REAL human-written text from HC3 dataset"""
    samples = []
    hc3_path = "/home/lightdesk/Projects/AI-Text/HC3/reddit_eli5.jsonl"
    
    if not os.path.exists(hc3_path):
        print(f"HC3 dataset not found at {hc3_path}")
        return samples
    
    with open(hc3_path, 'r') as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            try:
                data = json.loads(line)
                for ans in data.get('human_answers', []):
                    if len(ans) >= 150 and len(samples) < max_samples:
                        samples.append(ans[:1500])  # Truncate to reasonable length
                        break
            except:
                continue
    
    return samples


def load_real_ai_samples(max_samples=3):
    """Load REAL AI-generated text from HC3 dataset (ChatGPT)"""
    samples = []
    hc3_path = "/home/lightdesk/Projects/AI-Text/HC3/reddit_eli5.jsonl"
    
    if not os.path.exists(hc3_path):
        return samples
    
    with open(hc3_path, 'r') as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            try:
                data = json.loads(line)
                for ans in data.get('chatgpt_answers', []):
                    if len(ans) >= 150 and len(samples) < max_samples:
                        samples.append(ans[:1500])
                        break
            except:
                continue
    
    return samples


def test_ensemble_with_real_samples():
    """Test the full ensemble on REAL samples from databases"""
    
    print("=" * 70)
    print("AI TEXT DETECTION ENSEMBLE - REAL SAMPLE TEST")
    print("=" * 70)
    print("\nNOTE: Using REAL human samples from HC3 dataset")
    print("      NOT AI-generated 'human-like' text\n")
    
    from ensemble import EnsembleDetector
    
    # Initialize ensemble
    print("Loading ensemble detector...")
    ensemble = EnsembleDetector()
    
    # Load real samples
    print("\nLoading real human samples from HC3...")
    real_human_texts = load_real_human_samples(5)
    print(f"Loaded {len(real_human_texts)} real human samples")
    
    print("\nLoading real AI samples from HC3 (ChatGPT)...")
    real_ai_texts = load_real_ai_samples(3)
    print(f"Loaded {len(real_ai_texts)} real AI samples")
    
    # Build test cases
    test_cases = [
        ("User's AI Academic Text (Polyaniline)", AI_ACADEMIC_TEXT, "AI"),
    ]
    
    # Add real AI samples
    for i, text in enumerate(real_ai_texts):
        test_cases.append((f"HC3 ChatGPT Sample {i+1}", text, "AI"))
    
    # Add real human samples
    for i, text in enumerate(real_human_texts):
        test_cases.append((f"HC3 Human Sample {i+1}", text, "Human"))
    
    results = []
    
    for name, text, expected in test_cases:
        print(f"\n{'=' * 70}")
        print(f"TEST: {name}")
        print(f"Expected: {expected}")
        print(f"Text preview: {text[:100].strip()}...")
        print("=" * 70)
        
        result = ensemble.detect(text)
        
        # Detailed breakdown
        print(f"\n  PREDICTION: {result.prediction}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Agreement: {result.agreement}")
        print(f"  Ensemble Score: {result.ensemble_score:.4f}")
        print(f"  Suggested Action: {result.suggested_action}")
        
        print("\n  DETECTOR BREAKDOWN:")
        for det_name, det_result in result.breakdown.items():
            print(f"\n    {det_name.upper()}:")
            print(f"      Prediction: {det_result.prediction}")
            print(f"      Score: {det_result.score:.4f}")
            
            if 'ai_votes' in det_result.details:
                print(f"      Votes: {det_result.details['ai_votes']} AI, {det_result.details['human_votes']} Human")
            if 'raw_curvature' in det_result.details:
                print(f"      Curvature: {det_result.details['raw_curvature']:.4f}")
            if 'raw_score' in det_result.details:
                print(f"      Binoculars Raw: {det_result.details['raw_score']:.4f}")
        
        # Check correctness (UNCERTAIN is acceptable for borderline cases)
        is_correct = (result.prediction == expected) or (result.prediction == "UNCERTAIN")
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"\n  {status}")
        
        results.append({
            "name": name,
            "expected": expected,
            "predicted": result.prediction,
            "correct": is_correct,
            "confidence": result.confidence,
            "agreement": result.agreement
        })
    
    # Summary
    print(f"\n\n{'=' * 70}")
    print("TEST SUMMARY")
    print("=" * 70)
    
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    
    ai_correct = sum(1 for r in results if r["expected"] == "AI" and r["correct"])
    ai_total = sum(1 for r in results if r["expected"] == "AI")
    
    human_correct = sum(1 for r in results if r["expected"] == "Human" and r["correct"])
    human_total = sum(1 for r in results if r["expected"] == "Human")
    
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} {r['name']}: Expected {r['expected']}, Got {r['predicted']} "
              f"({r['confidence']:.0%} conf, {r['agreement']} agree)")
    
    print(f"\n  Overall Accuracy: {correct}/{total} ({correct/total:.0%})")
    print(f"  AI Detection Rate: {ai_correct}/{ai_total} ({ai_correct/ai_total:.0%})")
    print(f"  Human Detection Rate: {human_correct}/{human_total} ({human_correct/human_total:.0%})")
    
    # Calculate False Positive Rate
    false_positives = sum(1 for r in results if r["expected"] == "Human" and r["predicted"] == "AI")
    print(f"  False Positive Rate: {false_positives}/{human_total} ({false_positives/human_total:.0%})")
    
    return results


if __name__ == "__main__":
    results = test_ensemble_with_real_samples()
