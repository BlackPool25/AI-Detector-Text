from datasets import load_dataset
import pandas as pd

def create_balanced_1m_dataset():
    """
    Create 1M diverse human text samples
    
    NOTE: Some datasets require loading scripts which are deprecated.
    Using streaming mode helps avoid some issues but may still fail.
    Consider using alternative datasets like HuggingFaceFW/fineweb or allenai/c4
    """
    samples = []
    
    # 1. Wikipedia - 500K (factual/encyclopedic)
    print("Loading Wikipedia...")
    # Updated to use wikimedia/wikipedia with newer date
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    wiki_count = 0
    for item in wiki:
        if wiki_count >= 500_000:
            break
        samples.append({
            'text': item['text'],
            'source': 'wikipedia'
        })
        wiki_count += 1
        if wiki_count % 50000 == 0:
            print(f"  Wikipedia: {wiki_count} samples loaded")
    
    # 2. C4 - 300K (web content/news)
    # C4 is a cleaned version of Common Crawl, more reliable than cc_news
    print("Loading C4...")
    c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
    c4_count = 0
    for item in c4:
        if c4_count >= 300_000:
            break
        samples.append({
            'text': item['text'],
            'source': 'c4_web'
        })
        c4_count += 1
        if c4_count % 50000 == 0:
            print(f"  C4: {c4_count} samples loaded")
    
    # 3. OpenWebText - 200K (Reddit-sourced web content)
    print("Loading OpenWebText...")
    # Note: This dataset uses loading scripts which may be deprecated
    # Using Skylion007/openwebtext as fallback
    try:
        owt = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
        owt_count = 0
        for item in owt:
            if owt_count >= 200_000:
                break
            samples.append({
                'text': item['text'],
                'source': 'web'
            })
            owt_count += 1
            if owt_count % 50000 == 0:
                print(f"  OpenWebText: {owt_count} samples loaded")
    except Exception as e:
        print(f"  OpenWebText failed: {e}")
        print("  Skipping OpenWebText, will use more from other sources")
    
    # Fill remaining quota from Wikipedia if we didn't get enough
    remaining = 1_000_000 - len(samples)
    if remaining > 0:
        print(f"Loading {remaining} additional Wikipedia samples to reach 1M...")
        wiki_extra = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        extra_count = 0
        # Skip the ones we already loaded
        for i, item in enumerate(wiki_extra):
            if i < 500_000:
                continue
            if extra_count >= remaining:
                break
            samples.append({
                'text': item['text'],
                'source': 'wikipedia'
            })
            extra_count += 1
    
    # Save to parquet
    df = pd.DataFrame(samples)
    df.to_parquet("human_text_1m_mixed.parquet")
    print(f"\nSaved {len(df)} samples")
    print(f"Source distribution:")
    print(df['source'].value_counts())
    
    return df

# Run it
if __name__ == "__main__":
    dataset = create_balanced_1m_dataset()
