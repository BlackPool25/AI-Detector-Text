# Dataset Loading Fixes - October 2025

## Issues Found and Fixed

### 1. **Wikipedia Dataset** ❌ → ✅
**Original:** `load_dataset("wikipedia", "20220301.en", split="train")`

**Problem:** 
- Dataset scripts are no longer supported in newer versions of `datasets` library
- Error: `RuntimeError: Dataset scripts are no longer supported`

**Fix:** 
```python
load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
```
- Changed to `wikimedia/wikipedia` (official Wikimedia organization)
- Updated to newer dump: `20231101.en` (6.4M articles)
- Added `streaming=True` for memory efficiency

---

### 2. **CC-News Dataset** ❌
**Original:** `load_dataset("cc_news", split="train")`

**Problem:**
- Dataset no longer exists at this path (404 error)
- Has been removed or relocated

**Fix:** Replaced with **C4 dataset**
```python
load_dataset("allenai/c4", "en", split="train", streaming=True)
```
- C4 is a cleaned Common Crawl dataset
- More reliable and actively maintained
- Contains web content including news

---

### 3. **OpenWebText Dataset** ⚠️ → ✅
**Original:** `load_dataset("openwebtext", split="train")`

**Problem:**
- Dataset uses loading scripts (deprecated)
- May fail in newer versions

**Fix:**
```python
load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
```
- Added full path: `Skylion007/openwebtext`
- Added `trust_remote_code=True` for script-based datasets
- Wrapped in try/except for graceful fallback

---

### 4. **The Pile Dataset (Books3)** ⚠️
**Original:** `load_dataset("EleutherAI/pile", split="train")`

**Problem:**
- Dataset uses loading scripts (deprecated)
- Books3 subset has copyright issues and may be unavailable
- Very large dataset (825 GB) - not practical to stream

**Fix:** **REMOVED** from script
- Books3 has been involved in copyright litigation
- Replaced quota with additional Wikipedia samples
- More reliable and legally safe

---

## Updated Dataset Mix (1M samples)

| Dataset | Count | Source Type | Notes |
|---------|-------|-------------|-------|
| Wikipedia | 500K | Encyclopedic/Factual | Increased from 400K |
| C4 | 300K | Web/News | Replaced cc_news |
| OpenWebText | 200K | Reddit-linked web | With fallback |
| **Total** | **1M** | Mixed | Diverse sources |

---

## Additional Improvements

1. **Progress Tracking**: Added progress prints every 50K samples
2. **Error Handling**: Wrapped OpenWebText in try/except
3. **Fallback Logic**: If any source fails, fill from Wikipedia
4. **Statistics**: Print source distribution at end
5. **Main Guard**: Added `if __name__ == "__main__"` for proper script execution

---

## Testing Recommendations

Before running the full script:

```python
# Test each dataset individually
from datasets import load_dataset

# Test 1: Wikipedia
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
print(next(iter(wiki)))

# Test 2: C4
c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
print(next(iter(c4)))

# Test 3: OpenWebText
owt = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)
print(next(iter(owt)))
```

---

## Alternative Datasets (If Issues Persist)

If any dataset still fails, consider these alternatives:

### For News/Web Content:
- `HuggingFaceFW/fineweb` - High-quality web data
- `HuggingFaceFW/fineweb-edu` - Educational web content
- `mc4` - Multilingual C4

### For Books/Long-form:
- `bookcorpus` - If available
- Additional Wikipedia for encyclopedic long-form

### For General Web:
- `oscar-corpus/OSCAR-2301` - Multilingual web crawl
- `bigcode/the-stack` - Code (if relevant)

---

## Estimated Runtime & Resources

- **Time**: 2-4 hours (depending on network speed)
- **Memory**: ~4-8 GB RAM (streaming mode)
- **Disk**: ~5-10 GB for final parquet file
- **Network**: ~15-20 GB download

---

## Known Limitations

1. **Streaming Mode**: Cannot shuffle data during download
2. **No Books**: Copyright concerns removed Books3
3. **Dataset Scripts**: Some may require `trust_remote_code=True`
4. **Rate Limiting**: HuggingFace may throttle large downloads

---

## Last Updated
October 19, 2025
