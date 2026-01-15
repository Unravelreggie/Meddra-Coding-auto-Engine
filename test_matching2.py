import sys
sys.path.insert(0, 'E:/PV_Pf/projects/LLT_engine')

from llt_engine.api import LLTEngine

# Initialize Engine
print("Initializing LLT Engine...")
engine = LLTEngine(
    meddra_path="E:/PV_Pf/projects/quarter_compare_tool/data/meddra_cache.xlsx",
    vector_parquet_path="E:/PV_Pf/projects/quarter_compare_tool/data/llt_emb.parquet",
    embed_config={
        "provider": "gitee",
        "url": "https://ai.gitee.com/v1/embeddings",
        "model": "bge-m3",
        "dimensions": 1024,
        "api_key": "LGTLVPJ8SQOMXAHV4NCFDLB0NZMNYCULQJFEOIWR"
    },
    vector_config={
        "term_col": "LLTTM_CN",
        "emb_col": "embedding"
    }
)

print("\n" + "="*60)
print("Testing with sample terms (lower threshold):")
print("="*60)

# Test terms - these should NOT be in MedDRA
test_terms = [
    "一般反应",  # Should try vector matching
    "其他过敏性反应",  # Should try vector matching
]

# Test with lower threshold
print("\n### Testing with min_sim=0.25 (lower threshold) ###")
for term in test_terms:
    print(f"\nTerm: '{term}'")
    # Call matcher directly with lower threshold
    result = engine.matcher.match(term, min_sim=0.25)
    print(f"  Code: {result.get('code', 'N/A')}")
    print(f"  Source: {result.get('source', 'N/A')}")
    print(f"  Score: {result.get('score', 0):.3f}")
