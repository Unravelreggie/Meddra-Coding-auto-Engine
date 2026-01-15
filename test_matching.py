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
print("Testing with sample terms:")
print("="*60)

# Test terms
test_terms = [
    "发热",  # Should match MedDRA
    "一般反应",  # Should try vector matching
    "测试不存在的术语123",  # Should fail both
]

for term in test_terms:
    print(f"\nTerm: '{term}'")
    result = engine.encode(term)
    print(f"  Code: {result.get('code', 'N/A')}")
    print(f"  Source: {result.get('source', 'N/A')}")
    print(f"  Score: {result.get('score', 0):.3f}")
