import sys
sys.path.insert(0, 'E:/PV_Pf/projects/LLT_engine')

from llt_engine.embedding import Embedder

# Test Gitee API
config = {
    "provider": "gitee",
    "url": "https://ai.gitee.com/v1/embeddings",
    "model": "bge-m3",
    "dimensions": 1024,
    "api_key": "LGTLVPJ8SQOMXAHV4NCFDLB0NZMNYCULQJFEOIWR"
}

embedder = Embedder(config)
print("Testing embedding with Gitee API...")
vec = embedder.embed("测试")

if vec is not None:
    print(f"✅ Embedding successful! Shape: {vec.shape}, First 5 values: {vec[:5]}")
else:
    print("❌ Embedding failed!")
