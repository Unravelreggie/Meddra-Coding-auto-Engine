import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import yaml

# Ensure llt_engine package is in path (current dir is root of project typically)
# If running as 'python -m llt_engine.server', this works relative.
# But adding explicit path to be safe if run from inside.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from llt_engine.api import LLTEngine

app = FastAPI(title="LLT Engine API")

# Global Engine Instance
ENGINE: Optional[LLTEngine] = None

# Configuration (Hardcoded for now as per user context, or Env Vars)
# User's paths:
DATA_DIR_QTOOL = r"E:\PV_Pf\projects\quarter_compare_tool\data"
MEDDRA_PATH = os.path.join(DATA_DIR_QTOOL, "meddra_cache.xlsx")
VECTOR_PATH = os.path.join(DATA_DIR_QTOOL, "llt_emb.parquet")

# Vector Config matches user's parquet
VECTOR_CONFIG = {
    "term_col": "LLTTM_CN",
    "emb_col": "embedding"
}

# =============================================================================
# Embedding Configuration Loader
# =============================================================================
def load_embed_config_from_settings() -> dict:
    """
    Load embedding configuration from quarter_compare_tool/configs/settings.yaml
    Falls back to hardcoded defaults if settings.yaml not found or invalid.
    """
    # Default fallback config
    default_config = {
        "provider": "local",
        "url": "http://127.0.0.1:9501/embed",
        "model": "bge-m3",
        "dimensions": 1024,
        "api_key": ""
    }
    
    try:
        # Path to settings.yaml in quarter_compare_tool
        settings_path = os.path.join(DATA_DIR_QTOOL, "..", "configs", "settings.yaml")
        settings_path = os.path.abspath(settings_path)
        
        if not os.path.exists(settings_path):
            print(f"[Server] Settings file not found: {settings_path}, using defaults")
            return default_config
            
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = yaml.safe_load(f)
            
        if not settings or 'vector_db' not in settings:
            print(f"[Server] No vector_db config in settings.yaml, using defaults")
            return default_config
            
        vdb = settings['vector_db']
        config = {
            "provider": vdb.get('provider', 'local'),
            "url": vdb.get('url', default_config['url']),
            "model": vdb.get('model', default_config['model']),
            "dimensions": int(vdb.get('dimensions', default_config['dimensions'])),
            "api_key": vdb.get('api_key', '')
        }
        
        # If using gitee but no API key in settings, use default from llt_config.py
        if config['provider'] == 'gitee' and not config['api_key']:
            config['api_key'] = "LGTLVPJ8SQOMXAHV4NCFDLB0NZMNYCULQJFEOIWR"
            print(f"[Server] Using default Gitee API key")
            
        print(f"[Server] Loaded embedding config from settings.yaml: provider={config['provider']}, url={config['url']}")
        return config
        
    except Exception as e:
        print(f"[Server] Error loading settings.yaml: {e}, using defaults")
        return default_config

# Load embedding config dynamically
EMBED_CONFIG = load_embed_config_from_settings()

class MatchRequest(BaseModel):
    terms: List[str]

@app.on_event("startup")
def startup_event():
    global ENGINE
    print("[Server] Starting up LLT Engine...")
    
    if not os.path.exists(MEDDRA_PATH):
        print(f"[Server] WARNING: MedDRA path not found: {MEDDRA_PATH}")
    
    if not os.path.exists(VECTOR_PATH):
        print(f"[Server] WARNING: Vector path not found: {VECTOR_PATH}")

    try:
        # Defaults for MedDRA Config (Project 1: Sheet 0, Col A -> 28.1)
        # Future Project: Can set meddra_config={"key_col": "A", "val_col": "E"}
        MEDDRA_CONFIG = {
            "key_col": None, # Use Heuristic/Default
            "val_col": None  # Use Heuristic/Default
        }
        
        # Log embedding configuration
        api_key_display = "***" if EMBED_CONFIG.get('api_key') else "(none)"
        print(f"[Server] Embedding Config: provider={EMBED_CONFIG['provider']}, url={EMBED_CONFIG['url']}, api_key={api_key_display}")
        
        ENGINE = LLTEngine(
            meddra_path=MEDDRA_PATH,
            vector_parquet_path=VECTOR_PATH,
            embed_config=EMBED_CONFIG,
            vector_config=VECTOR_CONFIG,
            meddra_config=MEDDRA_CONFIG
        )
        print("[Server] LLT Engine Ready.")
    except Exception as e:
        print(f"[Server] FATAL: Failed to initialize Engine: {e}")
        # We don't exit to allow checking health, but matching will fail.

@app.get("/health")
def health_check():
    status = "ready" if ENGINE else "initializing_or_failed"
    return {"status": status, "meddra_path": MEDDRA_PATH, "vector_path": VECTOR_PATH}

@app.post("/match_batch")
def match_batch(req: MatchRequest):
    if not ENGINE:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    results = []
    for term in req.terms:
        try:
            res = ENGINE.encode(term)
            results.append(res)
        except Exception as e:
            # Fallback for individual error
            results.append({
                "term": term, 
                "code": "", 
                "source": "Error", 
                "score": 0.0, 
                "error": str(e)
            })
            
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9005)
