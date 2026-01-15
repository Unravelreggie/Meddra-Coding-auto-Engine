import json
import urllib.request
import numpy as np
from typing import Optional, Dict, Any

class Embedder:
    def __init__(self, config: Dict[str, Any]):
        """
        config: {
            "provider": "local" | "openai",
            "url": "http://...",
            "api_key": "...",
            "model": "...",
            "dimensions": 1024
        }
        """
        self.config = config

    def embed(self, text: str) -> Optional[np.ndarray]:
        if not text: return None
        
        cfg = self.config
        provider = cfg.get("provider", "local")
        url = cfg.get("url")
        if not url: return None
        
        try:
            if provider == "local":
                payload = {"text": text}
                headers = {"Content-Type": "application/json"}
                data = json.dumps(payload).encode("utf-8")
                
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    
                vec = res_json.get("embedding")
                
            else: # openai compatible
                api_key = cfg.get("api_key", "")
                model = cfg.get("model", "text-embedding-3-small")
                
                payload = {
                    "model": model, 
                    "input": text, 
                    "encoding_format": "float"
                }
                if cfg.get("dimensions"):
                    payload["dimensions"] = int(cfg["dimensions"])
                    
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                data = json.dumps(payload).encode("utf-8")
                
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=20) as resp:
                    res_json = json.loads(resp.read().decode("utf-8"))
                    
                data_list = res_json.get("data", [])
                if not data_list: return None
                vec = data_list[0].get("embedding")
                
            if vec is None: return None
            
            v = np.asarray(vec, dtype=np.float32)
            # Normalize
            norm = np.linalg.norm(v) + 1e-9
            return v / norm
            
        except Exception as e:
            # print(f"Embedding error: {e}")
            return None
