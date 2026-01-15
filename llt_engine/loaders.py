import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, List

def load_meddra_28_1_map(path_xlsx: str, col_key: str = None, col_val: str = None) -> Dict[str, str]:
    """
    Load exact match mapping from Excel.
    If col_key/col_val provided, use them. Else use heuristics (A col -> 28.1 col).
    """
    if not os.path.exists(path_xlsx):
        print(f"Warning: MedDRA file not found at {path_xlsx}")
        return {}

    try:
        # User might have renamed sheets or file structure
        try:
            df = pd.read_excel(path_xlsx, sheet_name="层级结构分析", engine="openpyxl")
        except ValueError:
            # Fallback to first sheet
            print("Sheet '层级结构分析' not found. Loading first sheet.")
            df = pd.read_excel(path_xlsx, sheet_name=0, engine="openpyxl")
    except Exception as e:
        print(f"Error loading MedDRA sheet: {e}")
        return {}

    if df.shape[1] < 1: return {}
    
    cols = [str(c).strip() for c in df.columns]
    
    # Check provided columns
    target_key = cols[0]
    target_val = cols[0]
    
    if col_key and col_key in cols:
        target_key = col_key
    else:
        # Default to Column A
        target_key = cols[0]
        
    if col_val and col_val in cols:
        target_val = col_val
    else:
        # Heuristic
        for c in cols:
            if "28.1" in c.replace(" ", "") or "LLT" in c:
                target_val = c
                break
                
    print(f"[Loader] Using MedDRA Map: Key='{target_key}' -> Val='{target_val}'")
    
    mapping = {}
    for _, row in df.iterrows():
        k = str(row.get(target_key, "")).strip()
        v = str(row.get(target_val, "")).strip()
        if k and v:
            mapping[k] = v
            
    return mapping
            
    mapping = {}
    for _, row in df.iterrows():
        # Simple normalization
        k = str(row.get(col_term, "")).strip()
        v = str(row.get(col_out, "")).strip()
        if k and v:
            mapping[k] = v
            
    return mapping

def load_vector_store(parquet_path: str, col_term: str = 'term', col_emb: str = 'embedding') -> Tuple[List[str], np.ndarray]:
    """
    Load embeddings from parquet.
    Returns (terms_list, normalized_matrix).
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Vector store not found: {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    if col_term not in df.columns or col_emb not in df.columns:
        raise ValueError(f"Missing columns {col_term}/{col_emb} in {parquet_path}")
        
    terms = df[col_term].astype(str).str.strip().tolist()
    emb_list = df[col_emb].tolist()
    
    mat = np.asarray(emb_list, dtype=np.float32)
    
    # Normalize
    norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat_n = mat / norm
    
    return terms, mat_n
