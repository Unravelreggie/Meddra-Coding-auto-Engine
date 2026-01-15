from .loaders import load_meddra_28_1_map, load_vector_store
from .cleaning import clean_term_suffix
from .embedding import Embedder
from .matcher import LLTMatcher
import os

class LLTEngine:
    def __init__(self, meddra_path: str, vector_parquet_path: str, embed_config: dict, vector_config: dict = None, meddra_config: dict = None):
        print(f"[LLTEngine] Initializing...")
        print(f"[LLTEngine] Loading MedDRA map from {meddra_path}...")
        
        # Default MedDRA Map Config
        m_key = None
        m_val = None
        if meddra_config:
            m_key = meddra_config.get("key_col")
            m_val = meddra_config.get("val_col")
            
        self.meddra_map = load_meddra_28_1_map(meddra_path, col_key=m_key, col_val=m_val)
        print(f"[LLTEngine] MedDRA exact map size: {len(self.meddra_map)}")
        
        self.vec_terms = []
        self.vec_matrix = None
        
        if vector_parquet_path and os.path.exists(vector_parquet_path):
            print(f"[LLTEngine] Loading Vector Store from {vector_parquet_path}...")
            # Default vector settings
            v_term = "term"
            v_emb = "embedding"
            if vector_config:
                v_term = vector_config.get("term_col", v_term)
                v_emb = vector_config.get("emb_col", v_emb)
                
            try:
                self.vec_terms, self.vec_matrix = load_vector_store(vector_parquet_path, col_term=v_term, col_emb=v_emb)
                print(f"[LLTEngine] Vector Store loaded. Terms: {len(self.vec_terms)}")
            except Exception as e:
                print(f"[LLTEngine] Failed to load vector store: {e}")
        else:
            print(f"[LLTEngine] Vector Store path invalid or missing: {vector_parquet_path}")
            
        self.embedder = Embedder(embed_config)
        
        self.matcher = LLTMatcher(
            exact_map=self.meddra_map,
            vector_terms=self.vec_terms,
            vector_matrix=self.vec_matrix,
            embed_func=self.embedder.embed
        )
        print(f"[LLTEngine] Ready.")
        
    def encode(self, raw_term: str) -> dict:
        """
        Encode a single term.
        Returns dict with keys: raw_term, cleaned_term, code, source, score
        """
        cleaned = clean_term_suffix(raw_term)
        if not cleaned:
            return {
                "raw_term": raw_term,
                "cleaned_term": "",
                "code": "",
                "source": "Empty",
                "score": 0.0,
                "term": "" # Matcher returns 'term' as the cleaned term used for matching
            }
            
        res = self.matcher.match(cleaned)
        res['raw_term'] = raw_term
        res['cleaned_term'] = cleaned
        # Ensure 'code' is the result LLT
        return res
