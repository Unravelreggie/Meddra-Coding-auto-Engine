import numpy as np
from typing import Dict, Optional, Callable, Any

class LLTMatcher:
    def __init__(self, exact_map: Dict[str, str], vector_terms: list, vector_matrix: np.ndarray, embed_func: Callable[[str], np.ndarray]):
        self.exact_map = exact_map
        self.vector_terms = vector_terms
        self.vector_matrix = vector_matrix
        self.embed_func = embed_func
        
    def match(self, term: str, min_sim: float = 0.25) -> Dict[str, Any]:
        """
        Match a single term.
        Returns:
        {
            "term": str,
            "code": str (LLT Result),
            "source": str ("28.1", "Vector", "None"),
            "score": float
        }
        """
        if not term:
            return {"term": "", "code": "", "source": "Empty", "score": 0.0}
            
        # 1. Exact Match (MedDRA 28.1)
        if term in self.exact_map:
            return {
                "term": term, 
                "code": self.exact_map[term], 
                "source": "28.1", 
                "score": 1.0
            }
            
        # 2. Vector Search (if matrix available)
        if self.vector_matrix is not None and len(self.vector_terms) > 0 and self.embed_func:
            print(f"[Matcher] Attempting vector match for: '{term}'")
            query_vec = self.embed_func(term)
            if query_vec is not None:
                print(f"[Matcher] Got embedding vector, shape: {query_vec.shape}")
                sims = self.vector_matrix @ query_vec
                if sims.size > 0:
                    idx = np.argmax(sims)
                    best_score = float(sims[idx])
                    best_term = self.vector_terms[idx]
                    print(f"[Matcher] Best match: '{best_term}' with score {best_score:.3f} (min_sim={min_sim})")
                    
                    if best_score >= min_sim:
                        return {
                            "term": term, 
                            "code": best_term, # Vector store usually returns the standard term
                            "source": "Vector", 
                            "score": best_score
                        }
                    else:
                        print(f"[Matcher] Score {best_score:.3f} < min_sim {min_sim}, no match")
            else:
                print(f"[Matcher] Failed to get embedding for: '{term}'")
        else:
            print(f"[Matcher] Vector search not available (matrix={self.vector_matrix is not None}, terms={len(self.vector_terms) if self.vector_terms else 0}, embed_func={self.embed_func is not None})")
        
        # 3. No match
        return {"term": term, "code": "", "source": "None", "score": 0.0}
