import re

def clean_term_suffix(term: str) -> str:
    """
    Clean term by taking the suffix after the last hyphen if present.
    Example: "Allergy-Rash" -> "Rash"
    Also strips whitespace.
    """
    if not term:
        return ""
    
    # User requirement: If there is a hyphen, use the value AFTER it.
    # We assume 'last hyphen' if multiple present, to be safe for "A-B-C" -> "C"
    # Logic: Split by hyphen, take last element.
    
    if '-' in term:
        parts = term.split('-')
        # Take the last part that is not empty? 
        # Or just the strictly last part? User said "take the value after".
        # Let's take the last non-empty part to be robust against "A-B-"
        
        valid_parts = [p.strip() for p in parts if p.strip()]
        if valid_parts:
            return valid_parts[-1]
        else:
            return "" # "-" or " - " results in empty
            
    # Also handle Chinese hyphen if needed? User example used ascii hyphen.
    # But usually data has diverse hyphens.
    # Let's stick to strict ascii '-' as mostly requested, or standardized first.
    # The user instruction specifically said "-". 
    
    return term.strip()

def normalize_text(text: str) -> str:
    """
    Basic normalization: lower case, strip.
    """
    if not text: return ""
    return str(text).strip().lower()
