import re
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass(frozen=True)
class Delegate:
    role: str
    args: Dict[str, Any]

def extract_numeric_value(text: str) -> str:
    """
    extracts the last valid number following a 'final:' marker.
    useful for self-correction scenarios.
    """
    if not text: 
        return ""
    
    # split on final keyword
    parts = re.split(r'(?i)\bfinal\s*:?', text)
    if len(parts) < 2: 
        return ""

    candidate = parts[-1].strip()
    # find number pattern
    match = re.search(r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?', candidate)
    
    if match:
        val = match.group(0)
        # ensure it contains digits
        if not any(char.isdigit() for char in val): 
            return ""
        # strip trailing decimal
        if val.endswith('.'): 
            val = val[:-1]
        return val
    return ""

def parse_final_marker(text: str) -> Optional[str]:
    """
    checks if text starts with final marker.
    used for flow control.
    """
    # multiline start anchor
    pattern = re.compile(r'(?m)^\s*FINAL\s*:?\s*')
    if not text: 
        return None
        
    m = pattern.search(text)
    if not m: 
        return None
        
    rest = text[m.end():].strip()
    return text.strip() if rest else None

def parse_delegate(text: str) -> Optional[Delegate]:
    # check for call marker
    pattern = re.compile(r'(?m)^\s*CALL\s*:?\s*')
    if not text: 
        return None
        
    m = pattern.search(text)
    if not m: 
        return None
        
    rest = text[m.end():].strip()
    if not rest: 
        return None
        
    parts = rest.split(None, 1)
    role = parts[0].strip()
    args: Dict[str, Any] = {}
    
    # parse arguments
    if len(parts) > 1:
        raw = parts[1].strip()
        try:
            parsed = json.loads(raw)
            args = parsed if isinstance(parsed, dict) else {"_": parsed}
        except Exception:
            args = {"raw": raw}
            
    return Delegate(role=role, args=args)

def parse_all_delegates(text: str) -> List[Delegate]:
    calls = []
    for line in text.splitlines():
        s = line.strip()
        # stop if final reached
        if s.upper().startswith("FINAL"): 
            break
        if s.startswith("CALL:"):
            d = parse_delegate(s)
            if d: 
                calls.append(d)
    return calls
