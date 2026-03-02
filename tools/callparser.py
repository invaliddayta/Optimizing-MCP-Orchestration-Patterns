# std
from ast import arg
import re, json
from typing import Optional, Tuple

class CallParser:
    """Parses model output for CALL/FINAL protocol and extracts JSON args"""
    CALL_RE = re.compile(r"(?:^|\s)CALL\s*:\s*([A-Za-z0-9_\-\.]+)\s+(.*)$", re.IGNORECASE | re.DOTALL)
    CALL_JSON_RE = re.compile(r"(?:^|\s)CALL\s*:\s*(\{.*)$", re.IGNORECASE | re.DOTALL)
    FINAL_RE = re.compile(r"(?:^|\s)FINAL\s*:\s*([A-Za-z0-9_\-\.]+)\s+(.*)$", re.IGNORECASE | re.DOTALL)

    @staticmethod
    def strip_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
            t = re.sub(r"\s*```$", "", t)
        return t.strip()

    @staticmethod
    def _extract_balanced_json(s: str) -> Optional[dict]:
        i = s.find("{")
        if i < 0: return None
        depth = 0
        for j, ch in enumerate(s[i:], start=i):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = s[i:j+1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        return None
        return None

    def get_args(self, match) -> Tuple[Optional[str], Optional[dict]]:
        tool_name = match.group(1).strip()
        args_str = self.strip_fences(match.group(2).strip())

        # exact JSON first
        try:
            args = json.loads(args_str)
            if isinstance(args, dict):
                return tool_name, args
        except Exception:
            pass

        # fallback to balanced object
        args = self._extract_balanced_json(args_str)
        if isinstance(args, dict):
            return tool_name, args

        return tool_name, {}


    def parse_call(self, raw: str) -> Tuple[Optional[str] ,Optional[str], Optional[dict]]:
        # remove fences
        text = self.strip_fences(raw)
        
        call = self.CALL_RE.match(text)
        final = self.FINAL_RE.match(text)

        call_json = self.CALL_JSON_RE.match(text)

        if not call and not final and not call_json:
            for line in text.splitlines():
                s = line.strip()
                call = self.CALL_RE.match(s)
                if call: break
                
                final = self.FINAL_RE.match(s)
                if final: break
                
                # Check line for nameless call
                call_json = self.CALL_JSON_RE.match(s)
                if call_json: break
            
            if not final and not call and not call_json: 
                return (None, None, {})

        if final:
            tool_name, args = self.get_args(final) 
            return ("final", tool_name, args)
        elif call:
            tool_name, args = self.get_args(call) 
            return ("call", tool_name, args)
        elif call_json:
            args_str = call_json.group(1)
            args = self._extract_balanced_json(args_str) or {}
            return ("call", None, args)
            
        return (None, None, {})
