# std
import re, json
from typing import List, Optional, Any, Dict
from dataclasses import dataclass 
# external
from mcp import types

class ToolResolver:
    """Resolves tool names and extracts payloads from MCP results."""
    @staticmethod
    def normalize(s: str) -> str:
        return re.sub(r"[\W_]+", "", (s or "")).lower()

    def resolve(self, requested: str, tools: List[types.Tool]) -> Optional[str]:
        """Tries to find requested tool in available tools"""
        if not tools: return None
        rq = self.normalize(requested)
        by_norm = {self.normalize(getattr(t, "name", "")): getattr(t, "name", "") for t in tools if getattr(t, "name", None)}
        
        # 1. Exact match (normalized)
        if rq in by_norm: return by_norm[rq]
        
        # 2. Substring match
        for nkey, original in by_norm.items():
            if rq in nkey or nkey in rq:
                return original
                
        # 3. Fallback: If only one tool exists, assume that's the one (risky but useful for specialists)
        if len(tools) == 1:
            return getattr(tools[0], "name", None)
            
        return None

    def extract_payload(self, call_result: Any) -> Any:
        """
        Tries to find structuredContent in llm call. 
        If parsing fails, returns the error string so the Agent knows it failed.
        """
        # 1. Check for native MCP structured content
        sc = getattr(call_result, "structuredContent", None) or getattr(call_result, "structured_content", None)
        if sc is not None: return sc

        # 2. Check for text content and parse it
        contents = getattr(call_result, "content", []) or []
        texts = [c.text for c in contents if isinstance(c, types.TextContent)]
        
        if texts:
            txt = texts[0].strip()
            try: 
                return json.loads(txt)
            except json.JSONDecodeError as e:
                # Return the error message to the agent
                return f"Error: Tool Output was not valid JSON. {e}. Raw output: {txt[:50]}..."
            except Exception as e:
                return f"Error: Parsing failed. {e}"

        # 3. Last ditch effort: stringify the whole result and parse
        try: 
            return json.loads(str(call_result))
        except Exception: 
            return str(call_result)

# ----------------------------- Tool Inventory ------------------------------

@dataclass
class ToolCard:
    name: str
    description: str
    json_schema: Dict[str, Any]

class ToolInventory:
    """Builds a JSON inventory the LLM can read as context."""
    @staticmethod
    def from_mcp(tools_list: List[types.Tool]) -> List[ToolCard]:
        cards: List[ToolCard] = []
        for t in tools_list:
            schema_obj = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
            if schema_obj is None:
                json_schema = {"type": "object"}
            elif hasattr(schema_obj, "model_dump"):
                json_schema = schema_obj.model_dump()
            elif hasattr(schema_obj, "json_schema"):
                cand = schema_obj.json_schema() if callable(schema_obj.json_schema) else schema_obj.json_schema
                json_schema = cand if isinstance(cand, dict) else {"type": "object"}
            elif isinstance(schema_obj, dict):
                json_schema = schema_obj
            else:
                json_schema = {"type": "object"}
            cards.append(ToolCard(getattr(t, "name", "unknown"), getattr(t, "description", "") or "", json_schema))
        return cards

    @staticmethod
    def as_inventory_json(cards: List[ToolCard]) -> str:
        return json.dumps([{"name": c.name, "description": c.description, "json_schema": c.json_schema} for c in cards], indent=2)
