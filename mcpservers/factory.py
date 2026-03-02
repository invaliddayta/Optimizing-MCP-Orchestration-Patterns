import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from mcp import StdioServerParameters

_SERVER_CONFIG_CACHE: Dict[str, Any] = {}

def _load_server_config() -> Dict[str, Any]:
    """
    Loads config/servers_config.json from the central config directory.
    """
    if _SERVER_CONFIG_CACHE:
        return _SERVER_CONFIG_CACHE

    base_path = Path(os.getcwd())
    config_path = base_path / "config" / "servers_config.json"
    
    if not config_path.exists():
        current_file = Path(__file__)
        fallback_path = current_file.parent.parent / "config" / "servers_config.json"
        if fallback_path.exists():
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"Could not find servers_config.json at {config_path} or {fallback_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _SERVER_CONFIG_CACHE.update(data)
            return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse servers_config.json: {e}")

def server_params_for_role(role: str) -> StdioServerParameters:
    """
    Returns the StdioServerParameters for a given role based on config.
    """
    role = (role or "").lower()
    config = _load_server_config()
    
    servers = config.get("servers", {})
    defaults = config.get("defaults", {})
    
    if role not in servers:
        print(f"[Warning] No server config found for role '{role}'. Defaulting to calculator.")
        server_def = servers.get("calculator-expert", {"command": sys.executable, "args": ["-m", "mcp_server_calculator"]})
    else:
        server_def = servers[role]

    cmd = server_def.get("command", "python")
    if cmd == "${PYTHON}":
        cmd = sys.executable

    raw_args = server_def.get("args", [])
    resolved_args = []
    for arg in raw_args:
        if arg == "${PYTHON}":
            resolved_args.append(sys.executable)
        else:
            resolved_args.append(arg)

    full_env = dict(os.environ)
    full_env.update(defaults.get("environment", {}))
    full_env.update(server_def.get("environment", {}))

    return StdioServerParameters(
        command=cmd,
        args=resolved_args,
        env=full_env,
    )
