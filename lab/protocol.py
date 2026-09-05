"""Keep observed server properties distinct from declarations about model training."""

import hashlib
import re

import aiohttp

SHA256 = re.compile(r"[0-9a-f]{64}")


async def inspect_protocol(client, metadata: dict | None) -> dict:
    evidence = {
        "base_url": client.base_url,
        "model": client.model,
        "classification": "unverified",
        "declaration": None,
        "observed": {},
        "issues": [],
    }
    declarations = (metadata or {}).get("tool_protocols", [])
    if not isinstance(declarations, list) or any(not isinstance(d, dict) for d in declarations):
        raise ValueError("model metadata tool_protocols must be an array of objects")
    if any(
        not isinstance(d.get("base_url"), str) or not isinstance(d.get("model"), str)
        for d in declarations
    ):
        raise ValueError("Each tool protocol declaration requires string base_url and model fields")
    matches = [
        d
        for d in declarations
        if d.get("base_url", "").rstrip("/") == client.base_url and d.get("model") == client.model
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Duplicate tool protocol declarations for {client.model} at {client.base_url}"
        )
    declaration = matches[0] if matches else None
    evidence["declaration"] = declaration
    if not client.base_url.endswith("/v1"):
        evidence["issues"].append("Cannot locate /props: base URL does not end in /v1")
    else:
        url = client.base_url.removesuffix("/v1") + "/props"
        try:
            async with client.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=min(client.timeout_s, 10)),
                allow_redirects=False,
            ) as response:
                response.raise_for_status()
                props = await response.json()
                if response.status != 200 or not isinstance(props, dict):
                    raise ValueError("Expected HTTP 200 with a properties object")
                if any(
                    props.get(key) is not None and not isinstance(props[key], str)
                    for key in ("chat_template", "chat_template_tool_use", "build_info")
                ):
                    raise ValueError("Template and build properties must be strings or null")
                templates = {
                    key: props[key]
                    for key in ("chat_template", "chat_template_tool_use")
                    if isinstance(props.get(key), str) and props[key]
                }
                evidence["observed"] = {
                    "url": url,
                    "build_info": props.get("build_info"),
                    "model_path": props.get("model_path"),
                    "templates": templates,
                    "template_sha256": {
                        key: hashlib.sha256(value.encode()).hexdigest()
                        for key, value in templates.items()
                    },
                }
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            evidence["issues"].append(f"Server properties unavailable: {type(exc).__name__}: {exc}")

    if declaration is None:
        evidence["issues"].append(
            "No matching tool_protocols declaration; API success does not prove native support"
        )
        return evidence
    required = (
        "format",
        "handler",
        "handler_source",
        "gguf_sha256",
        "quantization",
        "build_info",
        "chat_template_sha256",
    )
    for key in required:
        if not isinstance(declaration.get(key), str) or not declaration[key].strip():
            evidence["issues"].append(f"Declaration requires a nonempty {key}")
    format = declaration.get("format")
    if not isinstance(format, str) or format not in {"native", "generic"}:
        evidence["issues"].append("Declared format must be native or generic")
    handler = declaration.get("handler")
    handler = handler.strip().lower() if isinstance(handler, str) else ""
    if declaration.get("format") == "native":
        if (
            not isinstance(declaration.get("training_source"), str)
            or not declaration["training_source"].strip()
        ):
            evidence["issues"].append("Native declaration requires a training_source reference")
        if handler == "generic":
            evidence["issues"].append("A Generic handler cannot be declared native")
    elif declaration.get("format") == "generic" and handler != "generic":
        evidence["issues"].append("Generic declarations must name the Generic handler")
    for key in ("gguf_sha256", "chat_template_sha256"):
        if not isinstance(declaration.get(key), str) or not SHA256.fullmatch(declaration[key]):
            evidence["issues"].append(f"Declaration {key} must be a lowercase SHA-256 hex digest")
    if "chat_template_tool_use_sha256" not in declaration:
        evidence["issues"].append(
            "Declare chat_template_tool_use_sha256 explicitly (null if absent)"
        )
    hashes = evidence["observed"].get("template_sha256", {})
    for key in ("chat_template", "chat_template_tool_use"):
        expected = declaration.get(key + "_sha256")
        actual = hashes.get(key)
        if expected != actual or (key == "chat_template" and not actual):
            evidence["issues"].append(f"Server {key} hash does not match declaration")
    if declaration.get("build_info") != evidence["observed"].get("build_info"):
        evidence["issues"].append("Server build_info does not match declaration")
    if not evidence["issues"]:
        evidence["classification"] = declaration["format"] + "_declared"
    # /props does not attest the weights or selected handler, nor a model's training history.
    evidence["verification_scope"] = (
        "Template hashes and build string only; weights, handler and training are user-declared"
    )
    return evidence
