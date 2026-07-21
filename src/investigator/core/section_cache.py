"""Content-addressed prompt cache identities for incremental investigations.

The upstream cache keys every prompt by repository commit. X4 can instead bind
each prompt to the evidence and dependency outputs that can affect that section.
This keeps the existing workflow and artifact assembly intact while allowing an
unrelated source change to reuse a previously grounded section result.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SECTION_CACHE_POLICY_VERSION = "1"

_SOURCE_BUNDLE_MARKER = "## Source Evidence Bundle"
_TEMP_REPO_HEADER = re.compile(r"(?m)^Repository: (?P<name>.+)_[0-9a-f]{8}$")
_SOURCE_BLOCK = re.compile(
    r"\n\n### `(?P<path>[^`]+)`\n\n```text\n(?P<body>.*?)\n```",
    re.DOTALL,
)

# These sections make repository-wide claims, so every grounded source block
# remains part of their identity. Specialized sections can be more selective.
_GLOBAL_SECTIONS = {
    "hl_overview",
    "module_deep_dive",
    "core_entities",
    "data_mapping",
    "security_check",
}

_SECTION_KEYWORDS = {
    "dependencies": ("package", "depend", "requirements", "pyproject", "pom.xml", "cargo.toml", "go.mod", "lock"),
    "DBs": ("database", "db", "schema", "migration", "entity", "model", "sql", "repository", "prisma", "orm"),
    "APIs": ("api", "route", "router", "endpoint", "controller", "graphql", "openapi", "request", "response", "http"),
    "events": ("event", "queue", "topic", "broker", "message", "kafka", "servicebus", "pubsub", "webhook"),
    "service_dependencies": ("service", "client", "integration", "http", "grpc", "queue", "topic", "api"),
    "deployment": ("deploy", "docker", "compose", "workflow", "pipeline", "infra", "terraform", "bicep", "helm", "k8s", "azure", "aws"),
    "authentication": ("auth", "login", "identity", "oauth", "oidc", "jwt", "session", "token", "credential"),
    "authorization": ("authoriz", "permission", "policy", "role", "rbac", "scope", "access", "guard"),
    "monitoring": ("monitor", "observ", "metric", "trace", "telemetry", "logger", "logging", "alert", "health"),
    "ml_services": ("ai", "llm", "model", "openai", "anthropic", "azure", "foundry", "prompt", "embedding", "vector"),
    "feature_flags": ("feature", "flag", "toggle", "launchdarkly", "experiment"),
    "prompt_security_check": ("prompt", "llm", "agent", "model", "openai", "anthropic", "injection", "tool", "mcp"),
}

_BASELINE_NAMES = {
    "agents.md",
    "dockerfile",
    "package.json",
    "pyproject.toml",
    "readme.md",
}


def _section_evidence(repo_structure: str, step_name: str) -> str:
    """Return the deterministic evidence subset relevant to one section."""
    repo_structure = _TEMP_REPO_HEADER.sub(r"Repository: \g<name>", repo_structure, count=1)
    marker_at = repo_structure.find(_SOURCE_BUNDLE_MARKER)
    if marker_at < 0 or step_name in _GLOBAL_SECTIONS:
        return repo_structure

    structure = repo_structure[:marker_at]
    source = repo_structure[marker_at:]
    keywords = _SECTION_KEYWORDS.get(step_name)
    if not keywords:
        return repo_structure

    selected: list[str] = []
    selected_paths: list[str] = []
    for match in _SOURCE_BLOCK.finditer(source):
        path = match.group("path")
        body = match.group("body")
        searchable = f"{path}\n{body}".lower()
        if path.rsplit("/", 1)[-1].lower() in _BASELINE_NAMES or any(keyword in searchable for keyword in keywords):
            selected_paths.append(path)
            selected.append(match.group(0))

    # Include matching tree entries so adding a clearly relevant file changes
    # the identity even before it is selected into the bounded source bundle.
    tree_lines = []
    for line in structure.splitlines():
        lowered = line.lower()
        if len(tree_lines) < 3 or any(keyword in lowered for keyword in keywords):
            tree_lines.append(line)

    header = source.split("\n\n### `", 1)[0]
    return "\n".join(tree_lines) + "\n\n" + header + "".join(selected) + "\npaths=" + json.dumps(selected_paths)


def section_cache_identity(
    *,
    repo_structure: str,
    prompt_content: str,
    previous_context: str | None,
    step_name: str,
    prompt_version: str,
    config_overrides: dict[str, Any] | None,
) -> str:
    """Build a stable cache identity from the effective section inputs."""
    overrides = config_overrides or {}
    relevant_config = {
        key: overrides.get(key)
        for key in ("claude_model", "max_tokens", "temperature")
        if overrides.get(key) is not None
    }
    payload = {
        "policy": SECTION_CACHE_POLICY_VERSION,
        "step": step_name,
        "promptVersion": prompt_version,
        "prompt": prompt_content,
        "evidence": _section_evidence(repo_structure, step_name),
        "context": previous_context or "",
        "config": relevant_config,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
