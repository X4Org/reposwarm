"""Bounded, secret-aware source evidence for repository analysis.

X4 modification: RepoSwarm previously supplied only a depth-limited filename
tree to the model. This module adds line-addressable source excerpts behind an
opt-in worker setting so architecture claims can be grounded in repository
evidence without sending an unbounded checkout.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceCandidate:
    path: Path
    relative_path: str
    score: int
    bucket: str


class SourceBundleBuilder:
    """Build a deterministic, bounded bundle of high-signal source excerpts."""

    ALLOWED_SUFFIXES = {
        ".bicep", ".c", ".cc", ".conf", ".cpp", ".cs", ".css", ".go",
        ".graphql", ".h", ".hpp", ".html", ".java", ".js", ".json",
        ".jsx", ".kt", ".kts", ".md", ".mjs", ".mts", ".php", ".ps1",
        ".py", ".rb", ".rs", ".sh", ".sql", ".swift", ".tf", ".toml",
        ".ts", ".tsx", ".vue", ".xml", ".yaml", ".yml",
    }
    SKIP_DIRS = {
        ".git", ".next", ".nuxt", ".pytest_cache", ".tox", ".venv",
        "__pycache__", "build", "coverage", "dist", "node_modules", "target", "venv",
    }
    SKIP_NAMES = {
        "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "uv.lock",
        "poetry.lock", "cargo.lock", "composer.lock", "gemfile.lock",
    }
    HIGH_SIGNAL_NAMES = {
        "agents.md", "dockerfile", "pyproject.toml",
        "vite.config.ts", "docker-compose.yml", "docker-compose.yaml",
        "compose.yml", "compose.yaml", "openapi.json", "openapi.yaml", "openapi.yml",
    }
    PATH_KEYWORDS = {
        "agent", "api", "auth", "database", "deploy", "event", "feature", "flag",
        "infra", "migration", "model", "monitor", "observability", "permission",
        "policy", "prompt", "route", "schema", "security", "service", "workflow",
    }
    CONTENT_KEYWORDS = re.compile(
        r"\b(route|endpoint|router|controller|auth|permission|role|event|queue|topic|"
        r"deploy|workflow|database|migration|schema|entity|model|metric|trace|log|"
        r"feature.?flag|prompt|agent|secret|token)\b",
        re.IGNORECASE,
    )
    SECRET_ASSIGNMENT = re.compile(
        r"(?i)\b(api[_-]?key|client[_-]?secret|password|private[_-]?key|"
        r"access[_-]?token|refresh[_-]?token|connection[_-]?string)\b(\s*[:=]\s*)(.+)$"
    )

    def __init__(
        self,
        logger,
        max_chars: int = 120_000,
        max_files: int = 120,
        max_file_bytes: int = 128_000,
        max_lines_per_file: int = 180,
        max_files_per_bucket: int = 10,
    ):
        self.logger = logger
        self.max_chars = max_chars
        self.max_files = max_files
        self.max_file_bytes = max_file_bytes
        self.max_lines_per_file = max_lines_per_file
        self.max_files_per_bucket = max_files_per_bucket

    def build(self, repo_path: str) -> str:
        root = Path(repo_path).resolve()
        selected = self._select_candidates(self._discover_candidates(root))
        parts = [
            "## Source Evidence Bundle\n\n"
            "The following excerpts are repository source evidence. Each line is prefixed with\n"
            "`relative/path:line`; use that exact form when citing architecture claims.\n"
            "Content that resembles a credential assignment is replaced with `[REDACTED]`."
        ]
        total_chars = len(parts[0])
        per_file_budget = max(400, min(12_000, self.max_chars // max(10, min(self.max_files, 30))))
        included = 0
        for candidate in selected:
            excerpt = self._excerpt(candidate)
            if not excerpt:
                continue
            block = f"\n\n### `{candidate.relative_path}`\n\n```text\n{excerpt}\n```"
            if len(block) > per_file_budget:
                block = block[:per_file_budget].rsplit("\n", 1)[0] + "\n```"
            remaining = self.max_chars - total_chars
            if remaining <= 200:
                break
            if len(block) > remaining:
                block = block[:remaining].rsplit("\n", 1)[0] + "\n```"
            parts.append(block)
            total_chars += len(block)
            included += 1
            if total_chars >= self.max_chars:
                break
        self.logger.info(
            "Source grounding bundle captured %s files (%s characters)",
            included,
            total_chars,
        )
        return "".join(parts)

    def _discover_candidates(self, root: Path) -> list[SourceCandidate]:
        candidates: list[SourceCandidate] = []
        for current_root, dirs, files in os.walk(root):
            dirs[:] = sorted(directory for directory in dirs if directory not in self.SKIP_DIRS)
            for filename in sorted(files):
                path = Path(current_root) / filename
                relative = path.relative_to(root).as_posix()
                if not self._eligible(path, relative):
                    continue
                candidates.append(SourceCandidate(path, relative, self._score(relative), self._bucket(relative)))
        return candidates

    def _eligible(self, path: Path, relative: str) -> bool:
        name = path.name.lower()
        if name in self.SKIP_NAMES or name.startswith(".env"):
            return False
        if path.suffix.lower() in {".key", ".pem", ".p12", ".pfx", ".crt", ".cer"}:
            return False
        if path.suffix.lower() not in self.ALLOWED_SUFFIXES and name not in {"dockerfile", "makefile"}:
            return False
        try:
            size = path.stat().st_size
        except OSError:
            return False
        return 0 < size <= self.max_file_bytes and "/.git/" not in f"/{relative}/"

    def _score(self, relative: str) -> int:
        lowered = relative.lower()
        name = Path(relative).name.lower()
        score = 10
        if name in self.HIGH_SIGNAL_NAMES:
            score += 70
        elif name == "package.json":
            score += 25
        elif name == "tsconfig.json":
            score += 10
        elif name == "readme.md":
            score += 10
        if lowered.startswith(".github/workflows/"):
            score += 90
        if lowered.startswith(("infra/", "db/", "migrations/", "azure_template/")):
            score += 35
        score += sum(14 for keyword in self.PATH_KEYWORDS if keyword in lowered)
        if Path(relative).suffix.lower() in {".ts", ".tsx", ".js", ".mjs", ".py", ".go", ".rs", ".java", ".cs"}:
            score += 10
        return score

    def _bucket(self, relative: str) -> str:
        parts = relative.split("/")
        if len(parts) >= 2 and parts[0] in {"apps", "packages", "src"}:
            return "/".join(parts[:2])
        return parts[0] if len(parts) > 1 else "(root)"

    def _select_candidates(self, candidates: list[SourceCandidate]) -> list[SourceCandidate]:
        buckets: dict[str, list[SourceCandidate]] = {}
        for candidate in candidates:
            buckets.setdefault(candidate.bucket, []).append(candidate)
        for values in buckets.values():
            values.sort(key=lambda item: (-item.score, item.relative_path))
        bucket_order = sorted(
            buckets,
            key=lambda bucket: (-buckets[bucket][0].score, bucket),
        )
        selected: list[SourceCandidate] = []
        for position in range(self.max_files_per_bucket):
            for bucket in bucket_order:
                values = buckets[bucket]
                if position >= len(values):
                    continue
                selected.append(values[position])
                if len(selected) >= self.max_files:
                    return selected
        return selected

    def _excerpt(self, candidate: SourceCandidate) -> str:
        try:
            content = candidate.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""
        lines = content.splitlines()
        selected_lines = self._select_lines(lines)
        rendered = []
        in_private_key = False
        for line_number in selected_lines:
            value = lines[line_number - 1]
            if "-----BEGIN" in value and "PRIVATE KEY-----" in value:
                in_private_key = True
            if in_private_key:
                value = "[REDACTED PRIVATE KEY]"
            if "-----END" in value and "PRIVATE KEY-----" in value:
                in_private_key = False
            value = self.SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)
            rendered.append(f"{candidate.relative_path}:{line_number} | {value}")
        return "\n".join(rendered)

    def _select_lines(self, lines: list[str]) -> list[int]:
        if len(lines) <= self.max_lines_per_file:
            return list(range(1, len(lines) + 1))
        selected = set(range(1, min(31, len(lines) + 1)))
        for index, line in enumerate(lines, start=1):
            if not self.CONTENT_KEYWORDS.search(line):
                continue
            selected.update(range(max(1, index - 2), min(len(lines), index + 2) + 1))
            if len(selected) >= self.max_lines_per_file - 10:
                break
        selected.update(range(max(1, len(lines) - 9), len(lines) + 1))
        return sorted(selected)[: self.max_lines_per_file]
