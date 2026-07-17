"""Tests for the X4 bounded source-grounding extension."""

import logging
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from investigator.core.claude_analyzer import ClaudeAnalyzer
from investigator.core.config import Config
from investigator.core.repository_analyzer import RepositoryAnalyzer
from investigator.core.source_bundle import SourceBundleBuilder


def test_source_bundle_contains_line_addressable_evidence_and_redacts_secrets(tmp_path: Path):
    source = tmp_path / "src" / "auth" / "routes.ts"
    source.parent.mkdir(parents=True)
    source.write_text(
        "export const route = '/admin';\n"
        "const api_key = 'live-secret-value';\n"
        "export function authorize(role: string) { return role === 'admin'; }\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.production").write_text("API_KEY=must-not-leak\n", encoding="utf-8")

    bundle = SourceBundleBuilder(logging.getLogger(__name__), max_chars=10_000).build(str(tmp_path))

    assert "## Source Evidence Bundle" in bundle
    assert "src/auth/routes.ts:1" in bundle
    assert "src/auth/routes.ts:3" in bundle
    assert "live-secret-value" not in bundle
    assert "[REDACTED]" in bundle
    assert ".env.production" not in bundle
    assert "must-not-leak" not in bundle


def test_source_bundle_is_bounded_and_diverse(tmp_path: Path):
    for area in ("apps/admin", "packages/core", "src/runtime"):
        directory = tmp_path / area
        directory.mkdir(parents=True)
        for index in range(4):
            (directory / f"route-{index}.ts").write_text(
                "\n".join(f"export const route{line} = '/v{line}';" for line in range(80)),
                encoding="utf-8",
            )

    bundle = SourceBundleBuilder(
        logging.getLogger(__name__),
        max_chars=5_000,
        max_files=12,
        max_files_per_bucket=2,
    ).build(str(tmp_path))

    assert len(bundle) <= 5_010
    assert "apps/admin/" in bundle
    assert "packages/core/" in bundle
    assert "src/runtime/" in bundle


def test_repository_analyzer_appends_source_bundle_only_when_enabled(tmp_path: Path):
    (tmp_path / "app.ts").write_text("export const endpoint = '/health';\n", encoding="utf-8")
    analyzer = RepositoryAnalyzer(logging.getLogger(__name__))

    with patch.object(Config, "SOURCE_GROUNDING", False):
        structure_only = analyzer.get_structure(str(tmp_path))
    with patch.object(Config, "SOURCE_GROUNDING", True), patch.object(Config, "SOURCE_BUNDLE_MAX_CHARS", 10_000):
        grounded = analyzer.get_structure(str(tmp_path))

    assert "## Source Evidence Bundle" not in structure_only
    assert "## Source Evidence Bundle" in grounded
    assert "app.ts:1" in grounded


@patch('anthropic.Anthropic')
def test_claude_analyzer_requires_exact_source_citations_when_bundle_is_present(mock_anthropic):
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Grounded result")]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client
    analyzer = ClaudeAnalyzer("test-key", Mock())
    analyzer.client = mock_client

    result = analyzer.analyze_with_context(
        "Analyze APIs from {repo_structure}",
        "Repository: fixture\n\n## Source Evidence Bundle\napp.ts:1 | export const endpoint = '/health';",
    )

    assert result == "Grounded result"
    sent_prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Mandatory source-evidence rules" in sent_prompt
    assert "relative/path.ext:line" in sent_prompt
    assert "app.ts:1" in sent_prompt
    assert "override any later instruction" in sent_prompt
    assert "at least three complete sentences" in sent_prompt
    assert sent_prompt.rstrip().endswith("answer the requested analysis directly.")


@patch('anthropic.Anthropic')
def test_source_grounding_overrides_terse_negative_finding_instructions(mock_anthropic):
    mock_client = Mock()
    mock_client.messages.create.return_value = Mock(content=[Mock(text="Grounded negative finding")])
    mock_anthropic.return_value = mock_client
    analyzer = ClaudeAnalyzer("test-key", Mock())
    analyzer.client = mock_client

    analyzer.analyze_with_context(
        'If no APIs exist, simply return "no HTTP API".\n\n{repo_structure}',
        "## Source Evidence Bundle\npackage.json:1 | {",
    )

    sent_prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert sent_prompt.index('simply return "no HTTP API"') < sent_prompt.index("override any later instruction")
