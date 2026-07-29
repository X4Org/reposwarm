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
from investigator.core.section_cache import section_cache_identity


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


def test_source_bundle_preserves_secret_environment_references(tmp_path: Path):
    source = tmp_path / "src" / "db.ts"
    source.parent.mkdir(parents=True)
    source.write_text(
        "const config = {\n"
        "  password: process.env.DB_READ_PASSWORD || process.env.DB_PASSWORD,\n"
        "  apiKey: process.env.API_KEY,\n"
        "  clientSecret: 'live-secret-value',\n"
        "};\n",
        encoding="utf-8",
    )

    bundle = SourceBundleBuilder(logging.getLogger(__name__), max_chars=10_000).build(str(tmp_path))

    assert "password: process.env.DB_READ_PASSWORD || process.env.DB_PASSWORD" in bundle
    assert "apiKey: process.env.API_KEY" in bundle
    assert "live-secret-value" not in bundle
    assert "clientSecret: [REDACTED]" in bundle


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


def test_source_grounding_policy_versions_prompt_cache_entries():
    with patch.object(Config, "SOURCE_GROUNDING", False):
        assert Config.prompt_cache_version("2") == "2"
    with patch.object(Config, "SOURCE_GROUNDING", True):
        with patch.object(Config, "SECTION_CACHE", False):
            assert Config.prompt_cache_version("2") == "2-x4g3"
        with patch.object(Config, "SECTION_CACHE", True):
            assert Config.prompt_cache_version("2") == "2-x4g3-sc1"


def test_section_cache_reuses_unaffected_specialized_evidence():
    before = """Repository: fixture

  api.ts
  db.ts

## Source Evidence Bundle

### `src/api.ts`

```text
src/api.ts:1 | export const route = '/v1';
```

### `src/db.ts`

```text
src/db.ts:1 | export const schema = 'users';
```"""
    after = before.replace("'/v1'", "'/v2'")
    common = {
        "prompt_content": "Analyze databases from {repo_structure}",
        "previous_context": None,
        "step_name": "DBs",
        "prompt_version": "1-x4g3-sc1",
        "config_overrides": {"claude_model": "test-model"},
    }

    assert section_cache_identity(repo_structure=before, **common) == section_cache_identity(repo_structure=after, **common)


def test_section_cache_ignores_generated_checkout_suffix():
    before = """Repository: fixture_0bf8526a
============================

  api.ts

## Source Evidence Bundle

### `src/api.ts`

```text
src/api.ts:1 | export const route = '/v1';
```"""
    after = before.replace("fixture_0bf8526a", "fixture_9bb59064")
    common = {
        "prompt_content": "Analyze {repo_structure}",
        "previous_context": None,
        "step_name": "hl_overview",
        "prompt_version": "2-x4g3-sc1",
        "config_overrides": {"claude_model": "test-model"},
    }

    assert section_cache_identity(repo_structure=before, **common) == section_cache_identity(repo_structure=after, **common)


def test_section_cache_invalidates_matching_evidence_and_global_sections():
    before = """Repository: fixture

  db.ts

## Source Evidence Bundle

### `src/db.ts`

```text
src/db.ts:1 | export const schema = 'users';
```"""
    after = before.replace("'users'", "'accounts'")
    base = {
        "prompt_content": "Analyze {repo_structure}",
        "previous_context": None,
        "prompt_version": "1-x4g3-sc1",
        "config_overrides": {},
    }

    assert section_cache_identity(repo_structure=before, step_name="DBs", **base) != section_cache_identity(repo_structure=after, step_name="DBs", **base)
    assert section_cache_identity(repo_structure=before, step_name="hl_overview", **base) != section_cache_identity(repo_structure=after, step_name="hl_overview", **base)


def test_section_cache_invalidates_when_dependency_context_changes():
    structure = "Repository: fixture\n\n## Source Evidence Bundle"
    base = {
        "repo_structure": structure,
        "prompt_content": "Analyze services from {repo_structure} {previous_context}",
        "step_name": "service_dependencies",
        "prompt_version": "1-x4g3-sc1",
        "config_overrides": {},
    }

    assert section_cache_identity(previous_context="API v1", **base) != section_cache_identity(previous_context="API v2", **base)


def test_section_cache_invalidates_when_model_policy_changes():
    base = {
        "repo_structure": "Repository: fixture\n\n## Source Evidence Bundle",
        "prompt_content": "Analyze {repo_structure}",
        "previous_context": None,
        "step_name": "APIs",
        "prompt_version": "1-x4g3-sc1",
    }

    mini = section_cache_identity(config_overrides={"claude_model": "mini", "max_tokens": 6000}, **base)
    nano = section_cache_identity(config_overrides={"claude_model": "nano", "max_tokens": 6000}, **base)
    assert mini != nano


@patch('anthropic.Anthropic')
def test_claude_analyzer_requires_exact_source_citations_when_bundle_is_present(mock_anthropic):
    mock_client = Mock()
    mock_response = Mock()
    grounded_result = (
        "The fixture exposes a health endpoint in the supplied evidence at app.ts:1. "
        "This statement is bounded to the source bundle and does not infer other APIs."
    )
    mock_response.content = [Mock(text=grounded_result)]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic.return_value = mock_client
    analyzer = ClaudeAnalyzer("test-key", Mock())
    analyzer.client = mock_client

    result = analyzer.analyze_with_context(
        "Analyze APIs from {repo_structure}",
        "Repository: fixture\n\n## Source Evidence Bundle\napp.ts:1 | export const endpoint = '/health';",
        usage_tag="reposwarm:fixture:APIs",
    )

    assert result == grounded_result
    sent_prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "Mandatory source-evidence rules" in sent_prompt
    assert "relative/path.ext:line" in sent_prompt
    assert "app.ts:1" in sent_prompt
    assert "override any later instruction" in sent_prompt
    assert "at least three complete sentences" in sent_prompt
    assert sent_prompt.rstrip().endswith("answer the requested analysis directly.")
    assert mock_client.messages.create.call_args.kwargs["metadata"]["user_id"] == "reposwarm:fixture:APIs"


@patch('anthropic.Anthropic')
def test_source_grounding_overrides_terse_negative_finding_instructions(mock_anthropic):
    mock_client = Mock()
    grounded_result = (
        "The bounded evidence includes package metadata at package.json:1. "
        "No HTTP API implementation was found in that supplied evidence. "
        "This is not a repository-wide claim beyond the inspected bundle."
    )
    mock_client.messages.create.return_value = Mock(content=[Mock(text=grounded_result)])
    mock_anthropic.return_value = mock_client
    analyzer = ClaudeAnalyzer("test-key", Mock())
    analyzer.client = mock_client

    analyzer.analyze_with_context(
        'If no APIs exist, simply return "no HTTP API".\n\n{repo_structure}',
        "## Source Evidence Bundle\npackage.json:1 | {",
    )

    sent_prompt = mock_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert sent_prompt.index('simply return "no HTTP API"') < sent_prompt.index("override any later instruction")


@patch('anthropic.Anthropic')
def test_source_grounding_retries_a_terse_uncited_response_once(mock_anthropic):
    mock_client = Mock()
    repaired_result = (
        "The bounded event search inspected the package configuration at package.json:1. "
        "No event broker integration was found in the supplied evidence. "
        "That result does not establish that events are absent outside the inspected bundle."
    )
    mock_client.messages.create.side_effect = [
        Mock(content=[Mock(text="no events")]),
        Mock(content=[Mock(text=repaired_result)]),
    ]
    mock_anthropic.return_value = mock_client
    analyzer = ClaudeAnalyzer("test-key", Mock())
    analyzer.client = mock_client

    result = analyzer.analyze_with_context(
        "Analyze events from {repo_structure}",
        "## Source Evidence Bundle\npackage.json:1 | {",
    )

    assert result == repaired_result
    assert mock_client.messages.create.call_count == 2
    repair_messages = mock_client.messages.create.call_args.kwargs["messages"]
    assert repair_messages[1] == {"role": "assistant", "content": "no events"}
    assert "at least 80 characters" in repair_messages[2]["content"]
    assert "relative/path.ext:line" in repair_messages[2]["content"]


@patch('anthropic.Anthropic')
def test_source_grounding_repairs_missing_paths_and_out_of_bundle_ranges(mock_anthropic):
    mock_client = Mock()
    invalid_result = (
        "The API implementation is described at src/missing.ts:10 and "
        "src/api.ts:1-4, which purportedly establishes its request boundary."
    )
    repaired_result = (
        "The supplied evidence shows the API request boundary at src/api.ts:1-2. "
        "This claim is limited to the exact lines present in the source bundle."
    )
    mock_client.messages.create.side_effect = [
        Mock(content=[Mock(text=invalid_result)]),
        Mock(content=[Mock(text=repaired_result)]),
    ]
    mock_anthropic.return_value = mock_client
    analyzer = ClaudeAnalyzer("test-key", Mock())
    analyzer.client = mock_client

    result = analyzer.analyze_with_context(
        "Analyze APIs from {repo_structure}",
        "## Source Evidence Bundle\n"
        "src/api.ts:1 | export const route = '/health';\n"
        "src/api.ts:2 | export const method = 'GET';",
    )

    assert result == repaired_result
    assert mock_client.messages.create.call_count == 2
    repair_prompt = mock_client.messages.create.call_args.kwargs["messages"][2]["content"]
    assert "cited path is absent from the Source Evidence Bundle: src/missing.ts" in repair_prompt
    assert "cited range is absent from the Source Evidence Bundle: src/api.ts:1-4" in repair_prompt


@patch('anthropic.Anthropic')
def test_source_grounding_repairs_unverified_high_severity_security_findings(mock_anthropic):
    mock_client = Mock()
    invalid_result = (
        "**Severity:** HIGH\n"
        "**Classification:** Architecture risk\n"
        "**Evidence status:** Partial\n"
        "**Exploit preconditions:** Administrative log access.\n"
        "The logging risk is visible at src/logger.ts:1."
    )
    repaired_result = (
        "**Severity:** MEDIUM\n"
        "**Classification:** Architecture risk\n"
        "**Evidence status:** Partial\n"
        "**Exploit preconditions:** Administrative log access.\n"
        "**Existing controls checked:** Operator-only log access.\n"
        "The bounded risk is visible at src/logger.ts:1."
    )
    mock_client.messages.create.side_effect = [
        Mock(content=[Mock(text=invalid_result)]),
        Mock(content=[Mock(text=repaired_result)]),
    ]
    mock_anthropic.return_value = mock_client
    analyzer = ClaudeAnalyzer("test-key", Mock())
    analyzer.client = mock_client

    result = analyzer.analyze_with_context(
        "## X4 evidence classification contract\nAnalyze security from {repo_structure}",
        "## Source Evidence Bundle\nsrc/logger.ts:1 | console.error(error.stack);",
    )

    assert result == repaired_result
    assert mock_client.messages.create.call_count == 2
    repair_prompt = mock_client.messages.create.call_args.kwargs["messages"][2]["content"]
    assert "re-evaluate severity" in repair_prompt
    assert "missing verified evidence fields" in repair_prompt


@patch('anthropic.Anthropic')
def test_source_grounding_repairs_high_findings_with_privileged_only_preconditions(mock_anthropic):
    mock_client = Mock()
    invalid_result = (
        "**Severity:** HIGH\n"
        "**Classification:** Verified vulnerability\n"
        "**Evidence status:** Confirmed\n"
        "**Exploit preconditions:** The attacker can trigger an error written to operator logs.\n"
        "**Existing controls checked:** No redaction is visible.\n"
        "The logger writes stack traces at src/logger.ts:1."
    )
    repaired_result = (
        "**Severity:** MEDIUM\n"
        "**Classification:** Architecture risk\n"
        "**Evidence status:** Confirmed\n"
        "**Exploit preconditions:** The attacker can trigger an error; log access is separately privileged.\n"
        "**Existing controls checked:** No redaction is visible.\n"
        "The bounded logging risk is visible at src/logger.ts:1."
    )
    mock_client.messages.create.side_effect = [
        Mock(content=[Mock(text=invalid_result)]),
        Mock(content=[Mock(text=repaired_result)]),
    ]
    mock_anthropic.return_value = mock_client
    analyzer = ClaudeAnalyzer("test-key", Mock())
    analyzer.client = mock_client

    result = analyzer.analyze_with_context(
        "## X4 evidence classification contract\nAnalyze security from {repo_structure}",
        "## Source Evidence Bundle\nsrc/logger.ts:1 | console.error(error.stack);",
    )

    assert result == repaired_result
    repair_prompt = mock_client.messages.create.call_args.kwargs["messages"][2]["content"]
    assert "privileged-only prerequisite" in repair_prompt


@patch('anthropic.Anthropic')
def test_source_grounding_repairs_high_findings_based_on_missing_controls(mock_anthropic):
    mock_client = Mock()
    invalid_result = (
        "**Severity:** HIGH\n"
        "**Classification:** Verified vulnerability\n"
        "**Evidence status:** Confirmed\n"
        "**Exploit preconditions:** Any attacker-controlled data that reaches this sink.\n"
        "**Existing controls checked:** No parameter binding is shown.\n"
        "The current value is a static query-builder reference, so this is not a proven "
        "active injection path at src/repository.ts:1."
    )
    repaired_result = (
        "**Severity:** MEDIUM\n"
        "**Classification:** Architecture risk\n"
        "**Evidence status:** Partial\n"
        "**Exploit preconditions:** A future change would need to introduce attacker-controlled data.\n"
        "**Existing controls checked:** The current value is a static query-builder reference.\n"
        "The bounded pattern is visible at src/repository.ts:1."
    )
    mock_client.messages.create.side_effect = [
        Mock(content=[Mock(text=invalid_result)]),
        Mock(content=[Mock(text=repaired_result)]),
    ]
    mock_anthropic.return_value = mock_client
    analyzer = ClaudeAnalyzer("test-key", Mock())
    analyzer.client = mock_client

    result = analyzer.analyze_with_context(
        "## X4 evidence classification contract\nAnalyze security from {repo_structure}",
        "## Source Evidence Bundle\nsrc/repository.ts:1 | const ref = knex.ref('posts.id');",
    )

    assert result == repaired_result
    repair_prompt = mock_client.messages.create.call_args.kwargs["messages"][2]["content"]
    assert "uncertainty or missing-control reasoning" in repair_prompt
