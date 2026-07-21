"""
Claude API integration for the Claude Investigator.
"""

import os
import re
from typing import Optional
from .config import Config


class ClaudeAnalyzer:
    """Handles Claude API interactions for analysis."""

    # Model mapping from standard Claude model names to Bedrock model IDs
    BEDROCK_MODEL_MAPPING = {
        "claude-opus-4-6-20260120": "us.anthropic.claude-opus-4-6",
        "claude-opus-4-5-20251101": "us.anthropic.claude-opus-4-5-20251101-v1:0",
        "claude-opus-4-1-20250805": "us.anthropic.claude-opus-4-1-20250805-v1:0",
        "claude-sonnet-4-6-20260120": "us.anthropic.claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-sonnet-4-20250514": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-opus-4-20250514": "us.anthropic.claude-opus-4-20250514-v1:0",
    }

    # X4 modification: when the worker supplies a source bundle, require every
    # concrete claim to remain traceable to its line-addressable evidence.
    SOURCE_GROUNDING_INSTRUCTIONS = """## Mandatory source-evidence rules

- Treat the Source Evidence Bundle as the factual basis for this analysis.
- Cite concrete claims with one or more exact `relative/path.ext:line` references from the bundle.
- Do not invent file contents, endpoints, events, controls, or deployment mechanisms.
- If evidence is insufficient, state what remains unknown and cite the files that were inspected; do not claim source access was unavailable.
- Distinguish "not found in the supplied evidence" from "does not exist".
- These rules override any later instruction to return only a short response such as "no HTTP API", "no events", or "No LLM usage detected".
- For a negative finding, write at least three complete sentences: describe the bounded search scope, cite one or more relevant `relative/path.ext:line` entries, and explain what was not found without turning that into a repository-wide certainty.
- Keep the requested RepoSwarm section name and answer the requested analysis directly.

"""

    SOURCE_CITATION_PATTERN = re.compile(
        r"(?<![\w./-])[\w.-]+(?:/[\w.-]+)*\.[A-Za-z0-9]+:\d+"
    )
    SOURCE_GROUNDING_REPAIR_INSTRUCTIONS = """Your previous response does not satisfy the mandatory source-evidence rules.

Rewrite the complete answer. It must contain at least 80 characters and at least one exact `relative/path.ext:line` citation copied from the Source Evidence Bundle. If the requested subsystem was not found, use at least three complete sentences to describe the bounded evidence inspected, cite relevant evidence, and state only that no matching implementation was found in that evidence. Do not return a terse phrase such as "no events" or "no database"."""

    def __init__(self, api_key: str, logger):
        self.logger = logger
        self.use_bedrock = (
            os.getenv('CLAUDE_PROVIDER') == 'bedrock' or
            os.getenv('CLAUDE_CODE_USE_BEDROCK') == '1'
        )

        if self.use_bedrock:
            from anthropic import AnthropicBedrock
            aws_region = os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
            self.client = AnthropicBedrock(aws_region=aws_region)
            self.logger.info(f"Using Bedrock provider in region {aws_region}")
        else:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.logger.info("Using standard Anthropic API")
    
    def _get_model_id(self, model_name: str) -> str:
        """
        Get the appropriate model ID for the current provider.

        For Bedrock, converts standard Claude model names to Bedrock model IDs.
        For standard API, returns the model name as-is.

        Args:
            model_name: Standard Claude model name

        Returns:
            Model ID appropriate for the current provider
        """
        if self.use_bedrock:
            bedrock_model = self.BEDROCK_MODEL_MAPPING.get(model_name)
            if bedrock_model:
                self.logger.debug(f"Mapped {model_name} to Bedrock model {bedrock_model}")
                return bedrock_model
            else:
                self.logger.info(f"No Bedrock mapping for {model_name}, using as-is")
                return model_name
        return model_name

    def clean_prompt(self, prompt_template: str) -> str:
        """
        Clean the prompt template by removing version lines and other metadata.
        
        Args:
            prompt_template: Raw prompt template that may contain version headers
            
        Returns:
            Cleaned prompt template ready for Claude
        """
        if not prompt_template:
            return prompt_template

        lines = prompt_template.split('\n')
        
        # Only clean if version line exists at the beginning
        if lines and lines[0].startswith('version'):
            lines = lines[1:]
            self.logger.debug("Removed version line from prompt")
            
            # Remove any leading empty lines after version removal
            while lines and lines[0].strip() == '':
                lines = lines[1:]
            
            cleaned_prompt = '\n'.join(lines)
            self.logger.debug(f"Cleaned prompt ({len(cleaned_prompt)} characters)")
            
            return cleaned_prompt
        else:
            # No version line found, return as-is
            return prompt_template

    @classmethod
    def _is_source_grounded_response(cls, analysis_text: str) -> bool:
        """Return whether a model response meets the minimum grounding contract."""
        return (
            len(analysis_text.strip()) >= 80
            and cls.SOURCE_CITATION_PATTERN.search(analysis_text) is not None
        )
    
    def analyze_with_context(self, prompt_template: str, repo_structure: str, 
                           previous_context: Optional[str] = None,
                           config_overrides: Optional[dict] = None,
                           usage_tag: Optional[str] = None) -> str:
        """
        Analyze using Claude with optional context from previous analyses.
        
        Args:
            prompt_template: Prompt template to use
            repo_structure: Repository structure string
            previous_context: Previous analysis results to include as context
            config_overrides: Optional dict with claude_model, max_tokens overrides
            
        Returns:
            Analysis result from Claude
        """
        if config_overrides is None:
            config_overrides = {}
        
        # Clean the prompt template first (remove version lines, etc.)
        cleaned_template = self.clean_prompt(prompt_template)
        
        # Replace placeholders in the cleaned prompt
        prompt = cleaned_template.replace("{repo_structure}", repo_structure)
        source_grounded = "## Source Evidence Bundle" in repo_structure
        
        # Add previous context if available
        if previous_context:
            context_section = f"\n\n## Previous Analysis Context\n\n{previous_context}\n\n"
            prompt = prompt.replace("{previous_context}", context_section)
        else:
            # Remove the placeholder if no context
            prompt = prompt.replace("{previous_context}", "")

        # Put the evidence rules last so legacy prompts that ask for a terse
        # "none found" answer cannot override the source-grounding contract.
        if source_grounded:
            prompt = f"{prompt}\n\n{self.SOURCE_GROUNDING_INSTRUCTIONS}"
        
        self.logger.debug(f"Prompt created ({len(prompt)} characters)")
        self.logger.debug(f"Prompt preview (first 1000 chars): {prompt[:1000]}...")
        
        try:
            # Use config overrides or defaults
            claude_model = config_overrides.get("claude_model") or Config.CLAUDE_MODEL
            max_tokens = config_overrides.get("max_tokens") or Config.MAX_TOKENS

            # Get the appropriate model ID for the current provider
            model_id = self._get_model_id(claude_model)

            self.logger.info("Sending analysis request to Claude API")
            self.logger.debug(f"Using model: {model_id}, max_tokens: {max_tokens}")

            request_kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if usage_tag:
                request_kwargs["metadata"] = {"user_id": usage_tag[:256]}
            response = self.client.messages.create(
                **request_kwargs
            )
            
            analysis_text = response.content[0].text
            self.logger.info(f"Received analysis from Claude ({len(analysis_text)} characters)")
            self.logger.debug(f"Analysis preview (first 1000 chars): {analysis_text[:1000]}...")

            if source_grounded and not self._is_source_grounded_response(analysis_text):
                self.logger.warning(
                    "Source-grounded response was too short or lacked a line citation; retrying once"
                )
                repair_response = self.client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    **({"metadata": {"user_id": usage_tag[:256]}} if usage_tag else {}),
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": analysis_text},
                        {"role": "user", "content": self.SOURCE_GROUNDING_REPAIR_INSTRUCTIONS},
                    ],
                )
                analysis_text = repair_response.content[0].text
                self.logger.info(
                    f"Received repaired analysis from Claude ({len(analysis_text)} characters)"
                )
                self.logger.debug(
                    f"Repaired analysis preview (first 1000 chars): {analysis_text[:1000]}..."
                )
            
            return analysis_text
            
        except Exception as e:
            self.logger.error(f"Claude API request failed: {str(e)}")
            raise Exception(f"Failed to get analysis from Claude: {str(e)}")
    
    def analyze_structure(self, repo_structure: str, prompt_template: str) -> str:
        """
        Analyze repository structure using Claude.
        
        Args:
            repo_structure: Repository structure string
            prompt_template: Prompt template to use
            
        Returns:
            Analysis result from Claude
        """
        return self.analyze_with_context(prompt_template, repo_structure, None)
