"""
LLM Client for Jarvis-Granite Live Telemetry.

Provides interface to IBM Granite LLM via WatsonX for generating
race engineer responses. Uses prompt templates for consistent
personality and context injection.

Features:
- Proactive responses (event-triggered)
- Reactive responses (query-driven)
- Configurable verbosity levels
- Retry logic with Tenacity
"""

import json
from typing import Any, Dict, Optional, Union

from jarvis_granite.schemas.events import Event
from jarvis_granite.live.context import LiveSessionContext


class LLMError(Exception):
    """Exception raised for LLM-related errors."""
    pass


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

PROACTIVE_TEMPLATE_MINIMAL = """You are a concise F1 race engineer. Alert the driver about: {event_type}.
Data: {event_data}
Context: {context}
Respond in under 15 words."""

PROACTIVE_TEMPLATE_MODERATE = """You are an experienced F1 race engineer speaking to your driver during a race.
An event has been detected that requires your attention.

Event Type: {event_type}
Event Data: {event_data}

Current Session Context:
{context}

Provide a clear, actionable radio message (1-2 sentences). Be direct but informative.
Focus on what the driver needs to know and any recommended action."""

PROACTIVE_TEMPLATE_VERBOSE = """You are an experienced F1 race engineer speaking to your driver during a race.
An important event has been detected.

Event Type: {event_type}
Event Details: {event_data}

Current Session Context:
{context}

Provide a detailed radio message explaining:
1. What happened
2. The implications
3. Recommended action
Keep it under 4 sentences for clarity during racing."""

REACTIVE_TEMPLATE_MINIMAL = """You are a concise F1 race engineer. Driver asks: "{query}"
Context: {context}
Answer in under 15 words."""

REACTIVE_TEMPLATE_MODERATE = """You are an experienced F1 race engineer responding to your driver's question during a race.

Driver's Question: "{query}"

Current Session Context:
{context}

Provide a clear, helpful response (1-2 sentences). Be direct and informative.
Focus on answering the question with relevant data from the context."""

REACTIVE_TEMPLATE_VERBOSE = """You are an experienced F1 race engineer responding to your driver's question during a race.

Driver's Question: "{query}"

Current Session Context:
{context}

Provide a comprehensive response that:
1. Directly answers the question
2. Includes relevant supporting data
3. Offers any strategic insights if applicable
Keep it under 4 sentences for clarity during racing."""


class LLMClient:
    """
    Client for IBM Granite LLM via WatsonX.

    Generates race engineer responses for both proactive (event-triggered)
    and reactive (query-driven) scenarios.

    Attributes:
        model_id: WatsonX model identifier
        max_tokens: Maximum tokens for response generation
        temperature: LLM temperature for response variety
        verbosity: Response verbosity level (minimal, moderate, verbose)
        max_retries: Maximum retry attempts for failed requests
    """

    def __init__(
        self,
        watsonx_url: str,
        watsonx_project_id: str,
        watsonx_api_key: str,
        model_id: str = "ibm/granite-3-8b-instruct",
        max_tokens: int = 150,
        temperature: float = 0.7,
        verbosity: str = "moderate",
        max_retries: int = 3
    ):
        """
        Initialize LLM Client.

        Args:
            watsonx_url: WatsonX API URL
            watsonx_project_id: WatsonX project ID
            watsonx_api_key: WatsonX API key
            model_id: Model identifier (default: ibm/granite-3-8b-instruct)
            max_tokens: Maximum response tokens (default: 150)
            temperature: Response temperature (default: 0.7)
            verbosity: Response verbosity - minimal, moderate, verbose (default: moderate)
            max_retries: Max retry attempts (default: 3)
        """
        self.watsonx_url = watsonx_url
        self.watsonx_project_id = watsonx_project_id
        self.watsonx_api_key = watsonx_api_key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbosity = verbosity
        self.max_retries = max_retries

        # Select templates based on verbosity
        self._setup_templates()

    def _setup_templates(self) -> None:
        """Set up prompt templates based on verbosity level."""
        if self.verbosity == "minimal":
            self.proactive_template = PROACTIVE_TEMPLATE_MINIMAL
            self.reactive_template = REACTIVE_TEMPLATE_MINIMAL
        elif self.verbosity == "verbose":
            self.proactive_template = PROACTIVE_TEMPLATE_VERBOSE
            self.reactive_template = REACTIVE_TEMPLATE_VERBOSE
        else:  # moderate (default)
            self.proactive_template = PROACTIVE_TEMPLATE_MODERATE
            self.reactive_template = REACTIVE_TEMPLATE_MODERATE

    def format_proactive_prompt(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        context: str
    ) -> str:
        """
        Format proactive prompt for event-triggered response.

        Args:
            event_type: Type of event (e.g., fuel_critical, tire_warning)
            event_data: Event-specific data
            context: Session context string

        Returns:
            Formatted prompt string
        """
        # Format event data for readability
        if isinstance(event_data, dict):
            event_data_str = json.dumps(event_data, indent=2)
        else:
            event_data_str = str(event_data)

        return self.proactive_template.format(
            event_type=event_type,
            event_data=event_data_str,
            context=context
        )

    def format_reactive_prompt(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Format reactive prompt for query-driven response.

        Args:
            query: Driver's question
            context: Session context string

        Returns:
            Formatted prompt string
        """
        return self.reactive_template.format(
            query=query,
            context=context
        )

    def format_context(
        self,
        context: Union[LiveSessionContext, str]
    ) -> str:
        """
        Format session context for prompt injection.

        Args:
            context: LiveSessionContext instance or pre-formatted string

        Returns:
            Formatted context string
        """
        if isinstance(context, str):
            return context

        # Use context's built-in formatting method
        return context.to_prompt_context()

    async def generate_proactive_response(
        self,
        event: Event,
        context: Union[LiveSessionContext, str]
    ) -> str:
        """
        Generate proactive response for detected event.

        Args:
            event: Detected event
            context: Session context

        Returns:
            Generated response text

        Raises:
            LLMError: If LLM invocation fails
        """
        context_str = self.format_context(context)

        prompt = self.format_proactive_prompt(
            event_type=event.type,
            event_data=event.data,
            context=context_str
        )

        try:
            response = await self._invoke_llm(prompt)
            return self._clean_response(response)
        except Exception as e:
            raise LLMError(f"Failed to generate proactive response: {e}") from e

    async def generate_reactive_response(
        self,
        query: str,
        context: Union[LiveSessionContext, str]
    ) -> str:
        """
        Generate reactive response for driver query.

        Args:
            query: Driver's question
            context: Session context

        Returns:
            Generated response text

        Raises:
            LLMError: If LLM invocation fails
        """
        context_str = self.format_context(context)

        prompt = self.format_reactive_prompt(
            query=query,
            context=context_str
        )

        try:
            response = await self._invoke_llm(prompt)
            return self._clean_response(response)
        except Exception as e:
            raise LLMError(f"Failed to generate reactive response: {e}") from e

    async def _invoke_llm(self, prompt: str) -> str:
        """
        Invoke the LLM with retry logic.

        In production, this would use langchain_ibm.WatsonxLLM.
        For testing, this method can be mocked.

        Args:
            prompt: Formatted prompt

        Returns:
            LLM response text
        """
        # Placeholder for actual WatsonX invocation
        # In production:
        # from langchain_ibm import WatsonxLLM
        # llm = WatsonxLLM(
        #     model_id=self.model_id,
        #     url=self.watsonx_url,
        #     project_id=self.watsonx_project_id,
        #     apikey=self.watsonx_api_key,
        # )
        # return await llm.ainvoke(prompt)

        # For now, return placeholder (will be mocked in tests)
        return f"Response to: {prompt[:50]}..."

    def _clean_response(self, response: str) -> str:
        """
        Clean and format LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response string
        """
        if not response:
            return ""

        # Strip whitespace and normalize
        cleaned = response.strip()

        return cleaned
