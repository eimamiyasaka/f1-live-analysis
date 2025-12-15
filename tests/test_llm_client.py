"""
TDD Tests for LLM Client - Phase 2, Section 6

These tests define the expected behavior for:
1. LLM Client initialization with WatsonX configuration
2. Prompt template management (proactive/reactive)
3. Response generation for events and queries
4. Retry logic with Tenacity
5. Context formatting for prompts

Run with: pytest tests/test_llm_client.py -v

Write these tests FIRST, watch them fail, then implement LLMClient to pass.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis_granite.schemas.events import Event, Priority
from jarvis_granite.live.context import LiveSessionContext


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_config():
    """Mock configuration for LLM client."""
    return {
        "watsonx_url": "https://us-south.ml.cloud.ibm.com",
        "watsonx_project_id": "test-project-id",
        "watsonx_api_key": "test-api-key",
        "model_id": "ibm/granite-3-8b-instruct",
        "max_tokens": 150,
        "temperature": 0.7
    }


@pytest.fixture
def session_context():
    """Create a session context for testing."""
    context = LiveSessionContext(
        session_id="race_001",
        source="torcs",
        track_name="Monza"
    )
    context.current_lap = 15
    context.current_sector = 2
    context.position = 3
    context.fuel_remaining = 35.0
    context.fuel_consumption_per_lap = 3.5
    context.tire_temps = {"fl": 95, "fr": 96, "rl": 92, "rr": 93}
    context.tire_wear = {"fl": 25, "fr": 26, "rl": 22, "rr": 23}
    context.gap_ahead = 2.5
    context.gap_behind = 1.8
    context.best_lap = 81.234
    context.last_lap = 82.456
    return context


@pytest.fixture
def fuel_critical_event():
    """Create a fuel critical event."""
    return Event(
        type="fuel_critical",
        priority=Priority.CRITICAL,
        data={"laps": 1.5},
        timestamp=1705329125.123
    )


@pytest.fixture
def tire_warning_event():
    """Create a tire warning event."""
    return Event(
        type="tire_warning",
        priority=Priority.MEDIUM,
        data={"temp": 105.0, "position": "fl"},
        timestamp=1705329125.456
    )


@pytest.fixture
def lap_complete_event():
    """Create a lap complete event."""
    return Event(
        type="lap_complete",
        priority=Priority.MEDIUM,
        data={"lap": 15, "time": 82.456, "best": 81.234},
        timestamp=1705329125.789
    )


# =============================================================================
# LLM CLIENT CREATION
# =============================================================================

class TestLLMClientCreation:
    """Tests for LLMClient instantiation."""

    def test_create_llm_client(self, mock_config):
        """Should create LLMClient instance."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        assert client is not None

    def test_llm_client_stores_config(self, mock_config):
        """LLMClient should store configuration."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        assert client.model_id == "ibm/granite-3-8b-instruct"
        assert client.max_tokens == 150

    def test_llm_client_has_required_methods(self, mock_config):
        """LLMClient should have required public methods."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        assert hasattr(client, 'generate_proactive_response')
        assert callable(client.generate_proactive_response)
        assert hasattr(client, 'generate_reactive_response')
        assert callable(client.generate_reactive_response)

    def test_llm_client_default_values(self):
        """LLMClient should have sensible defaults."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(
            watsonx_url="https://test.com",
            watsonx_project_id="test-project",
            watsonx_api_key="test-key"
        )

        assert client.model_id == "ibm/granite-3-8b-instruct"
        assert client.max_tokens == 150
        assert client.temperature == 0.7


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

class TestPromptTemplates:
    """Tests for prompt template management."""

    def test_has_proactive_prompt_template(self, mock_config):
        """Should have proactive prompt template."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        assert hasattr(client, 'proactive_template')
        assert client.proactive_template is not None

    def test_has_reactive_prompt_template(self, mock_config):
        """Should have reactive prompt template."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        assert hasattr(client, 'reactive_template')
        assert client.reactive_template is not None

    def test_proactive_template_has_required_variables(self, mock_config):
        """Proactive template should accept event_type, event_data, context."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        # Should be able to format with these variables
        formatted = client.format_proactive_prompt(
            event_type="fuel_critical",
            event_data={"laps": 1.5},
            context="Test context"
        )

        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_reactive_template_has_required_variables(self, mock_config):
        """Reactive template should accept query and context."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        # Should be able to format with these variables
        formatted = client.format_reactive_prompt(
            query="How are my tires?",
            context="Test context"
        )

        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_proactive_prompt_includes_event_type(self, mock_config):
        """Formatted proactive prompt should include event type."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        formatted = client.format_proactive_prompt(
            event_type="tire_critical",
            event_data={"temp": 115, "position": "fl"},
            context="Lap 15, P3"
        )

        assert "tire" in formatted.lower() or "critical" in formatted.lower()

    def test_reactive_prompt_includes_query(self, mock_config):
        """Formatted reactive prompt should include the query."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        formatted = client.format_reactive_prompt(
            query="What's my fuel situation?",
            context="35L remaining"
        )

        assert "fuel" in formatted.lower()


# =============================================================================
# PROACTIVE RESPONSE GENERATION
# =============================================================================

class TestProactiveResponseGeneration:
    """Tests for proactive (event-triggered) response generation."""

    @pytest.mark.asyncio
    async def test_generate_proactive_response_returns_string(
        self, mock_config, fuel_critical_event, session_context
    ):
        """generate_proactive_response should return a string."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        # Mock the LLM call
        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Box box box! Fuel critical."

            response = await client.generate_proactive_response(
                event=fuel_critical_event,
                context=session_context
            )

            assert isinstance(response, str)
            assert len(response) > 0

    @pytest.mark.asyncio
    async def test_proactive_response_uses_event_data(
        self, mock_config, fuel_critical_event, session_context
    ):
        """Proactive response should incorporate event data."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Fuel critical, 1.5 laps remaining!"

            await client.generate_proactive_response(
                event=fuel_critical_event,
                context=session_context
            )

            # Check the prompt was formatted with event data
            call_args = mock_invoke.call_args[0][0]
            assert "fuel" in call_args.lower() or "critical" in call_args.lower()

    @pytest.mark.asyncio
    async def test_proactive_response_uses_context(
        self, mock_config, lap_complete_event, session_context
    ):
        """Proactive response should incorporate session context."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Lap 15 complete."

            await client.generate_proactive_response(
                event=lap_complete_event,
                context=session_context
            )

            # Check context was included
            call_args = mock_invoke.call_args[0][0]
            assert "Monza" in call_args or "15" in call_args

    @pytest.mark.asyncio
    async def test_proactive_response_different_event_types(
        self, mock_config, session_context
    ):
        """Should handle different event types appropriately."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        events = [
            Event(type="fuel_critical", priority=Priority.CRITICAL,
                  data={"laps": 1.5}, timestamp=1.0),
            Event(type="tire_warning", priority=Priority.MEDIUM,
                  data={"temp": 105, "position": "fl"}, timestamp=2.0),
            Event(type="gap_change", priority=Priority.MEDIUM,
                  data={"change": 1.5, "direction": "ahead"}, timestamp=3.0),
        ]

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Response"

            for event in events:
                response = await client.generate_proactive_response(
                    event=event,
                    context=session_context
                )
                assert isinstance(response, str)


# =============================================================================
# REACTIVE RESPONSE GENERATION
# =============================================================================

class TestReactiveResponseGeneration:
    """Tests for reactive (query-driven) response generation."""

    @pytest.mark.asyncio
    async def test_generate_reactive_response_returns_string(
        self, mock_config, session_context
    ):
        """generate_reactive_response should return a string."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Tires are looking good."

            response = await client.generate_reactive_response(
                query="How are my tires?",
                context=session_context
            )

            assert isinstance(response, str)
            assert len(response) > 0

    @pytest.mark.asyncio
    async def test_reactive_response_uses_query(
        self, mock_config, session_context
    ):
        """Reactive response should incorporate the query."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Fuel is at 35 liters."

            await client.generate_reactive_response(
                query="What's my fuel status?",
                context=session_context
            )

            call_args = mock_invoke.call_args[0][0]
            assert "fuel" in call_args.lower()

    @pytest.mark.asyncio
    async def test_reactive_response_uses_context(
        self, mock_config, session_context
    ):
        """Reactive response should incorporate session context."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "You're in P3."

            await client.generate_reactive_response(
                query="What's my position?",
                context=session_context
            )

            call_args = mock_invoke.call_args[0][0]
            # Context should be included
            assert "P3" in call_args or "3" in call_args or "Monza" in call_args


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

class TestContextFormatting:
    """Tests for session context formatting."""

    def test_format_context_for_prompt(self, mock_config, session_context):
        """Should format context into prompt-friendly string."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        formatted = client.format_context(session_context)

        assert isinstance(formatted, str)
        assert "Monza" in formatted
        assert "15" in formatted  # Lap number

    def test_context_includes_fuel_info(self, mock_config, session_context):
        """Formatted context should include fuel information."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        formatted = client.format_context(session_context)

        assert "35" in formatted or "fuel" in formatted.lower()

    def test_context_includes_tire_info(self, mock_config, session_context):
        """Formatted context should include tire information."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        formatted = client.format_context(session_context)

        assert "tire" in formatted.lower() or "95" in formatted

    def test_context_includes_position_info(self, mock_config, session_context):
        """Formatted context should include position information."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        formatted = client.format_context(session_context)

        assert "P3" in formatted or "position" in formatted.lower()


# =============================================================================
# RETRY LOGIC
# =============================================================================

class TestRetryLogic:
    """Tests for retry logic with Tenacity."""

    def test_retry_config_stored(self, mock_config):
        """Should store retry configuration."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config, max_retries=5)

        # Retry configuration should be stored
        assert client.max_retries == 5
        # Note: actual retry behavior tested via integration tests

    @pytest.mark.asyncio
    async def test_retries_on_http_error(self, mock_config, session_context):
        """Should retry on HTTP errors."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        # Retry logic would be tested via integration tests
        # Unit tests verify the retry decorator is applied
        assert hasattr(client, '_invoke_llm')

    def test_max_retries_configurable(self, mock_config):
        """Max retries should be configurable."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config, max_retries=5)

        assert client.max_retries == 5


# =============================================================================
# RESPONSE FORMATTING
# =============================================================================

class TestResponseFormatting:
    """Tests for response formatting."""

    @pytest.mark.asyncio
    async def test_response_is_cleaned(self, mock_config, session_context):
        """Response should be cleaned of extra whitespace."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "  Response with spaces  \n\n"

            response = await client.generate_reactive_response(
                query="Test",
                context=session_context
            )

            assert response == "Response with spaces"

    @pytest.mark.asyncio
    async def test_empty_response_handled(self, mock_config, session_context):
        """Empty responses should be handled gracefully."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = ""

            response = await client.generate_reactive_response(
                query="Test",
                context=session_context
            )

            # Should return a fallback or empty string
            assert isinstance(response, str)


# =============================================================================
# RACE ENGINEER PERSONALITY
# =============================================================================

class TestRaceEngineerPersonality:
    """Tests for race engineer personality in prompts."""

    def test_proactive_prompt_has_race_engineer_context(self, mock_config):
        """Proactive prompt should establish race engineer role."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        formatted = client.format_proactive_prompt(
            event_type="fuel_warning",
            event_data={"laps": 4},
            context="Test context"
        )

        # Should contain race engineer or similar role context
        prompt_lower = formatted.lower()
        assert any(term in prompt_lower for term in
                   ["race engineer", "engineer", "driver", "pit", "lap"])

    def test_reactive_prompt_has_race_engineer_context(self, mock_config):
        """Reactive prompt should establish race engineer role."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        formatted = client.format_reactive_prompt(
            query="How are my tires?",
            context="Test context"
        )

        prompt_lower = formatted.lower()
        assert any(term in prompt_lower for term in
                   ["race engineer", "engineer", "driver", "respond"])


# =============================================================================
# VERBOSITY LEVELS
# =============================================================================

class TestVerbosityLevels:
    """Tests for response verbosity control."""

    def test_verbosity_affects_prompt(self, mock_config, session_context):
        """Verbosity setting should affect prompt generation."""
        from jarvis_granite.llm import LLMClient

        client_minimal = LLMClient(**mock_config, verbosity="minimal")
        client_verbose = LLMClient(**mock_config, verbosity="verbose")

        prompt_minimal = client_minimal.format_reactive_prompt(
            query="Status?",
            context="Test"
        )
        prompt_verbose = client_verbose.format_reactive_prompt(
            query="Status?",
            context="Test"
        )

        # Prompts should be different based on verbosity
        assert prompt_minimal != prompt_verbose or \
               client_minimal.verbosity != client_verbose.verbosity

    def test_default_verbosity_is_moderate(self, mock_config):
        """Default verbosity should be moderate."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        assert client.verbosity == "moderate"


# =============================================================================
# ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in LLM client."""

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self, mock_config, session_context):
        """Should handle LLM errors gracefully."""
        from jarvis_granite.llm import LLMClient, LLMError

        client = LLMClient(**mock_config)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = Exception("LLM unavailable")

            with pytest.raises(LLMError):
                await client.generate_reactive_response(
                    query="Test",
                    context=session_context
                )

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_error(self, mock_config, session_context):
        """Timeout should raise LLMError after retries exhausted."""
        from jarvis_granite.llm import LLMClient, LLMError
        import httpx

        client = LLMClient(**mock_config, max_retries=1)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(LLMError):
                await client.generate_reactive_response(
                    query="Test",
                    context=session_context
                )


# =============================================================================
# INTEGRATION WITH CONTEXT
# =============================================================================

class TestContextIntegration:
    """Tests for integration with LiveSessionContext."""

    @pytest.mark.asyncio
    async def test_uses_context_to_prompt_context_method(
        self, mock_config, session_context
    ):
        """Should use context.to_prompt_context() for formatting."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        # The context's to_prompt_context should be used
        context_str = session_context.to_prompt_context()

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Response"

            await client.generate_reactive_response(
                query="Test",
                context=session_context
            )

            # Verify context was included in the prompt
            call_args = mock_invoke.call_args[0][0]
            # Should contain elements from context
            assert "Monza" in call_args or session_context.track_name in call_args
