"""
Agents for Jarvis-Granite Live Telemetry.

This module contains specialized agents for different aspects
of the live telemetry system:

- TelemetryAgent: Rule-based event detection from telemetry data
- LLMClient: IBM Granite LLM client for response generation
"""

from jarvis_granite.agents.telemetry_agent import TelemetryAgent
from jarvis_granite.agents.llm_client import LLMClient, LLMError

__all__ = ["TelemetryAgent", "LLMClient", "LLMError"]
