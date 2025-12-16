"""
Voice module for Jarvis-Granite Live Telemetry.

Provides Text-to-Speech (TTS) and Speech-to-Text (STT) clients
using IBM Watson services.
"""

from jarvis_granite.voice.watson_tts import WatsonTTSClient, TTSError
from jarvis_granite.voice.watson_stt import WatsonSTTClient, STTError

__all__ = [
    "WatsonTTSClient",
    "TTSError",
    "WatsonSTTClient",
    "STTError",
]
