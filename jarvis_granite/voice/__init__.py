"""
Voice module for Jarvis-Granite Live Telemetry.

Provides Text-to-Speech (TTS) and Speech-to-Text (STT) clients
using IBM Watson services, and WebRTC transport via LiveKit.

Phase 5, Sections 11-12:
- Watson TTS/STT Clients (Section 11)
- LiveKit Integration (Section 12)
"""

from jarvis_granite.voice.watson_tts import WatsonTTSClient, TTSError
from jarvis_granite.voice.watson_stt import WatsonSTTClient, STTError
from jarvis_granite.voice.livekit_client import LiveKitClient, LiveKitError, VoiceAgent

__all__ = [
    # Watson TTS/STT
    "WatsonTTSClient",
    "TTSError",
    "WatsonSTTClient",
    "STTError",
    # LiveKit
    "LiveKitClient",
    "LiveKitError",
    "VoiceAgent",
]
