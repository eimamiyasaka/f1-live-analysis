"""
WebSocket Handler for Jarvis-Granite Live Telemetry.

Handles WebSocket connections, message routing, and session management.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import ValidationError

from jarvis_granite.live.context import LiveSessionContext
from jarvis_granite.schemas.telemetry import TelemetryData
from jarvis_granite.schemas.messages import (
    parse_client_message,
    SessionInitMessage,
    TelemetryMessage,
    TextQueryMessage,
    ConfigUpdateMessage,
    SessionEndMessage,
)
from jarvis_granite.agents.telemetry_agent import TelemetryAgent
from config.config import ThresholdsConfig


def _utcnow_iso() -> str:
    """Get current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class WebSocketHandler:
    """
    Handles WebSocket connections and message routing.

    Manages active sessions, routes messages to appropriate handlers,
    and coordinates between the telemetry agent and session context.

    Attributes:
        active_sessions: Dictionary of active session data keyed by session_id
        telemetry_agent: Agent for rule-based telemetry processing
    """

    def __init__(self, thresholds: Optional[ThresholdsConfig] = None):
        """
        Initialize WebSocketHandler.

        Args:
            thresholds: Optional threshold configuration for telemetry agent
        """
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.telemetry_agent = TelemetryAgent(thresholds=thresholds)

    async def handle_connection(self, websocket) -> None:
        """
        Handle a new WebSocket connection.

        Accepts the connection and begins message processing loop.

        Args:
            websocket: FastAPI WebSocket instance
        """
        await websocket.accept()

        session_id: Optional[str] = None

        try:
            while True:
                # Receive and parse message
                try:
                    data = await websocket.receive_json()
                except json.JSONDecodeError:
                    await self._send_error(
                        websocket,
                        "VALIDATION_ERROR",
                        "Invalid JSON format"
                    )
                    continue

                # Route message
                result = await self.handle_message(data, websocket, session_id)

                # Update session_id if session was initialized
                if result and result.get("session_id"):
                    session_id = result["session_id"]

        except Exception as e:
            # Clean up session on disconnect
            if session_id and session_id in self.active_sessions:
                del self.active_sessions[session_id]

    async def handle_message(
        self,
        data: Dict[str, Any],
        websocket,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Route and handle an incoming message.

        Args:
            data: Parsed JSON message data
            websocket: WebSocket connection
            session_id: Current session ID (if any)

        Returns:
            Result dictionary with session_id if applicable
        """
        message_type = data.get("type")

        # Route based on message type
        if message_type == "session_init":
            return await self._handle_session_init(data, websocket)

        elif message_type == "telemetry":
            return await self._handle_telemetry(data, websocket, session_id)

        elif message_type == "text_query":
            return await self._handle_text_query(data, websocket, session_id)

        elif message_type == "config_update":
            return await self._handle_config_update(data, websocket, session_id)

        elif message_type == "session_end":
            return await self._handle_session_end(data, websocket, session_id)

        else:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Unknown message type: {message_type}"
            )
            return None

    async def _handle_session_init(
        self,
        data: Dict[str, Any],
        websocket
    ) -> Optional[Dict[str, Any]]:
        """Handle session initialization message."""
        try:
            # Validate message
            message = SessionInitMessage(**data)
        except ValidationError as e:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Invalid session_init: {str(e)}"
            )
            return None

        # Check for duplicate session
        if message.session_id in self.active_sessions:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Session {message.session_id} already exists"
            )
            return None

        # Create session context
        context = LiveSessionContext(
            session_id=message.session_id,
            source=message.source,
            track_name=message.track_name
        )

        # Store session
        self.active_sessions[message.session_id] = {
            "context": context,
            "config": message.config.copy(),
            "websocket": websocket
        }

        # Generate LiveKit token (placeholder - real implementation would use LiveKit SDK)
        livekit_token = self._generate_livekit_token(message.session_id)

        # Send session confirmed response
        response = {
            "type": "session_confirmed",
            "session_id": message.session_id,
            "config": message.config,
            "livekit": {
                "url": "wss://livekit.example.com",
                "token": livekit_token,
                "room_name": f"{message.session_id}_voice"
            }
        }

        await websocket.send_json(response)

        return {"session_id": message.session_id}

    async def _handle_telemetry(
        self,
        data: Dict[str, Any],
        websocket,
        session_id: Optional[str]
    ) -> None:
        """Handle telemetry message."""
        # Check session exists
        if not session_id or session_id not in self.active_sessions:
            await self._send_error(
                websocket,
                "SESSION_NOT_FOUND",
                "No active session. Send session_init first."
            )
            return None

        try:
            # Validate message
            message = TelemetryMessage(**data)
        except ValidationError as e:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Invalid telemetry: {str(e)}"
            )
            return None

        session = self.active_sessions[session_id]
        context = session["context"]

        # Update context with telemetry
        context.update(message.data)

        # Detect events using telemetry agent
        events = self.telemetry_agent.detect_events(message.data, context)

        # Process detected events (would queue for orchestrator in full implementation)
        for event in events:
            # Log or handle events
            pass

        return None

    async def _handle_text_query(
        self,
        data: Dict[str, Any],
        websocket,
        session_id: Optional[str]
    ) -> None:
        """Handle text query message."""
        # Check session exists
        if not session_id or session_id not in self.active_sessions:
            await self._send_error(
                websocket,
                "SESSION_NOT_FOUND",
                "No active session. Send session_init first."
            )
            return None

        try:
            # Validate message
            message = TextQueryMessage(**data)
        except ValidationError as e:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Invalid text_query: {str(e)}"
            )
            return None

        session = self.active_sessions[session_id]
        context = session["context"]

        # In full implementation, this would:
        # 1. Queue the query for the orchestrator
        # 2. Get AI response
        # 3. Send ai_response message back
        # For now, just acknowledge receipt

        return None

    async def _handle_config_update(
        self,
        data: Dict[str, Any],
        websocket,
        session_id: Optional[str]
    ) -> None:
        """Handle configuration update message."""
        # Check session exists
        if not session_id or session_id not in self.active_sessions:
            await self._send_error(
                websocket,
                "SESSION_NOT_FOUND",
                "No active session. Send session_init first."
            )
            return None

        try:
            # Validate message
            message = ConfigUpdateMessage(**data)
        except ValidationError as e:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Invalid config_update: {str(e)}"
            )
            return None

        session = self.active_sessions[session_id]

        # Update session config
        session["config"].update(message.config)

        return None

    async def _handle_session_end(
        self,
        data: Dict[str, Any],
        websocket,
        session_id: Optional[str]
    ) -> None:
        """Handle session end message."""
        if session_id and session_id in self.active_sessions:
            del self.active_sessions[session_id]

        return None

    async def send_heartbeat(self, session_id: str, websocket) -> None:
        """
        Send heartbeat message to client.

        Args:
            session_id: Session to send heartbeat for
            websocket: WebSocket connection
        """
        is_active = session_id in self.active_sessions

        heartbeat = {
            "type": "heartbeat",
            "timestamp": _utcnow_iso(),
            "session_active": is_active
        }

        await websocket.send_json(heartbeat)

    async def _send_error(
        self,
        websocket,
        error_code: str,
        message: str
    ) -> None:
        """Send error message to client."""
        error = {
            "type": "error",
            "error_code": error_code,
            "message": message,
            "timestamp": _utcnow_iso()
        }

        await websocket.send_json(error)

    def _generate_livekit_token(self, session_id: str) -> str:
        """
        Generate LiveKit access token.

        In production, this would use the LiveKit SDK to generate
        a proper JWT token. For now, returns a placeholder.

        Args:
            session_id: Session ID for the token

        Returns:
            JWT token string
        """
        # Placeholder - real implementation would use:
        # from livekit import api
        # token = api.AccessToken(api_key, api_secret)
        # token.add_grant(api.VideoGrant(room_join=True, room=f"{session_id}_voice"))
        # return token.to_jwt()

        return f"placeholder_token_{session_id}_{int(time.time())}"
