"""
FastAPI Application for Jarvis-Granite Live Telemetry.

Provides WebSocket endpoint at /live for real-time telemetry streaming
and bidirectional communication.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from jarvis_granite.live.websocket_handler import WebSocketHandler


def _utcnow_iso() -> str:
    """Get current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Jarvis-Granite Live Telemetry",
        description="Real-time F1 telemetry analysis with AI race engineer",
        version="0.1.0"
    )

    # Create shared WebSocket handler
    handler = WebSocketHandler()

    @app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring."""
        return {
            "status": "healthy",
            "timestamp": _utcnow_iso(),
            "active_sessions": len(handler.active_sessions)
        }

    @app.websocket("/live")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint for live telemetry streaming.

        Connection Flow:
        1. Client connects to /live
        2. Client sends session_init message
        3. Server responds with session_confirmed (includes LiveKit token)
        4. Client streams telemetry, server sends AI responses
        5. Client sends session_end when done
        """
        await websocket.accept()

        session_id: Optional[str] = None

        try:
            while True:
                # Receive message
                try:
                    raw_data = await websocket.receive_text()
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    await _send_error(
                        websocket,
                        "VALIDATION_ERROR",
                        "Invalid JSON format"
                    )
                    continue

                # Route message through handler
                result = await handler.handle_message(data, websocket, session_id)

                # Track session_id after initialization
                if result and result.get("session_id"):
                    session_id = result["session_id"]

        except WebSocketDisconnect:
            # Clean up on disconnect
            if session_id and session_id in handler.active_sessions:
                del handler.active_sessions[session_id]
        except Exception as e:
            # Clean up on error
            if session_id and session_id in handler.active_sessions:
                del handler.active_sessions[session_id]

    return app


async def _send_error(websocket: WebSocket, error_code: str, message: str) -> None:
    """Send error message to client."""
    error = {
        "type": "error",
        "error_code": error_code,
        "message": message,
        "timestamp": _utcnow_iso()
    }
    await websocket.send_json(error)


# Create default app instance for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
