"""
FastAPI Application for Jarvis-Granite Live Telemetry.

Provides WebSocket endpoint at /live for real-time telemetry streaming
and bidirectional communication with AI-powered race engineer responses.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from jarvis_granite.live.websocket_handler import WebSocketHandler
from jarvis_granite.llm import LLMClient
from config.config import LiveConfig
from config.config_loader import load_live_config, load_environment_settings

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    """Get current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def create_llm_client() -> Optional[LLMClient]:
    """
    Create LLM client from environment settings.

    Returns:
        LLMClient if credentials are available, None otherwise
    """
    try:
        env = load_environment_settings()

        # Check if credentials are provided
        if not env.watsonx_api_key or not env.watsonx_project_id:
            logger.warning(
                "WatsonX credentials not configured - AI responses will use fallback"
            )
            # Return client anyway - it will use fallback responses
            return LLMClient(
                watsonx_url=env.watsonx_url,
                watsonx_project_id=env.watsonx_project_id or "placeholder",
                watsonx_api_key=env.watsonx_api_key or "placeholder",
            )

        return LLMClient(
            watsonx_url=env.watsonx_url,
            watsonx_project_id=env.watsonx_project_id,
            watsonx_api_key=env.watsonx_api_key,
        )

    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        return None


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Jarvis-Granite Live Telemetry",
        description="Real-time F1 telemetry analysis with AI race engineer",
        version="2.1.0"
    )

    # Load configuration
    try:
        config = load_live_config()
    except Exception as e:
        logger.warning(f"Failed to load config, using defaults: {e}")
        config = LiveConfig()

    # Create LLM client
    llm_client = create_llm_client()

    # Create shared WebSocket handler with LLM support
    handler = WebSocketHandler(
        thresholds=config.thresholds,
        llm_client=llm_client,
        config=config,
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint for monitoring."""
        return {
            "status": "healthy",
            "timestamp": _utcnow_iso(),
            "active_sessions": len(handler.active_sessions),
            "ai_enabled": handler.race_engineer_agent is not None,
        }

    @app.get("/config")
    async def get_config():
        """Get current configuration (for debugging)."""
        return {
            "verbosity": config.verbosity.level,
            "thresholds": {
                "tire_temp_warning": config.thresholds.tire_temp_warning,
                "tire_temp_critical": config.thresholds.tire_temp_critical,
                "fuel_warning_laps": config.thresholds.fuel_warning_laps,
                "fuel_critical_laps": config.thresholds.fuel_critical_laps,
                "gap_change_threshold": config.thresholds.gap_change_threshold,
            },
            "min_proactive_interval": config.min_proactive_interval_seconds,
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
                logger.info(f"Session {session_id} disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
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

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
