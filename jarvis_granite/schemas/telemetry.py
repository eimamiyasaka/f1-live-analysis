"""
Telemetry data schemas for Jarvis-Granite Live Telemetry.

Defines Pydantic models for:
- TireTemps: Tire temperatures for all four corners
- TireWear: Tire wear percentages
- GForces: Lateral and longitudinal G-forces
- TelemetryData: Complete telemetry snapshot
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TireTemps(BaseModel):
    """
    Tire temperatures in Celsius for all four corners.

    Attributes:
        fl: Front-left tire temperature
        fr: Front-right tire temperature
        rl: Rear-left tire temperature
        rr: Rear-right tire temperature
    """
    fl: float = Field(..., description="Front-left tire temperature (°C)")
    fr: float = Field(..., description="Front-right tire temperature (°C)")
    rl: float = Field(..., description="Rear-left tire temperature (°C)")
    rr: float = Field(..., description="Rear-right tire temperature (°C)")

    @field_validator('fl', 'fr', 'rl', 'rr')
    @classmethod
    def temperature_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Tire temperature cannot be negative')
        return v


class TireWear(BaseModel):
    """
    Tire wear percentages for all four corners.

    Values range from 0 (fresh) to 100 (completely worn).

    Attributes:
        fl: Front-left tire wear percentage
        fr: Front-right tire wear percentage
        rl: Rear-left tire wear percentage
        rr: Rear-right tire wear percentage
    """
    fl: float = Field(..., ge=0, le=100, description="Front-left tire wear (%)")
    fr: float = Field(..., ge=0, le=100, description="Front-right tire wear (%)")
    rl: float = Field(..., ge=0, le=100, description="Rear-left tire wear (%)")
    rr: float = Field(..., ge=0, le=100, description="Rear-right tire wear (%)")


class GForces(BaseModel):
    """
    G-force measurements.

    Attributes:
        lateral: Lateral G-force (positive = right turn)
        longitudinal: Longitudinal G-force (positive = acceleration)
    """
    lateral: float = Field(..., description="Lateral G-force")
    longitudinal: float = Field(..., description="Longitudinal G-force")


class TelemetryData(BaseModel):
    """
    Complete telemetry data snapshot from a racing session.

    This schema matches the telemetry data format from Section 10
    of the documentation.
    """
    # Speed and engine
    speed_kmh: float = Field(..., ge=0, description="Current speed in km/h")
    rpm: int = Field(..., ge=0, description="Engine RPM")
    gear: int = Field(..., ge=0, le=8, description="Current gear (0=neutral)")

    # Driver inputs
    throttle: float = Field(..., ge=0, le=1, description="Throttle position (0-1)")
    brake: float = Field(..., ge=0, le=1, description="Brake pressure (0-1)")
    steering_angle: float = Field(..., ge=-1, le=1, description="Steering input (-1 to 1)")

    # Resources
    fuel_remaining: float = Field(..., ge=0, description="Fuel remaining in liters")

    # Tire data
    tire_temps: TireTemps = Field(..., description="Tire temperatures")
    tire_wear: TireWear = Field(..., description="Tire wear percentages")

    # Physics
    g_forces: GForces = Field(..., description="G-force measurements")

    # Track position
    track_position: float = Field(..., ge=0, le=1, description="Position on track (0-1)")
    lap_number: int = Field(..., ge=0, description="Current lap number")
    lap_time_current: float = Field(..., ge=0, description="Current lap time in seconds")
    sector: int = Field(..., ge=1, le=3, description="Current sector (1-3)")

    # Race position (optional - may not be available in all modes)
    position: Optional[int] = Field(default=None, ge=1, description="Race position")
    gap_ahead: Optional[float] = Field(default=None, description="Gap to car ahead (seconds)")
    gap_behind: Optional[float] = Field(default=None, description="Gap to car behind (seconds)")
