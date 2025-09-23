from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class Resolution(BaseModel):
    width: int = Field(..., ge=0)
    height: int = Field(..., ge=0)


class TemperatureRange(BaseModel):
    min: float = Field(..., le=1000.0)  # Reasonable upper for thermal
    max: float = Field(..., ge=-100.0)


class Anomaly(BaseModel):
    frame_number: int = Field(..., ge=0)
    timestamp: datetime
    type: str = Field(..., min_length=1)  # e.g., "hotspot"
    location: Dict[str, Any] = Field(...)  # {"x": float, "y": float, "bbox": list[float]}
    temperature: float = Field(..., ge=0.0)
    severity: str = Field(..., pattern=r"^(low|medium|high)$")


class VideoMetadata(BaseModel):
    video_id: str = Field(..., min_length=1)
    equipment_type: str = Field(default="motor", min_length=1)
    equipment_id: str = Field(..., min_length=1)
    upload_timestamp: datetime
    duration_seconds: float = Field(..., ge=0.0)
    frame_rate: int = Field(..., ge=1)
    resolution: Resolution
    temperature_range: TemperatureRange
    anomalies_detected: List[Anomaly] = Field(default_factory=list)
    file_path: str = Field(..., min_length=1)


class SensorData(BaseModel):
    temperature: float = Field(..., ge=-100.0)
    vibration: float = Field(..., ge=0.0)
    rpm: Optional[float] = Field(None, ge=0.0)


class FailurePrediction(BaseModel):
    timeline_days: int = Field(..., ge=0)
    severity: str = Field(..., pattern=r"^(low|medium|high)$")
    factors: List[str] = Field(default_factory=list)


class Remediation(BaseModel):
    action: str = Field(..., min_length=1)
    urgency: str = Field(..., pattern=r"^(low|medium|high)$")
    cost_estimate: float = Field(..., ge=0.0)
    priority_score: float = Field(..., ge=0.0, le=1.0)


class LogData(BaseModel):
    log_id: str = Field(..., min_length=1)
    equipment_id: str = Field(..., min_length=1)
    timestamp: datetime
    sensor_data: SensorData
    anomaly_summary: str = Field(..., min_length=1)
    historical_analysis: Optional[str] = None
    failure_prediction: Optional[FailurePrediction] = None
    remediation: List[Remediation] = Field(default_factory=list)


class PerformanceMetrics(BaseModel):
    efficiency: float = Field(..., ge=0.0, le=100.0)
    trend: str = Field(..., min_length=1)  # e.g., "Stable"


class Predictions(BaseModel):
    potential_failure: bool
    estimated_timeline: str = Field(..., min_length=1)
    contributing_factors: List[str] = Field(default_factory=list)


class RagSources(BaseModel):
    manuals: List[str] = Field(default_factory=list)
    historical_logs: List[str] = Field(default_factory=list)


class Insight(BaseModel):
    insight_id: str = Field(..., min_length=1)
    equipment_id: str = Field(..., min_length=1)
    analysis_timestamp: datetime
    health_score: float = Field(..., ge=0.0, le=100.0)
    risk_assessment: str = Field(..., min_length=1)
    performance_metrics: PerformanceMetrics
    predictions: Predictions
    recommendations: List[Remediation] = Field(default_factory=list)
    rag_sources: RagSources
    ethical_notes: Optional[str] = None


class ManualMetadata(BaseModel):
    version: str = Field(..., min_length=1)
    last_updated: datetime


class ManualChunk(BaseModel):
    manual_id: str = Field(..., min_length=1)
    equipment_type: str = Field(..., min_length=1)
    section_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    keywords: List[str] = Field(default_factory=list)
    metadata: ManualMetadata