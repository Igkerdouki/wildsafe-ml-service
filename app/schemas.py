from pydantic import BaseModel, Field
from typing import Optional


class SpeciesPrediction(BaseModel):
    species: str = Field(description="Species name")
    confidence: float = Field(description="Confidence score 0.0-1.0")


class PoseAnalysis(BaseModel):
    """Pose estimation result for person behavior detection."""
    state: str = Field(description="Person state: normal, fallen, distress, no_person")
    pose_behavior: Optional[str] = Field(
        default=None,
        description="Detected pose: standing, sitting, lying_down, hunched"
    )
    body_angle: Optional[float] = Field(
        default=None,
        description="Torso angle from vertical in degrees"
    )
    confidence: float = Field(description="Pose detection confidence")


class PredictionResponse(BaseModel):
    predicted_species: str = Field(description="Top predicted species")
    confidence: float = Field(description="Confidence score for top prediction")
    alert: bool = Field(default=False, description="Whether to trigger alert")
    speaker_frequency_hz: int = Field(default=0, description="Speaker frequency for deterrence")
    all_predictions: list[SpeciesPrediction] = Field(
        default_factory=list,
        description="Top 5 species predictions with confidence scores"
    )
    frame_width: int = Field(description="Input frame width")
    frame_height: int = Field(description="Input frame height")
    inference_time_ms: float = Field(description="Inference time in milliseconds")
    pose_analysis: Optional[PoseAnalysis] = Field(
        default=None,
        description="Pose analysis for person behavior detection"
    )


class VideoFrameResult(BaseModel):
    frame_number: int
    timestamp_ms: float
    predicted_species: str
    confidence: float


class VideoPredictionResponse(BaseModel):
    predicted_species: str = Field(description="Overall predicted species for video")
    confidence: float = Field(description="Average confidence for predicted species")
    top3_species: list[str] = Field(description="Top 3 species by weighted vote")
    vote_counts: dict[str, int] = Field(description="Vote counts per species")
    total_frames: int
    frames_processed: int
    fps: float
    processing_fps: float
    frames: list[VideoFrameResult]


class Base64ImageRequest(BaseModel):
    image: str = Field(description="Base64-encoded image data")
    confidence_threshold: float = Field(default=0.1, ge=0.0, le=1.0)


class URLImageRequest(BaseModel):
    url: str = Field(description="URL to fetch image from")
    confidence_threshold: float = Field(default=0.1, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_type: str = Field(default="zero-shot-classifier")
    target_classes: list[str]
    accuracy: str = Field(default="100% (test set)")
