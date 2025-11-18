from pydantic import BaseModel, Field
from typing import Optional


class GenerateVideoRequest(BaseModel):
    """Video generation request model"""
    text: str = Field(..., description="Text content for lip synchronization")
    audio_file: str = Field(default="voice/voice.mp3", description="Path to the audio file")
    gender: int = Field(default=1, description="Gender of the speaker (1 for male, 0 for female)")
    char_interval: float = Field(default=0.5, description="Duration per character in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "你好，这是一个口型同步测试",
                "audio_file": "voice/voice.mp3",
                "gender": 1,
                "char_interval": 0.5
            }
        }


class GenerateVideoResponse(BaseModel):
    """Video generation response model"""
    success: bool = Field(..., description="Whether the video generation was successful")
    video_id: str = Field(..., description="Unique video identifier")
    video_url: str = Field(..., description="URL to access the generated video")
    message: str = Field(..., description="Response message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "video_id": "a1b2c3d4e5f6g7h8i9j0",
                "video_url": "https://example.com/api/v1/vitual/videos/a1b2c3d4e5f6g7h8i9j0.mp4",
                "message": "视频生成成功"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "message": "口型视频生成服务运行正常"
            }
        }