import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
import os
import json
from datetime import datetime
import cv2
from utils import VideoMetadata, LogData, Insight, Resolution, TemperatureRange
from agent.core import analyze_input
from rag.rag import RAGManager
from ml.models import detect_anomalies
import shutil


app = FastAPI(title="Thermal AI API", version="1.0.0", description="API for thermal imaging analysis in manufacturing equipment.")


rag_manager = RAGManager()


class VideoUpload(BaseModel):
    equipment_id: str
    equipment_type: Optional[str] = "motor"


class LogUpload(BaseModel):
    log_data: List[LogData]


@app.post("/upload/video", response_model=Insight)
async def upload_video(file: UploadFile = File(...), metadata: VideoUpload = Form(...)):
    if file.filename is None or not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only MP4 videos supported")
    
    video_id = str(uuid.uuid4())
    file_path = f"data/videos/{video_id}.mp4"
    os.makedirs("data/videos", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract metadata
    cap = cv2.VideoCapture(file_path)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Mock temperature range
    temp_range = {"min": 20.0, "max": 100.0}
    
    # Detect anomalies
    anomalies = detect_anomalies(file_path)
    
    video_meta = VideoMetadata(
        video_id=video_id,
        equipment_type=metadata.equipment_type or "motor",
        equipment_id=metadata.equipment_id,
        upload_timestamp=datetime.now(),
        duration_seconds=duration,
        frame_rate=int(frame_rate),
        resolution=Resolution(width=width, height=height),
        temperature_range=TemperatureRange(min=temp_range["min"], max=temp_range["max"]),
        anomalies_detected=anomalies,
        file_path=file_path
    )
    
    # Analyze with agent
    input_data = {"video_path": file_path, "equipment_id": metadata.equipment_id}
    insight = analyze_input(input_data)
    
    # Save insight
    os.makedirs("outputs/insights", exist_ok=True)
    with open(f"outputs/insights/{video_id}.json", "w") as f:
        json.dump(insight.dict(), f, default=str)
    
    return insight


@app.post("/upload/log", response_model=Insight)
async def upload_log(log_json: str):
    try:
        log_data_list = json.loads(log_json)
        logs = [LogData(**log) for log in log_data_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid log JSON: {str(e)}")
    
    if not logs:
        raise HTTPException(status_code=400, detail="No logs provided")
    
    equipment_id = logs[0].equipment_id
    
    # Save logs
    os.makedirs("data/logs", exist_ok=True)
    for log in logs:
        log_id = log.log_id or str(uuid.uuid4())
        log.log_id = log_id
        with open(f"data/logs/{log_id}.json", "w") as f:
            json.dump(log.dict(), f, default=str)
    
    # Ingest to RAG
    rag_manager.ingest_new_logs([f"data/logs/{log.log_id}.json" for log in logs])
    
    # Analyze with agent
    input_data = {"log_data": [log.dict() for log in logs]}
    insight = analyze_input(input_data)
    
    # Save insight
    os.makedirs("outputs/insights", exist_ok=True)
    insight_id = insight.insight_id or str(uuid.uuid4())
    insight.insight_id = insight_id
    with open(f"outputs/insights/{insight_id}.json", "w") as f:
        json.dump(insight.dict(), f, default=str)
    
    return insight


@app.get("/insights/{insight_id}", response_model=Insight)
async def get_insight(insight_id: str):
    file_path = f"outputs/insights/{insight_id}.json"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Insight not found")
    
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return Insight(**data)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)