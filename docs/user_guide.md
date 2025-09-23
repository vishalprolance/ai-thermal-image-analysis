# Thermal AI User Guide

## Overview
This system provides GenAI-powered analysis of thermal imaging videos and IoT sensor data for manufacturing equipment (motors and bearings). It detects anomalies, forecasts failures, retrieves context from manuals and historical logs via RAG, and generates insights/recommendations.

## Installation and Setup
1. Clone the repository (if applicable).
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set environment variables:
   - `GROQ_API_KEY`: For LangChain LLM (sign up at groq.com).
   - AWS credentials for IoT (optional for local).
4. Create directories if not present (already done).
5. Ingest manuals (place PDFs in `data/manuals/`):
   ```
   python rag/ingest_manuals.py data/manuals/*.pdf
   ```
   (Create ingest_manuals.py if needed: Use RAGManager.setup_manual_db).
6. Start services with Docker:
   ```
   docker-compose up -d
   ```
   - API: http://localhost:8000/docs (Swagger)
   - Dashboard: http://localhost:7860
   - MQTT Broker: localhost:1883 (for IoT mocks)

## Usage

### 1. Video Analysis
- **API**: POST /upload/video
  ```
  curl -X POST "http://localhost:8000/upload/video" -F "file=@path/to/video.mp4" -F 'metadata={"equipment_id": "MOTOR-001", "equipment_type": "motor"}'
  ```
  Returns Insight JSON with anomalies, predictions, recs.
- **Dashboard**: Upload video in "Video Analysis" tab; view anomalies, insight, PDF report, alert.

### 2. Log Analysis
- **API**: POST /upload/log (JSON body: list of logs)
  ```
  curl -X POST "http://localhost:8000/upload/log" -H "Content-Type: application/json" -d '[{"equipment_id": "MOTOR-001", "timestamp": "2023-01-01T00:00:00", "sensor_data": {"temperature": 85.0, "vibration": 2.5, "rpm": 1500}, "anomaly_summary": "Hotspot"}]'
  ```
  Ingests to RAG, analyzes with historical context.
- **Dashboard**: Paste JSON in "Log Analysis" tab; get insight, alert, PDF.

### 3. IoT Real-Time Monitoring
- Local Mock:
  ```
  python iot/mqtt_client.py
  ```
  Publishes mock sensor data to localhost:1883; triggers analysis every 24 messages.
- AWS Integration: Set AWS env vars, call `client.start_aws(equipment_id="MOTOR-001")`.
  - For AWS IoT Core: Use certificates for secure MQTT.
  - For AWS MQ: Replace with Stomp/AMQP client (e.g., rabbitmq).

### 4. RAG Query
- Dashboard "RAG Query" tab: Enter query (e.g., "bearing maintenance"), get context from manuals/logs.

### 5. Insights and Outputs
- GET /insights/{id}: Retrieve saved insight.
- Dashboard "Overview": Health trends chart.
- "Predictions": Pie chart of factors, prioritized recs.
- Alerts: Auto-sent for low health/high risk (mock email/Slack; configure in ethics.py).

### 6. Ethics and Validation
- Run validation:
  ```
  python utils/ethics.py
  ```
  Checks ML accuracy, agent structure.
- Middleware auto-anonymizes PII in uploads.
- Human review flagged for low-confidence.

## Troubleshooting
- **No GPU for ML**: Set CPU_ONLY=1 env var.
- **Chroma DB errors**: Ensure vector_dbs dir writable.
- **LangChain errors**: Check API key; fallback to local LLM (Ollama).
- **MQTT connection**: Install Mosquitto or use Docker broker.
- **Low accuracy**: Retrain ML with real data; run validate_ml_models().

## Deployment Notes
- Cloud: Use ECS/EKS for Docker; S3 for data; RDS for metadata.
- Scale: Add Celery for batch jobs; Ray for ML.
- Security: Add JWT auth to API; encrypt volumes.

For issues, check logs or run `pytest tests/validate.py`.