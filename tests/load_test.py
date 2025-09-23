from locust import HttpUser, task, between
import json


class ApiUser(HttpUser):
    wait_time = between(1, 3)

    @task(2)
    def upload_video_mock(self):
        # Mock video upload
        files = {'file': ('video.mp4', open('data/videos/sample.mp4', 'rb') if os.path.exists('data/videos/sample.mp4') else b'')}
        data = {'metadata': json.dumps({"equipment_id": "MOTOR-001", "equipment_type": "motor"})}
        self.client.post("/upload/video", files=files, data=data)

    @task(3)
    def upload_log_mock(self):
        log_data = json.dumps([{"equipment_id": "MOTOR-001", "timestamp": "2023-01-01T00:00:00", "sensor_data": {"temperature": 85.0, "vibration": 2.5, "rpm": 1500}, "anomaly_summary": "Hotspot"}])
        self.client.post("/upload/log", json=log_data)

    @task(1)
    def get_insight(self):
        self.client.get("/insights/test_id")

    @task(1)
    def health_check(self):
        self.client.get("/health")


# Run with: locust -f tests/load_test.py --host=http://localhost:8000 --users=50 --spawn-rate=5