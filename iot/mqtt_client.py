from typing import Optional
import paho.mqtt.client as mqtt
import json
from datetime import datetime
from typing import List, Dict, Any
from utils.models import LogData, SensorData
from agent.core import analyze_input
import os
import threading
import time
from boto3 import client as boto3_client
from botocore.exceptions import ClientError


class IotClient:
    def __init__(self, broker="localhost", port=1883, aws_region=None, aws_endpoint=None, aws_access_key=None, aws_secret_key=None):
        self.broker = broker
        self.port = port
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.buffer: List[Dict[str, Any]] = []  # Rolling buffer for 24 hours (mock)
        self.lock = threading.Lock()
        self.is_aws = bool(aws_region)
        if self.is_aws:
            self.aws_client = boto3_client(
                'iot-data', 
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                endpoint_url=aws_endpoint
            )
            # For AWS MQ, use MQTT over WebSocket or TLS; this is simplified for IoT Core
        self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT broker")
            self.connected = True
            client.subscribe("/equipment/+/sensors")  # Wildcard for all equipment
        else:
            print(f"Connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        equipment_id = topic.split('/')[-1]
        data = json.loads(msg.payload.decode())
        data["timestamp"] = datetime.now().isoformat()
        data["equipment_id"] = equipment_id
        data["anomaly_summary"] = "Real-time sensor reading"  # Temp

        with self.lock:
            self.buffer.append(data)
            if len(self.buffer) > 24:  # Keep last 24 entries (hourly mock)
                self.buffer.pop(0)

        # Trigger analysis every 24 entries (mock 1 hour)
        if len(self.buffer) % 24 == 0:
            self.trigger_analysis(equipment_id)

    def trigger_analysis(self, equipment_id: str):
        with self.lock:
            log_data = self.buffer.copy()
        input_data = {"log_data": log_data}
        insight = analyze_input(input_data)
        print(f"Analysis triggered for {equipment_id}: Health {insight.health_score}")

    def start_local(self):
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()

    def publish_mock(self, equipment_id: str, data: Dict[str, Any]):
        topic = f"/equipment/{equipment_id}/sensors"
        payload = json.dumps(data)
        self.client.publish(topic, payload)

    def start_aws(self, equipment_id: str = "MOTOR-001", data: Optional[Dict[str, Any]] = None):
        # For AWS IoT Core, use AWS IoT SDK or paho with certs; simplified boto3 for data plane
        # Note: For full AWS MQ (RabbitMQ/AMQP), use different client; this is for IoT Core MQTT
        print("AWS IoT integration: Subscribe via AWS IoT Core rules or SDK.")
        if data is None:
            data = {"temperature": 75.0, "vibration": 1.2, "rpm": 1500}
        # Example publish to AWS IoT topic
        try:
            self.aws_client.publish(
                topic=f"equipment/{equipment_id}/sensors",
                payload=json.dumps(data)
            )
            print("Published to AWS IoT")
        except ClientError as e:
            print(f"AWS error: {e}")

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


# Mock script for testing
if __name__ == "__main__":
    # Local mock
    client = IotClient(broker="localhost", port=1883)
    client.start_local()
    
    # Publish mock data
    for i in range(5):
        mock_data = {"temperature": 70.0 + i, "vibration": 1.5, "rpm": 1500}
        client.publish_mock("MOTOR-001", mock_data)
        time.sleep(1)
    
    time.sleep(10)
    client.stop()
    
    # AWS example (set env vars)
    # client = IotClient(aws_region="us-east-1", aws_endpoint="your-iot-endpoint.iot.us-east-1.amazonaws.com", aws_access_key="key", aws_secret_key="secret")
    # client.start_aws()