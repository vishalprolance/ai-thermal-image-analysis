from typing import List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import numpy as np
from typing import List, Dict, Any
from utils.models import Anomaly, FailurePrediction
import json
from datetime import datetime


class ThermalFrameDataset(Dataset):
    def __init__(self, frames: List[np.ndarray], labels: Optional[List[int]] = None):
        self.frames = frames
        self.labels = labels or [0] * len(frames)  # 0 for normal
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame_tensor = self.transform(frame)
        label = self.labels[idx]
        return frame_tensor, label


class ThermalAutoencoder(nn.Module):
    def __init__(self):
        super(ThermalAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(model: ThermalAutoencoder, dataloader: DataLoader, epochs: int = 50, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for frames, _ in dataloader:
            frames = frames.to(device)
            recon = model(frames)
            loss = criterion(recon, frames)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "ml/models/anomaly_detector.pth")
    return model


def detect_anomalies(video_path: str, threshold: float = 0.05) -> List[Anomaly]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalAutoencoder()
    model.load_state_dict(torch.load("ml/models/anomaly_detector.pth", map_location=device))
    model.to(device)
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    anomalies = []
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224)).astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            recon = model(frame_tensor)
            mse = torch.mean((frame_tensor - recon) ** 2).item()
        
        if mse > threshold:
            # Mock location and temp (in prod, use calibration)
            anomaly = Anomaly(
                frame_number=frame_num,
                timestamp=datetime.now(),
                type="hotspot",
                location={"x": 100.0, "y": 100.0, "bbox": [90.0, 90.0, 110.0, 110.0]},
                temperature=85.0,  # Mock
                severity="high" if mse > 0.1 else "medium"
            )
            anomalies.append(anomaly)
        
        frame_num += 1
    
    cap.release()
    return anomalies


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], targets: List[np.ndarray]):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 128, num_layers: int = 2, output_steps: int = 7):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_steps * input_size)
        self.failure_head = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        forecast = self.fc(last_hidden)
        failure_prob = self.sigmoid(self.failure_head(last_hidden))
        return forecast, failure_prob


def train_lstm(model: LSTMForecaster, dataloader: DataLoader, epochs: int = 100, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        total_mse = 0
        total_bce = 0
        for seq, target in dataloader:
            seq, target = seq.to(device), target.to(device)
            forecast, prob = model(seq)
            mse = criterion_mse(forecast, target[:, :forecast.shape[1]])
            bce = criterion_bce(prob.squeeze(), target[:, -1, 0])  # Mock failure label
            loss = mse + bce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mse += mse.item()
            total_bce += bce.item()
        print(f"Epoch [{epoch+1}/{epochs}], MSE: {total_mse / len(dataloader):.4f}, BCE: {total_bce / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), "ml/models/lstm_forecaster.pth")
    return model


def forecast_failure(log_data: List[Dict[str, Any]]) -> FailurePrediction:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster()
    model.load_state_dict(torch.load("ml/models/lstm_forecaster.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # Prepare sequence (last 24 entries, features: temp, vibration, rpm)
    recent_logs = log_data[-24:] if len(log_data) >= 24 else log_data
    sequence = np.array([[lg['sensor_data']['temperature'], lg['sensor_data']['vibration'], lg['sensor_data'].get('rpm', 0)] for lg in recent_logs])
    seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, prob = model(seq_tensor)
    
    prob_val = prob.item()
    severity = "high" if prob_val > 0.7 else "medium" if prob_val > 0.3 else "low"
    timeline = 15 if severity == "high" else 30  # Mock
    
    return FailurePrediction(
        timeline_days=timeline,
        severity=severity,
        factors=["rising temperature", "high vibration"]  # Mock
    )