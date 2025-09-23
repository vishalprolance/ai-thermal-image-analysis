import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gradio as gr
import json
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import os
from typing import Dict, Any, Tuple, Optional
from utils import Insight, PerformanceMetrics, Predictions, RagSources, Remediation
from agent.core import analyze_input
from ml.models import detect_anomalies
from rag.rag import RAGManager
import requests  # For API calls
import shutil


rag_manager = RAGManager()


def auth_fn(username: str, password: str) -> Tuple[str, str]:
    """Simple auth for roles: admin/adminpass, user/userpass"""
    if username == "admin" and password == "adminpass":
        return (username, "Admin access granted")
    elif username == "user" and password == "userpass":
        return (username, "User access granted")
    else:
        return None


def generate_pdf(insight: Insight, filename: str = "insight_report.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750
    c.drawString(100, y, f"Thermal AI Insight Report")
    y -= 20
    c.drawString(100, y, f"Equipment ID: {insight.equipment_id}")
    y -= 20
    c.drawString(100, y, f"Health Score: {insight.health_score}/100")
    y -= 20
    c.drawString(100, y, f"Risk Assessment: {insight.risk_assessment}")
    y -= 20
    c.drawString(100, y, f"Trend: {insight.performance_metrics.trend}")
    y -= 20
    c.drawString(100, y, f"Potential Failure: {insight.predictions.potential_failure}")
    y -= 20
    c.drawString(100, y, f"Estimated Timeline: {insight.predictions.estimated_timeline}")
    y -= 20
    c.drawString(100, y, "Recommendations:")
    for rec in insight.recommendations:
        y -= 20
        c.drawString(120, y, f"- {rec.action} (Urgency: {rec.urgency}, Cost: ${rec.cost_estimate})")
    y -= 20
    c.drawString(100, y, f"Ethical Notes: {insight.ethical_notes or 'N/A'}")
    c.save()
    return filename


def send_alert_mock(insight: Insight):
    if insight.health_score < 50 or insight.predictions.potential_failure:
        return f"ALERT: High risk for {insight.equipment_id}! Health: {insight.health_score}. Check dashboard."
    return "No alert triggered."


def load_insights():
    insights = []
    if os.path.exists("outputs/insights"):
        for file in os.listdir("outputs/insights"):
            if file.endswith(".json"):
                with open(f"outputs/insights/{file}", "r") as f:
                    data = json.load(f)
                    insights.append(Insight(**data))
    return insights


def analyze_video(video_file):
    if video_file is None:
        return "No video uploaded.", None, None
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    anomalies = detect_anomalies(video_path)
    input_data = {"video_path": video_path, "equipment_id": "MOTOR-001"}  # Mock ID
    insight = analyze_input(input_data)
    alert = send_alert_mock(insight)
    pdf = generate_pdf(insight)
    return f"Anomalies detected: {len(anomalies)}", insight.dict(), alert, pdf


def analyze_log(log_json):
    try:
        log_data = json.loads(log_json)
        input_data = {"log_data": log_data}
        insight = analyze_input(input_data)
        alert = send_alert_mock(insight)
        pdf = generate_pdf(insight)
        return insight.dict(), alert, pdf
    except Exception as e:
        return {"error": str(e)}, "Error", None


def ingest_manual_pdf(pdf_file):
    if pdf_file is None:
        return "No PDF uploaded.", "Error"
    manual_path = "temp_manual.pdf"
    with open(manual_path, "wb") as f:
        f.write(pdf_file.read())
    try:
        rag_manager.setup_manual_db([manual_path])
        return "Manual ingested successfully.", "Success"
    except Exception as e:
        return f"Ingestion failed: {str(e)}", "Error"


def ingest_log_json(log_json):
    try:
        log_data_list = json.loads(log_json)
        os.makedirs("data/logs", exist_ok=True)
        log_files = []
        for i, log_data in enumerate(log_data_list):
            log_id = f"log_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            file_path = f"data/logs/{log_id}.json"
            with open(file_path, "w") as f:
                json.dump(log_data, f)
            log_files.append(file_path)
        rag_manager.ingest_new_logs(log_files)
        return "Logs ingested successfully.", "Success"
    except Exception as e:
        return f"Ingestion failed: {str(e)}", "Error"


def plot_health_trend(insights):
    if not insights:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    df_data = [{"Equipment": i.equipment_id, "Health": i.health_score, "Time": i.analysis_timestamp} for i in insights]
    fig = px.line(df_data, x="Time", y="Health", color="Equipment", title="Health Trend Over Time")
    return fig


def plot_predictions(insight):
    if not insight.predictions.contributing_factors:
        fig = go.Figure()
        fig.add_annotation(text="No predictions", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    fig = px.pie(values=[1]*len(insight.predictions.contributing_factors), names=insight.predictions.contributing_factors, title="Contributing Factors")
    return fig


def update_admin_tabs(role):
    if role == "admin":
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


with gr.Blocks(title="Thermal AI Dashboard") as demo:
    gr.Markdown("# Thermal AI Dashboard for Manufacturing Equipment Analysis")
    
    role = gr.State()  # Store role after auth
    
    with gr.Tab("Overview"):
        gr.Markdown("### Equipment Health Overview")
        insights = load_insights()
        trend_chart = gr.Plot(label="Health Trend", value=plot_health_trend(insights))
        refresh_btn = gr.Button("Refresh Insights")
        refresh_btn.click(fn=lambda: plot_health_trend(load_insights()), outputs=trend_chart)
    
    with gr.Tab("Video Analysis"):
        gr.Markdown("### Upload Thermal Video for Anomaly Detection")
        video_upload = gr.File(label="Upload MP4 Video", file_types=[".mp4"])
        anomalies_output = gr.Textbox(label="Anomalies Summary")
        insight_output = gr.JSON(label="Full Insight")
        alert_output = gr.Textbox(label="Alert")
        pdf_download = gr.File(label="Download PDF Report")
        analyze_video_btn = gr.Button("Analyze Video")
        analyze_video_btn.click(
            analyze_video,
            inputs=video_upload,
            outputs=[anomalies_output, insight_output, alert_output, pdf_download]
        )
    
    with gr.Tab("Log Analysis"):
        gr.Markdown("### Analyze Log Data")
        log_input = gr.Textbox(label="JSON Log Data (list of logs)", lines=10)
        log_insight = gr.JSON(label="Insight")
        log_alert = gr.Textbox(label="Alert")
        log_pdf = gr.File(label="Download PDF Report")
        analyze_log_btn = gr.Button("Analyze Logs")
        analyze_log_btn.click(
            analyze_log,
            inputs=log_input,
            outputs=[log_insight, log_alert, log_pdf]
        )
    
    with gr.Tab("Predictions"):
        gr.Markdown("### Failure Predictions and Recommendations")
        pred_chart = gr.Plot(label="Factors Pie Chart")
        recs_text = gr.Textbox(label="Recommendations")
        load_btn = gr.Button("Load Sample Predictions")
        load_btn.click(
            fn=lambda: (
                plot_predictions(Insight(
                    insight_id="temp",
                    equipment_id="temp",
                    analysis_timestamp=datetime.now(),
                    health_score=50,
                    risk_assessment="low",
                    performance_metrics=PerformanceMetrics(efficiency=50, trend="Stable"),
                    predictions=Predictions(potential_failure=True, estimated_timeline="15 days", contributing_factors=["hotspot", "vibration"]),
                    recommendations=[Remediation(action="Inspect", urgency="high", cost_estimate=100, priority_score=0.9)],
                    rag_sources=RagSources(manuals=[], historical_logs=[]),
                    ethical_notes=""
                )),
                " - Inspect bearings (high urgency, $100)\n - Lubricate (medium, $50)"
            ),
            outputs=[pred_chart, recs_text]
        )
    
    with gr.Tab("RAG Query"):
        gr.Markdown("### Query Manuals and Historical Logs")
        query_input = gr.Textbox(label="Query (e.g., 'bearing hotspot remediation')")
        context_output = gr.Textbox(label="Retrieved Context", lines=10)
        sources_output = gr.JSON(label="Sources")
        query_btn = gr.Button("Query RAG")
        query_btn.click(
            fn=lambda q: rag_manager.retrieve_context(q),
            inputs=query_input,
            outputs=[context_output, sources_output]
        )
    
    # Admin tabs
    admin_manual_tab = gr.Tab("Admin: Upload Manuals")
    with admin_manual_tab:
        gr.Markdown("### Upload Equipment Manuals (PDF)")
        manual_upload = gr.File(label="Upload PDF Manual", file_types=[".pdf"])
        manual_status = gr.Textbox(label="Ingestion Status")
        manual_btn = gr.Button("Ingest Manual")
        manual_btn.click(
            ingest_manual_pdf,
            inputs=manual_upload,
            outputs=manual_status
        )
    
    admin_log_tab = gr.Tab("Admin: Upload Historical Logs")
    with admin_log_tab:
        gr.Markdown("### Upload Historical Logs (JSON)")
        log_upload_json = gr.Textbox(label="JSON Logs (list of log objects)", lines=10)
        log_status = gr.Textbox(label="Ingestion Status")
        log_btn = gr.Button("Ingest Logs")
        log_btn.click(
            ingest_log_json,
            inputs=log_upload_json,
            outputs=log_status
        )
    
    # Role update (simulate after auth; in prod, use gr.Interface with auth and state update)
    gr.Markdown("**Admin features (upload manuals/logs) are now visible. For full auth, implement custom login.**")
    # Note: Gradio auth doesn't directly update UI; for full RBAC, use custom login interface or session state in a more advanced setup.


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)