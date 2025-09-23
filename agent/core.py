from langchain.agents import initialize_agent, AgentType
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory
from .ollama_adapter import OllamaAdapter
from typing import Dict, Any, List
import os
import json
from datetime import datetime
from ml.models import detect_anomalies, forecast_failure
from rag.rag import RAGManager
from utils.models import Insight, LogData, VideoMetadata, FailurePrediction, Remediation, PerformanceMetrics, Predictions, RagSources


# Initialize RAG
rag_manager = RAGManager()


def detect_anomalies_func(video_path: str) -> str:
    """Detect thermal anomalies in a video file for motors/bearings. Returns JSON list of anomalies."""
    anomalies = detect_anomalies(video_path)
    return json.dumps([a.dict() for a in anomalies])


def forecast_failure_func(log_data_json: str) -> str:
    """Forecast failure from historical or IoT log data. Input JSON string of list of logs. Returns JSON prediction."""
    log_data = json.loads(log_data_json)
    prediction = forecast_failure(log_data)
    return json.dumps(prediction.dict())


def retrieve_rag_context_func(query: str) -> str:
    """Retrieve context from equipment manuals and historical logs using RAG. Returns context and sources JSON."""
    context, sources = rag_manager.retrieve_context(query)
    return json.dumps({"context": context, "sources": sources})


detect_anomalies_tool = StructuredTool.from_function(
    func=detect_anomalies_func,
    name="detect_anomalies_tool",
    description="Detect thermal anomalies in a video file for motors/bearings. Input: video_path str. Output: JSON list of anomalies."
)

forecast_failure_tool = StructuredTool.from_function(
    func=forecast_failure_func,
    name="forecast_failure_tool",
    description="Forecast failure from historical or IoT log data. Input: JSON string of list of logs. Output: JSON prediction."
)

retrieve_rag_context_tool = StructuredTool.from_function(
    func=retrieve_rag_context_func,
    name="retrieve_rag_context_tool",
    description="Retrieve context from equipment manuals and historical logs using RAG. Input: query str. Output: JSON with context and sources."
)

tools = [detect_anomalies_tool, forecast_failure_tool, retrieve_rag_context_tool]


# LLM Setup: use a local OllamaAdapter to avoid external API dependencies.
try:
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma")
    ollama_base = os.getenv("OLLAMA_BASE_URL")
    llm_adapter = OllamaAdapter(model=ollama_model, base_url=ollama_base)
except Exception as e:
    # Fail fast with a clear message so devs know to install/configure Ollama
    raise RuntimeError(
        "Failed to initialize local Ollama adapter.\n"
        "Ensure the 'ollama' Python client is installed and the Ollama daemon is running locally.\n"
        f"Underlying error: {e}"
    )


# Minimal agent executor wrapper to mimic the previous agent_executor.invoke
class SimpleAgentExecutor:
    def __init__(self, adapter: OllamaAdapter):
        self.adapter = adapter

    def invoke(self, payload: dict) -> dict:
        # Expect payload like {"input": prompt}
        prompt = payload.get("input") if isinstance(payload, dict) else str(payload)
        if not prompt:
            return {"output": ""}
        text = self.adapter.generate(prompt)
        return {"output": text}


llm = llm_adapter
agent_executor = SimpleAgentExecutor(llm_adapter)


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
try:
    # Keep the LangChain-based agent initialization if LangChain is present
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    agent_executor = agent
except Exception:
    # If LangChain agent can't be initialized (missing/compatibility), keep the simple adapter executor.
    agent_executor = agent_executor


def analyze_input(input_data: Dict[str, Any]) -> Insight:
    """Main function to run agent on input (video or log). Returns Insight object."""
    if "video_path" in input_data:
        prompt_input = f"Analyze video for equipment {input_data.get('equipment_id', 'unknown')}: {input_data['video_path']}"
    elif "log_data" in input_data:
        prompt_input = f"Analyze log data for equipment {input_data['log_data'][0]['equipment_id'] if input_data['log_data'] else 'unknown'}: {len(input_data['log_data'])} entries."
    else:
        raise ValueError("Input must contain 'video_path' or 'log_data'.")
    
    result = agent_executor.invoke({"input": prompt_input})
    
    # Parse output JSON
    try:
        output_json = json.loads(result['output'])
    except:
        output_json = {"error": "Parsing failed", "raw": result['output']}
    
    # Construct Insight
    insight = Insight(
        insight_id=output_json.get("insight_id", "temp_id"),
        equipment_id=input_data.get("equipment_id", "unknown"),
        analysis_timestamp=datetime.now(),
        health_score=output_json.get("health_score", 50.0),
        risk_assessment=output_json.get("risk_assessment", "Unknown"),
        performance_metrics=PerformanceMetrics(
            efficiency=output_json.get("performance_metrics", {}).get("efficiency", 50.0),
            trend=output_json.get("performance_metrics", {}).get("trend", "Stable")
        ),
        predictions=Predictions(
            potential_failure=output_json.get("predictions", {}).get("potential_failure", False),
            estimated_timeline=output_json.get("predictions", {}).get("estimated_timeline", "Unknown"),
            contributing_factors=output_json.get("predictions", {}).get("contributing_factors", [])
        ),
        recommendations=[
            Remediation(**rec) for rec in output_json.get("recommendations", [])
        ],
        rag_sources=RagSources(
            manuals=output_json.get("rag_sources", {}).get("manuals", []),
            historical_logs=output_json.get("rag_sources", {}).get("historical_logs", [])
        ),
        ethical_notes=output_json.get("ethical_notes", "")
    )
    
    return insight


if __name__ == "__main__":
    # Example usage
    sample_log = [{"equipment_id": "MOTOR-001", "sensor_data": {"temperature": 85.0, "vibration": 2.5, "rpm": 1500}, "anomaly_summary": "Hotspot detected", "timestamp": datetime.now().isoformat()}]
    insight = analyze_input({"log_data": sample_log})
    print(insight.dict())