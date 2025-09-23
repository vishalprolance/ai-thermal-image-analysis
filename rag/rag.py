import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from langchain_ollama import OllamaEmbeddings
except Exception:
    # Try a sentence-transformers based fallback
    from typing import Iterable, List
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        class OllamaEmbeddings:
            """Fallback embedding wrapper using sentence-transformers.

            Implements embed_documents and embed_query used by this project.
            """
            def __init__(self, model: str = "all-MiniLM-L6-v2"):
                model_map = {"embeddinggemma:latest": "all-MiniLM-L6-v2"}
                chosen = model_map.get(model, model)
                self._model = SentenceTransformer(chosen)

            def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
                arr = self._model.encode(list(texts), show_progress_bar=False)
                return [list(map(float, v)) for v in arr]

            def embed_query(self, text: str) -> List[float]:
                vec = self._model.encode([text], show_progress_bar=False)[0]
                return list(map(float, vec))
    except Exception:
        # Final fallback: simple TF-IDF / HashingVectorizer based embeddings
        # This avoids heavy ML deps like huggingface_hub when unavailable.
        from sklearn.feature_extraction.text import HashingVectorizer
        import numpy as np

        class OllamaEmbeddings:
            """Deterministic simple embedding fallback using HashingVectorizer.

            Produces fixed-size vectors (sparse->dense) suitable for similarity
            search when real embeddings are unavailable.
            """
            def __init__(self, model: str = "hashing-fallback"):
                # 2**12 dims ~ 4096
                self._vec = HashingVectorizer(n_features=4096, alternate_sign=False, norm=None)

            def _to_dense(self, arr):
                # HashingVectorizer returns sparse matrix
                dense = arr.toarray()
                return [list(map(float, row)) for row in dense]

            def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
                X = self._vec.transform(list(texts))
                return self._to_dense(X)

            def embed_query(self, text: str) -> List[float]:
                X = self._vec.transform([text])
                return self._to_dense(X)[0]
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Tuple
from utils.models import ManualChunk, LogData
import os
import json
from datetime import datetime


embeddings = OllamaEmbeddings(model="embeddinggemma:latest")


class RAGManager:
    def __init__(self, persist_directory: str = "vector_dbs"):
        self.persist_directory = persist_directory
        self.manual_collection = "manuals"
        self.logs_collection = "historical_logs"
        self.manual_db = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.manual_collection,
            embedding_function=embeddings
        )
        self.logs_db = Chroma(
            persist_directory=self.persist_directory,
            collection_name=self.logs_collection,
            embedding_function=embeddings
        )


    def setup_manual_db(self, manual_paths: List[str]):
        docs = []
        for path in manual_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        
        # Add metadata
        for doc in splits:
            doc.metadata.update({
                "equipment_type": "motor",  # Default; can be extracted
                "source": doc.metadata.get("source", "")
            })
        
        self.manual_db.add_documents(splits)
        self.manual_db.persist()

    def ingest_new_logs(self, log_files: List[str]):
        docs = []
        for file_path in log_files:
            with open(file_path, "r") as f:
                log_data = json.load(f)
            
            # Create document from log
            content = f"Equipment ID: {log_data['equipment_id']}\nTimestamp: {log_data['timestamp']}\nSensor Data: {log_data['sensor_data']}\nAnomaly Summary: {log_data['anomaly_summary']}\nHistorical Analysis: {log_data.get('historical_analysis', '')}\nFailure Prediction: {log_data.get('failure_prediction', {})}"
            doc = Document(
                page_content=content,
                metadata={
                    "log_id": log_data['log_id'],
                    "equipment_id": log_data['equipment_id'],
                    "timestamp": log_data['timestamp'],
                    "source": file_path
                }
            )
            docs.append(doc)
        
        if docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            splits = splitter.split_documents(docs)
            self.logs_db.add_documents(splits)
            self.logs_db.persist()

    def retrieve_context(self, query: str, k: int = 5) -> Tuple[str, List[dict]]:
        # Retrieve from manuals
        manual_results = self.manual_db.similarity_search(query, k=k)
        logs_results = self.logs_db.similarity_search(query, k=k)
        
        all_results = manual_results + logs_results
        context = "\n\n".join([doc.page_content for doc in all_results])
        sources = [doc.metadata for doc in all_results]
        
        return context, sources

    def add_manual_chunk(self, chunk: ManualChunk):
        doc = Document(
            page_content=chunk.content,
            metadata={
                "manual_id": chunk.manual_id,
                "equipment_type": chunk.equipment_type,
                "section_id": chunk.section_id,
                "keywords": ", ".join(chunk.keywords),
                "version": chunk.metadata.version,
                "last_updated": chunk.metadata.last_updated.isoformat()
            }
        )
        self.manual_db.add_documents([doc])
        self.manual_db.persist()

    def add_log_data(self, log: LogData):
        content = f"Log ID: {log.log_id}\nEquipment ID: {log.equipment_id}\nTimestamp: {log.timestamp.isoformat()}\nSensor Data: {log.sensor_data}\nAnomaly Summary: {log.anomaly_summary}\nHistorical Analysis: {log.historical_analysis or ''}\nFailure Prediction: {log.failure_prediction.dict() if log.failure_prediction else {}}\nRemediation: { [r.dict() for r in log.remediation] }"
        doc = Document(
            page_content=content,
            metadata={
                "log_id": log.log_id,
                "equipment_id": log.equipment_id,
                "timestamp": log.timestamp.isoformat(),
                "source": "log_upload"
            }
        )
        self.logs_db.add_documents([doc])
        self.logs_db.persist()