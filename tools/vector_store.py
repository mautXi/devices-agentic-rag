"""
Tool 2: Vector Store for semantic device search.

Uses ChromaDB (HTTP client → container) and sentence-transformers for embeddings.
Stores device name + description and supports semantic similarity search.

Connection settings are read from env vars with sensible defaults:
  CHROMA_HOST  (default: localhost)
  CHROMA_PORT  (default: 8000)
"""

import json
import os
import time

import chromadb
from sentence_transformers import SentenceTransformer

from data.sample_data import DEVICES


COLLECTION_NAME = "measuring_devices"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))


class VectorStoreTool:
    name = "vector_store"
    description = (
        "Semantic search over measuring devices. "
        "Use this to find devices by purpose, measurement type, or use case. "
        "Returns device names, descriptions, and categories ranked by relevance."
    )

    def __init__(self, retries: int = 10, retry_delay: float = 3.0):
        print("[VectorStore] Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print(f"[VectorStore] Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
        self.client = self._connect(retries, retry_delay)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        if self.collection.count() == 0:
            self._seed_data()
        else:
            print(f"[VectorStore] Collection loaded ({self.collection.count()} devices).")

    def _connect(self, retries: int, delay: float) -> chromadb.HttpClient:
        for attempt in range(1, retries + 1):
            try:
                client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
                client.heartbeat()
                print(f"[VectorStore] Connected to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}.")
                return client
            except Exception:
                if attempt == retries:
                    raise RuntimeError(
                        f"ChromaDB not reachable at {CHROMA_HOST}:{CHROMA_PORT} after {retries} attempts.\n"
                        "Start the containers with:  podman-compose up -d"
                    )
                print(f"[VectorStore] ChromaDB not ready, retrying ({attempt}/{retries})...")
                time.sleep(delay)

    def _seed_data(self):
        print("[VectorStore] Seeding device data...")
        documents, ids, metadatas = [], [], []

        for device in DEVICES:
            text = f"{device['name']}. {device['description']} Use case: {device['use_case']}"
            documents.append(text)
            ids.append(device["id"])
            metadatas.append({
                "name": device["name"],
                "category": device["category"],
                "use_case": device["use_case"],
            })

        embeddings = self.embedder.encode(documents).tolist()
        self.collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        print(f"[VectorStore] Seeded {len(documents)} devices.")

    def search(self, query: str, top_k: int = 3) -> dict:
        """Semantic search — returns top_k most relevant devices."""
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "name": results["metadatas"][0][i]["name"],
                "category": results["metadatas"][0][i]["category"],
                "use_case": results["metadatas"][0][i]["use_case"],
                "description": results["documents"][0][i],
                "similarity_score": round(1 - results["distances"][0][i], 3),
            })

        return {"query": query, "results": hits}

    def get_device_by_name(self, name: str) -> dict:
        """Fetch a device by exact or partial name match."""
        all_results = self.collection.get(include=["documents", "metadatas"])
        matches = []
        name_lower = name.lower()
        for i, meta in enumerate(all_results["metadatas"]):
            if name_lower in meta["name"].lower():
                matches.append({
                    "name": meta["name"],
                    "category": meta["category"],
                    "use_case": meta["use_case"],
                    "description": all_results["documents"][i],
                })
        if not matches:
            return {"error": f"No device matching '{name}' found."}
        return {"matches": matches}

    # ------------------------------------------------------------------
    # Main callable entry point used by the agent
    # ------------------------------------------------------------------

    def run(self, query: str, **kwargs) -> str:
        """
        Dispatch based on JSON input from the agent:
          {"action": "search", "query": "oscilloscope for signal analysis", "top_k": 3}
          {"action": "get_device_by_name", "name": "Fluke"}
        """
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON input to vector_store tool."})

        action = params.get("action", "")

        if action == "search":
            result = self.search(
                query=params.get("query", ""),
                top_k=params.get("top_k", 3),
            )
        elif action == "get_device_by_name":
            result = self.get_device_by_name(params.get("name", ""))
        else:
            result = {
                "error": f"Unknown action '{action}'.",
                "available_actions": ["search", "get_device_by_name"],
            }

        return json.dumps(result, indent=2)
