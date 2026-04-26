"""
Tool 2: Vector Store for semantic device search.

Uses ChromaDB (HTTP client → container) and sentence-transformers for embeddings.
Stores device name + description and supports semantic similarity search.

Connection settings are read from env vars (set in .env):
  CHROMA_HOST   (default: localhost)
  CHROMA_PORT   (default: 8000)
  CHROMA_TOKEN  (required)
"""

import json
import os
import time

import chromadb
from chromadb.config import Settings
from langchain_core.tools import StructuredTool
from sentence_transformers import SentenceTransformer

from data.sample_data import DEVICES


COLLECTION_NAME = "measuring_devices"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_TOKEN = os.environ["CHROMA_TOKEN"]


class VectorStoreTool:
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
                client = chromadb.HttpClient(
                    host=CHROMA_HOST,
                    port=CHROMA_PORT,
                    settings=Settings(
                        chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                        chroma_client_auth_credentials=CHROMA_TOKEN,
                    ),
                )
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

    def get_tools(self) -> list:
        return [
            StructuredTool.from_function(
                func=self.search,
                name="search_devices",
                description="Semantic search over measuring devices by purpose, use case, or measurement type. Use for open-ended questions like 'what device should I use for RF work?'",
            ),
            StructuredTool.from_function(
                func=self.get_device_by_name,
                name="get_device_by_name",
                description="Find a device by its name. Use when you know the device name or part of it.",
            ),
        ]

    def search(self, query: str, top_k: int = 3) -> str:
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

        return json.dumps({"query": query, "results": hits})

    def get_device_by_name(self, name: str) -> str:
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
            return json.dumps({"error": f"No device matching '{name}' found."})
        return json.dumps({"matches": matches})
