"""
Tool 3: Hybrid search combining semantic vector search with knowledge graph enrichment.

Finds devices by meaning/purpose, then enriches each result with its component data
from the knowledge graph — giving a complete picture in a single tool call.
"""

import json

from langchain_core.tools import StructuredTool

from tools.knowledge_graph import KnowledgeGraphTool
from tools.vector_store import VectorStoreTool


class HybridSearchTool:
    def __init__(self, kg: KnowledgeGraphTool, vs: VectorStoreTool):
        self.kg = kg
        self.vs = vs

    def get_tools(self) -> list:
        return [
            StructuredTool.from_function(
                func=self.hybrid_search,
                name="hybrid_search",
                description=(
                    "Find devices by purpose or use case and return their components. "
                    "Use for complex queries like 'what device should I use for RF work and what's inside it?'"
                ),
            ),
        ]

    def hybrid_search(self, query: str, top_k: int = 3) -> str:
        """Semantic search enriched with component data from the knowledge graph."""
        vs_result = json.loads(self.vs.search(query, top_k))
        enriched = []
        for device in vs_result.get("results", []):
            kg_result = json.loads(self.kg.get_components_of_device(device["name"]))
            enriched.append({
                "name": device["name"],
                "category": device["category"],
                "use_case": device["use_case"],
                "similarity_score": device["similarity_score"],
                "components": kg_result.get("components", []),
            })
        return json.dumps({"query": query, "results": enriched})
