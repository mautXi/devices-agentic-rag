"""
Tool 1: Knowledge Graph for device components backed by Neo4j.

Graph schema:
  (:Device  {id, name, category})
  (:Component {id, name, description, manufacturer})
  (:Device)-[:USES]->(:Component)

Requires a running Neo4j instance — see neo4j_start.sh.
Connection settings are read from env vars with sensible defaults:
  NEO4J_URI      (default: bolt://localhost:7687)
  NEO4J_USER     (default: neo4j)
  NEO4J_PASSWORD (default: password)
"""

import json
import os
import time

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from data.sample_data import DEVICES, COMPONENTS

# Env vars for Neo4j connection; if not set, defaults assume local setup with podman-compose
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class KnowledgeGraphTool:
    name = "knowledge_graph"
    description = (
        "Query the knowledge graph for device components. "
        "Use this to find: which components a device uses, "
        "which devices use a specific component, "
        "component descriptions and manufacturers."
    )

    def __init__(self, retries: int = 10, retry_delay: float = 3.0):
        print("[KnowledgeGraph] Connecting to Neo4j...")
        self.driver = self._connect(retries, retry_delay)
        self._ensure_constraints()
        self._seed_if_empty()

    def close(self):
        self.driver.close()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _connect(self, retries: int, delay: float) -> GraphDatabase:
        for attempt in range(1, retries + 1):
            try:
                driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
                driver.verify_connectivity()
                print(f"[KnowledgeGraph] Connected to Neo4j at {NEO4J_URI}.")
                return driver
            except ServiceUnavailable:
                if attempt == retries:
                    raise RuntimeError(
                        f"Neo4j not reachable at {NEO4J_URI} after {retries} attempts.\n"
                    )
                print(f"[KnowledgeGraph] Neo4j not ready, retrying ({attempt}/{retries})...")
                time.sleep(delay)

    def _ensure_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE")

    def _seed_if_empty(self):
        with self.driver.session() as session:
            count = session.run("MATCH (d:Device) RETURN count(d) AS n").single()["n"]
            if count == 0:
                print("[KnowledgeGraph] Seeding data...")
                self._seed_data(session)
                print(f"[KnowledgeGraph] Seeded {len(DEVICES)} devices and {len(COMPONENTS)} components.")
            else:
                print(f"[KnowledgeGraph] Graph already contains data ({count} devices).")

    def _seed_data(self, session):
        for device in DEVICES:
            session.run(
                "MERGE (d:Device {id: $id}) SET d.name = $name, d.category = $category",
                id=device["id"], name=device["name"], category=device["category"],
            )

        for comp in COMPONENTS:
            session.run(
                """
                MERGE (c:Component {id: $id})
                SET c.name = $name, c.description = $description, c.manufacturer = $manufacturer
                """,
                id=comp["id"], name=comp["name"],
                description=comp["description"], manufacturer=comp["manufacturer"],
            )
            for device_id in comp["used_in"]:
                # Relationships between devices and components
                session.run(
                    """
                    MATCH (d:Device {id: $device_id}), (c:Component {id: $comp_id})
                    MERGE (d)-[:USES]->(c)
                    """,
                    device_id=device_id, comp_id=comp["id"],
                )

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_components_of_device(self, device_name: str) -> dict:
        cypher = """
            MATCH (d:Device)-[:USES]->(c:Component)
            WHERE toLower(d.name) CONTAINS toLower($name)
            RETURN d.name AS device, c.name AS name,
                   c.description AS description, c.manufacturer AS manufacturer
        """
        with self.driver.session() as session:
            records = session.run(cypher, name=device_name).data()

        if not records:
            return {"error": f"Device '{device_name}' not found in knowledge graph."}

        return {
            "device": records[0]["device"],
            "component_count": len(records),
            "components": [
                {"name": r["name"], "description": r["description"], "manufacturer": r["manufacturer"]}
                for r in records
            ],
        }

    def get_devices_using_component(self, component_name: str) -> dict:
        cypher = """
            MATCH (d:Device)-[:USES]->(c:Component)
            WHERE toLower(c.name) CONTAINS toLower($name)
            RETURN c.name AS component, c.manufacturer AS manufacturer,
                   d.name AS device_name, d.category AS category
        """
        with self.driver.session() as session:
            records = session.run(cypher, name=component_name).data()

        if not records:
            return {"error": f"Component '{component_name}' not found in knowledge graph."}

        return {
            "component": records[0]["component"],
            "manufacturer": records[0]["manufacturer"],
            "used_by_count": len(records),
            "devices": [{"name": r["device_name"], "category": r["category"]} for r in records],
        }

    def get_component_info(self, component_name: str) -> dict:
        cypher = """
            MATCH (c:Component)
            WHERE toLower(c.name) CONTAINS toLower($name)
            RETURN c.name AS name, c.description AS description, c.manufacturer AS manufacturer
            LIMIT 1
        """
        with self.driver.session() as session:
            record = session.run(cypher, name=component_name).single()

        if not record:
            return {"error": f"Component '{component_name}' not found in knowledge graph."}

        return {"name": record["name"], "description": record["description"], "manufacturer": record["manufacturer"]}

    def list_all_components(self) -> dict:
        cypher = "MATCH (c:Component) RETURN c.name AS name, c.manufacturer AS manufacturer ORDER BY c.name"
        with self.driver.session() as session:
            records = session.run(cypher).data()
        return {"total": len(records), "components": records}

    def list_all_devices(self) -> dict:
        cypher = "MATCH (d:Device) RETURN d.name AS name, d.category AS category ORDER BY d.name"
        with self.driver.session() as session:
            records = session.run(cypher).data()
        return {"total": len(records), "devices": records}

    # ------------------------------------------------------------------
    # Main callable entry point used by the agent
    # ------------------------------------------------------------------

    def run(self, query: str, **kwargs) -> str:
        """
        Dispatch based on the query string. The agent passes a JSON string like:
          {"action": "get_components_of_device", "device_name": "Fluke 87V Multimeter"}
          {"action": "get_devices_using_component", "component_name": "FPGA"}
          {"action": "get_component_info", "component_name": "ADC"}
          {"action": "list_all_components"}
          {"action": "list_all_devices"}
        """
        try:
            params = json.loads(query)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON input to knowledge_graph tool."})

        action = params.get("action", "")

        if action == "get_components_of_device":
            result = self.get_components_of_device(params.get("device_name", ""))
        elif action == "get_devices_using_component":
            result = self.get_devices_using_component(params.get("component_name", ""))
        elif action == "get_component_info":
            result = self.get_component_info(params.get("component_name", ""))
        elif action == "list_all_components":
            result = self.list_all_components()
        elif action == "list_all_devices":
            result = self.list_all_devices()
        else:
            result = {
                "error": f"Unknown action '{action}'.",
                "available_actions": [
                    "get_components_of_device",
                    "get_devices_using_component",
                    "get_component_info",
                    "list_all_components",
                    "list_all_devices",
                ],
            }

        return json.dumps(result, indent=2)
