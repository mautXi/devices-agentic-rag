"""
Tool 1: Knowledge Graph for device components backed by Neo4j.

Graph schema:
  (:Device  {id, name, category})
  (:Component {id, name, description, manufacturer})
  (:Device)-[:USES]->(:Component)

Connection settings are read from env vars (set in .env):
  NEO4J_URI      (required)
  NEO4J_USER     (required)
  NEO4J_PASSWORD (required)
"""

import json
import os
import time

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from data.sample_data import DEVICES, COMPONENTS

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]


class KnowledgeGraphTool:
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
                SET c.name = $name, c.description = $description,
                    c.manufacturer = $manufacturer, c.category = $category
                """,
                id=comp["id"], name=comp["name"], description=comp["description"],
                manufacturer=comp["manufacturer"], category=comp.get("category", ""),
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

    def get_tools(self) -> list:
        return [
            StructuredTool.from_function(
                func=self.get_components_of_device,
                name="get_components_of_device",
                description="Get all components used by a specific device. Use when asked what parts or components are inside a device.",
            ),
            StructuredTool.from_function(
                func=self.get_devices_using_component,
                name="get_devices_using_component",
                description="Find all devices that use a specific component. Use when asked which devices contain a particular part.",
            ),
            StructuredTool.from_function(
                func=self.get_component_info,
                name="get_component_info",
                description="Get description and manufacturer of a component by name.",
            ),
            StructuredTool.from_function(
                func=self.list_all_components,
                name="list_all_components",
                description="List every component in the knowledge graph with its manufacturer.",
            ),
            StructuredTool.from_function(
                func=self.list_all_devices,
                name="list_all_devices",
                description="List every device in the knowledge graph with its category.",
            ),
            StructuredTool.from_function(
                func=self.get_components_by_category,
                name="get_components_by_category",
                description=(
                    "Find all components belonging to a category. "
                    "Valid categories: signal_processing, protection, power, timing, rf, measurement, signal_generation."
                ),
            ),
        ]

    def get_components_of_device(self, device_name: str) -> str:
        cypher = """
            MATCH (d:Device)-[:USES]->(c:Component)
            WHERE toLower(d.name) CONTAINS toLower($name)
            RETURN d.name AS device, c.name AS name,
                   c.description AS description, c.manufacturer AS manufacturer
        """
        with self.driver.session() as session:
            records = session.run(cypher, name=device_name).data()

        if not records:
            return json.dumps({"error": f"Device '{device_name}' not found in knowledge graph."})

        return json.dumps({
            "device": records[0]["device"],
            "component_count": len(records),
            "components": [
                {"name": r["name"], "description": r["description"], "manufacturer": r["manufacturer"]}
                for r in records
            ],
        })

    def get_devices_using_component(self, component_name: str) -> str:
        cypher = """
            MATCH (d:Device)-[:USES]->(c:Component)
            WHERE toLower(c.name) CONTAINS toLower($name)
            RETURN c.name AS component, c.manufacturer AS manufacturer,
                   d.name AS device_name, d.category AS category
        """
        with self.driver.session() as session:
            records = session.run(cypher, name=component_name).data()

        if not records:
            return json.dumps({"error": f"Component '{component_name}' not found in knowledge graph."})

        return json.dumps({
            "component": records[0]["component"],
            "manufacturer": records[0]["manufacturer"],
            "used_by_count": len(records),
            "devices": [{"name": r["device_name"], "category": r["category"]} for r in records],
        })

    def get_component_info(self, component_name: str) -> str:
        cypher = """
            MATCH (c:Component)
            WHERE toLower(c.name) CONTAINS toLower($name)
            RETURN c.name AS name, c.description AS description,
                   c.manufacturer AS manufacturer, c.category AS category
            LIMIT 1
        """
        with self.driver.session() as session:
            record = session.run(cypher, name=component_name).single()

        if not record:
            return json.dumps({"error": f"Component '{component_name}' not found in knowledge graph."})

        return json.dumps({
            "name": record["name"],
            "description": record["description"],
            "manufacturer": record["manufacturer"],
            "category": record["category"],
        })

    def get_components_by_category(self, category: str) -> str:
        cypher = """
            MATCH (c:Component)
            WHERE toLower(c.category) = toLower($category)
            RETURN c.name AS name, c.description AS description, c.manufacturer AS manufacturer
            ORDER BY c.name
        """
        with self.driver.session() as session:
            records = session.run(cypher, category=category).data()

        if not records:
            return json.dumps({"error": f"No components found for category '{category}'."})

        return json.dumps({"category": category, "total": len(records), "components": records})

    def list_all_components(self) -> str:
        cypher = "MATCH (c:Component) RETURN c.name AS name, c.manufacturer AS manufacturer ORDER BY c.name"
        with self.driver.session() as session:
            records = session.run(cypher).data()
        return json.dumps({"total": len(records), "components": records})

    def list_all_devices(self) -> str:
        cypher = "MATCH (d:Device) RETURN d.name AS name, d.category AS category ORDER BY d.name"
        with self.driver.session() as session:
            records = session.run(cypher).data()
        return json.dumps({"total": len(records), "devices": records})
