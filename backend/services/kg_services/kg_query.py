from backend.core.neo4j import neo4j_driver


def run_cypher(query: str, params: dict | None = None) -> list[dict]:
    params = params or {}

    with neo4j_driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]


def delete_knowledge_graph() -> dict[str, int]:
    with neo4j_driver.session() as session:
        node_count = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
        relationship_count = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        session.run("MATCH (n) DETACH DELETE n").consume()

    return {
        "deleted_nodes": node_count,
        "deleted_relationships": relationship_count,
    }


def get_top_categories(limit: int = 10) -> list[dict]:
    return run_cypher(
        """
        MATCH (:Order)-[:CONTAINS]->(:Product)-[:IN_CATEGORY]->(c:Category)
        RETURN c.name AS category, count(*) AS orders
        ORDER BY orders DESC
        LIMIT $limit
        """,
        {"limit": limit},
    )
