from backend.core.neo4j import neo4j_driver


def run_cypher(query: str, params: dict | None = None) -> list[dict]:
    params = params or {}

    with neo4j_driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]
    

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