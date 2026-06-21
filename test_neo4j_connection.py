from neo4j import GraphDatabase

NEO4J_URI = "bolt://13.50.242.131:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Password123."

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

with driver.session() as session:
    result = session.run("RETURN 'Neo4j connected' AS message")
    print(result.single()["message"])

driver.close()