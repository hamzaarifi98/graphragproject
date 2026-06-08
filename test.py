from backend.pipeline.query_pipeline import query_graph

result = query_graph.invoke({
    "question": "Which products are sold the most in the month before the last"
})

print(result["answer"])