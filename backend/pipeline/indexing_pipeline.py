from backend.services.langgraph.question_router import route_question


if __name__ == "__main__":
    print(route_question({"question": "Who are the customers that buy the most?"}))
