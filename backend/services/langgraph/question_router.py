from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.schemas.router import QueryState

load_dotenv()




client = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


def route_question(state: QueryState) -> QueryState:
    question = state["question"]

    response = client.invoke([
        {
            "role": "system",
            "content": """
You are a router for a RAG invoice assistant.

If question is : [
        Who had most delayed deliveries in the last month?
        Who had most bad reviews in the last month?
        Who sold the most in the last month?
] or something similar you rout to kg because it needs graph 
relationships to answer. 

If question is : [
what is the total revenue for last month?
what is the average order value for last month?
whats the most sold products in last month
tell me total revenue of best seller
] then when route to sql.

Questions asking for revenue, sales amount, value, or money should route to sql.

Return only one word:

pdf 
sql 
kg
hybrid 


"""

        },
        {
            "role": "user",
            "content": question
        }
    ])

    route = response.content.strip().lower()

    if route not in ["pdf", "sql", "kg", "hybrid"]:
        route = "pdf"

    return {
        **state,
        "route": route
    }


def choose_route(state: QueryState) -> str:
    return state["route"]
