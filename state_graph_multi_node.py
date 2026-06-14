from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv


load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    sql_query: str
    sql_query_explanation: str


class Input(TypedDict):
    user_query: str


class Output(TypedDict):
    sql_query: str
    sql_query_explanation: str


model_low_temperature = ChatOpenAI(temperature=0.1, model="gpt-5.4-mini")
model_high_temperature = ChatOpenAI(temperature=0.7, model="gpt-5.4-mini")

sql_query_generate_prompt = SystemMessage(
    "You are a profeccional data analyst that generates SQL queries based on user queries."
)
sql_query_explain_prompt = SystemMessage(
    "You are a profeccional data analyst that explains the SQL query to user."
)


def generate_sql_query(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [sql_query_generate_prompt, *state["messages"], user_message]
    result = model_low_temperature.invoke(messages)
    return {"messages": messages, "sql_query": result.content}


def generate_sql_explanation(state: State) -> State:
    messages = [sql_query_explain_prompt, *state["messages"]]
    result = model_high_temperature.invoke(messages)
    return {"messages": messages, "sql_query_explanation": result.content}


builder = StateGraph(State, input_schema=Input, output_schema=Output)
builder.add_node("generate_sql_query", generate_sql_query)
builder.add_node("generate_sql_explanation", generate_sql_explanation)
builder.add_edge(START, "generate_sql_query")
builder.add_edge("generate_sql_query", "generate_sql_explanation")
builder.add_edge("generate_sql_explanation", END)
graph = builder.compile()
result = graph.invoke({"user_query": "How many employees are there?"})
print(result)
