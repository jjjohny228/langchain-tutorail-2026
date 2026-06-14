import ast
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated
from dotenv import load_dotenv


load_dotenv()


@tool
def calculator_tool(query: str) -> str:
    """A simple calculator tool. Input should be a math expression."""
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()

tools = [calculator_tool, search]

model = ChatOpenAI(model="gpt-5.4-mini", temperature=0.1).bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def model_node(state: State) -> State:
    result = model.invoke(state["messages"])
    return {"messages": [result]}


builder = StateGraph(State)

builder.add_node("model_node", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model_node")
builder.add_conditional_edges("model_node", tools_condition)
builder.add_edge("tools", "model_node")
graph = builder.compile()

for chunk in graph.stream(
    {"messages": "How old was the 30th president of the United States when he died?"}
):
    print(chunk)
