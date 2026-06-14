import ast
from typing import Annotated, TypedDict
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


@tool
def calculator(query: str) -> str:
    """A simple calculator tool. Input should be a mathematical expression."""
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]

embedding = OpenAIEmbeddings()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    selected_tools: list[str]


tools_retriever = InMemoryVectorStore.from_documents(
    documents=[
        Document(tool.description, metadata={"name": tool.name}) for tool in tools
    ],
    embedding=embedding,
).as_retriever()


def model_node(state: State) -> State:
    selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
    model = ChatOpenAI(temperature=0.1).bind_tools(selected_tools)
    res = model.invoke(state["messages"])
    return {"messages": res}


def search_tools(state: State) -> State:
    query = state["messages"][-1].content
    tool_documents = tools_retriever.invoke(query)
    return {
        "selected_tools": [document.metadata["name"] for document in tool_documents]
    }


builder = StateGraph(State)
builder.add_node("model_node", model_node)
builder.add_node("search_tools", search_tools)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "search_tools")
builder.add_edge("search_tools", "model_node")
builder.add_conditional_edges("model_node", tools_condition)
builder.add_edge("tools", "model_node")

graph = builder.compile()

for chunk in graph.stream({"messages": "What is price of SpaceX company stocks"}):
    print(chunk)
