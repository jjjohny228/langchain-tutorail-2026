from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


class State(TypedDict):
    messages: Annotated[
        list, add_messages
    ]  # добавляет метаданные что сообщения добавляем в конец списка


builder = StateGraph(State)

model = ChatOpenAI(temperature=0)


# дальше создаем ноду которая принимает state
def chatbot(state: State) -> State:
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


builder.add_node("chatbot", chatbot)

# add edges
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()
graph.get_graph().draw_mermaid_png()

input = {"messages": [HumanMessage("hi!")]}

result = graph.invoke(input)
print(result)
