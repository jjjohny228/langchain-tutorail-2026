from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


generate_prompt = SystemMessage(
    """You are an essay assistant tasked with writing excellent 3-paragraph
essays."""
    "Generate the best essay possible for the user's request."
    """If the user provides critique, respond with a revised version of your
previous attempts."""
)

reflection_prompt = SystemMessage(
    """You are a teacher grading an essay submission. Generate critique and
recommendations for the user's submission."""
    """Provide detailed recommendations, including requests for length, depth,
style, etc."""
)

model = ChatOpenAI(model="gpt-5.4-mini")


def generation(state: State) -> State:
    answer = model.invoke([generate_prompt] + state["messages"])
    return {"messages": [answer]}


def reflection(state: State) -> State:
    cls_map = {AIMessage: HumanMessage, HumanMessage: AIMessage}
    new_messages = [reflection_prompt, state["messages"][0]] + [
        cls_map[msg.__class__](msg.content) for msg in state["messages"][1:]
    ]
    answer = model.invoke(new_messages)
    return {"messages": [HumanMessage(content=answer.content)]}


def should_continue(state: State):
    if len(state["messages"]) > 6:
        print("You are a teacher grading an essay submission.", len(state["messages"]))
        return END
    else:
        return "reflection"


builder = StateGraph(State)
builder.add_node("generation", generation)
builder.add_node("reflection", reflection)

builder.add_edge(START, "generation")
builder.add_conditional_edges("generation", should_continue)
builder.add_edge("reflection", "generation")

graph = builder.compile()


for chunk in graph.stream({"messages": "Write esse about last nba finals"}):
    print(chunk)
