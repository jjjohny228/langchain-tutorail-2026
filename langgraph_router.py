from typing import TypedDict, Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.prompts import format_document
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

llm_with_low_temperature = ChatOpenAI(model="gpt-5.4-mini", temperature=0.1)
llm_with_high_temperature = ChatOpenAI(model="gpt-5.4-mini", temperature=0.7)


class State(TypedDict):
    user_query: str
    messages: Annotated[list, add_messages]
    domain: Literal["records", "insurance"]
    documents: list[Document]
    answer: str


class Input(TypedDict):
    user_query: str


class Output(TypedDict):
    documents: list[Document]
    answer: str


medical_records_store = InMemoryVectorStore.from_documents(
    documents=[], embedding=embeddings
)
records_retriever = medical_records_store.as_retriever()

medical_insurance_store = InMemoryVectorStore.from_documents(
    documents=[], embedding=embeddings
)
insurance_faqs_retriever = medical_insurance_store.as_retriever()


router_prompt = SystemMessage("""You need to decide which domain to route the user query to. You have two
domains to choose from:
- records: contains medical records of the patient, such as
diagnosis, treatment, and prescriptions.
- insurance: contains frequently asked questions about insurance
policies, claims, and coverage.
Output only the domain name.""")


def router_node(state: State) -> State:
    user_query = HumanMessage(state["user_query"])
    messages = [router_prompt, *state["messages"], user_query]
    result = llm_with_low_temperature.invoke(messages)
    return {"messages": result, "domain": result.content}


def pick_retriever(state: State) -> State:
    if state["domain"] == "records":
        return "retrieve_medical_records"
    elif state["domain"] == "insurance":
        return "retrieve_insurance_faqs"
    else:
        raise ValueError("Invalid domain")


def retrieve_medical_records(state: State) -> State:
    documents = records_retriever.invoke(state["user_query"])
    return {"documents": documents}


def retrieve_insurance_faqs(state: State) -> State:
    documents = insurance_faqs_retriever.invoke(state["user_query"])
    return {"documents": documents}


medical_records_prompt = SystemMessage(
    """You are a helpful medical chatbot who answers questions based on the
patient's medical records, such as diagnosis, treatment, and
prescriptions."""
)

insurance_faqs_prompt = SystemMessage(
    """You are a helpful medical insurance chatbot who answers frequently asked
questions about insurance policies, claims, and coverage."""
)


def get_answer(state: State) -> State:
    if state["domain"] == "records":
        system_prompt = medical_records_prompt
    elif state["domain"] == "insurance":
        system_prompt = insurance_faqs_prompt

    messages = [
        system_prompt,
        *state["messages"],
        HumanMessage(f"Documents: {state['documents']}"),
    ]
    result = llm_with_high_temperature.invoke(messages)
    return {"answer": result.content, "messages": messages}


builder = StateGraph(State, input_schema=Input, output_schema=Output)

builder.add_node("router_node", router_node)
builder.add_node("pick_retriever", pick_retriever)
builder.add_node("retrieve_medical_records", retrieve_medical_records)
builder.add_node("retrieve_insurance_faqs", retrieve_insurance_faqs)
builder.add_node("get_answer", get_answer)

builder.add_edge(START, "router_node")
builder.add_conditional_edges("router_node", pick_retriever)
builder.add_edge("retrieve_medical_records", "get_answer")
builder.add_edge("retrieve_insurance_faqs", "get_answer")
builder.add_edge("get_answer", END)

graph = builder.compile()

input = {"user_query": "Am I covered for COVID-19 treatment?"}
for c in graph.stream(input):
    print(c)
