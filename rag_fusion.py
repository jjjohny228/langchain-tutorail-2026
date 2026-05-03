from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_vector_retriever import embeddings
load_dotenv()

embeddings = OpenAIEmbeddings()

loader = TextLoader('greeks.txt')
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(document)

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector(embeddings, connection=connection)
retriever = db.as_retriever()

prompt_rag_fusion = ChatPromptTemplate.from_template("""You are a helpful
    assistant that generates multiple search queries based on a single input
    query. \n
    Generate multiple search queries related to: {question} \n
    Output (4 queries):""")


def parse_queries_output(message):
    return message.content.split('\n')

llm = ChatOpenAI(temperature=0)
query_gen = prompt_rag_fusion | llm | parse_queries_output

def reciprocal_rank_fusion(results: list[list], k=60):
    """reciprocal rank fusion on multiple lists of ranked documents
    and an optional parameter k used in the RRF formula
    """

    # Initialize a dictionary to hold fused scores for each document
    # Documents will be keyed by their contents to ensure uniqueness
    fused_scores = {}
    documents = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list,
        # with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Use the document contents as the key for uniqueness
            doc_str = doc.page_content
            # If the document hasn't been seen yet,
            # - initialize score to 0
            # - save it for later
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
                documents[doc_str] = doc

            # Update the score of the document using the RRF formula:
            # 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order
    # to get the final reranked results
    reranked_doc_strs = sorted(
        fused_scores, key=lambda d: fused_scores[d], reverse=True
    )
    # retrieve the corresponding doc for each doc_str
    return [
    documents[doc_str]
    for doc_str in reranked_doc_strs
    ]


retrieval_chain = query_gen | retriever.batch | reciprocal_rank_fusion

prompt = ChatPromptTemplate.from_template("""Answer the following question based
    on this context:
    {context}
    Question: {question}
    """)
llm = ChatOpenAI(temperature=0)


@chain
def multi_query_qa(input):
    # fetch relevant documents
    docs = retrieval_chain.invoke(input)
    # format prompt
    formatted = prompt.invoke({"context": docs, "question": input})
    # generate answer
    answer = llm.invoke(formatted)
    return answer

result = multi_query_qa.invoke("""Who are some key figures in the ancient greek history
of philosophy?""")

print(result)
