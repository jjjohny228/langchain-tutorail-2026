from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_postgres.vectorstores import PGVector
from langchain_classic.indexes import SQLRecordManager, index
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_core.runnables import RunnablePassthrough
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryStore # данные сохраняются в словарь, стираются в конце запуска программы
import uuid

from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('postiz_info.txt')
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(documents)

llm = ChatOpenAI()

prompt_text = "Summarize the following document:\n\n{doc}"
prompt = ChatPromptTemplate.from_template(prompt_text)

summarize_chain = {'doc': lambda x: x.page_content} | prompt | llm | StrOutputParser()

summaries = summarize_chain.batch(chunks, {"max_concurrency": 5})
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

embeddings = OpenAIEmbeddings()

# The vectorstore to use to index the child chunks
db = PGVector(
    embeddings=embeddings,
    connection=connection,
    collection_name='summaries',
    use_jsonb=True,
)

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# indexing the summaries in our vector store, whilst retaining the original
# documents in our document store:
retriever = MultiVectorRetriever(
    vectorstore=db,
    docstore=store,
    id_key=id_key
)

# Changed from summaries to chunks since we need same length as docs
doc_ids = [str(uuid.uuid4()) for _ in chunks]

# Each summary is linked to the original document by the doc_id
summary_docs = [Document(page_content=s, metadata={'id_key': doc_ids[i]}) for i, s in enumerate(summaries)]

# Add the document summaries to the vector store for similarity search
retriever.vectorstore.add_documents(summary_docs)

# Store the original documents in the document store, linked to their summaries
# via doc_ids
# This allows us to first search summaries efficiently, then fetch the full
# docs when needed
retriever.docstore.mset(list(zip(doc_ids, chunks)))

sub_docs = retriever.vectorstore.similarity_search('postiz', k=2)

# Whereas the retriever will return the larger source document chunks:
full_docs = retrieved_docs = retriever.invoke("chapter on philosophy")

print(sub_docs)
print(full_docs)
