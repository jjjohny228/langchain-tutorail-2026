import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('postiz_info.txt')
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(documents)

model = OpenAIEmbeddings()

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector.from_documents(chunks, model, connection=connection)

db.similarity_search("query", k=4)
db.add_documents()
