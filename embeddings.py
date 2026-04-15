import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('postiz_info.txt')
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_documents(documents)

model = OpenAIEmbeddings()

embeddings = model.embed_documents([document.page_content for document in chunks])

print(embeddings)


