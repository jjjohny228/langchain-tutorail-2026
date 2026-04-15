import os
from operator import index

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    loader = TextLoader('postiz_info.txt')
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_splits = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), model='text-embedding-3-small')
    PineconeVectorStore.from_documents(text_splits, embeddings, index_name=os.getenv('INDEX_NAME'))
    print('finished')


