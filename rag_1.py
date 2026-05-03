from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

loader = TextLoader('greeks.txt')
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(document)

embedding_model = OpenAIEmbeddings()

connection = 'postgresql+psycopg://langchain:langchain@localhost:6024/langchain'
db = PGVector.from_documents(docs, embedding_model, connection=connection)

question = 'Who are the key figures in the ancient greek history of philosophy?'

retriever = db.as_retriever()
context = retriever.invoke(question)

llm = ChatOpenAI(model_name='gpt-5.4-mini')
prompt = PromptTemplate.from_template('Question: {question}. Context: {context}')

chain = prompt | llm
result = chain.invoke({'question': question, 'context': context})
print(result)
