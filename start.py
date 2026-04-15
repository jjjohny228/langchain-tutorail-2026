from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

model = ChatOpenAI(model_name='gpt-5.4-nano', temperature=0.2)
loader = TextLoader('')

template = ChatPromptTemplate.from_messages([
    ('system', 'You are me slavor'),
    ('human', 'Question: {question}')
])

chatbot = template | model

print(chatbot.invoke({'question': 'What is a capital of Poland.'}))
