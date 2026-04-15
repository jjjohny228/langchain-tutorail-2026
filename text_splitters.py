from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader

loader = TextLoader('postiz_info.txt')
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

text = splitter.split_documents(documents)

# code splitter
PYTHON_CODE = """
def hello_world():
print("Hello, World!")
# Call the function
hello_world()
"""

python_splitter = RecursiveCharacterTextSplitter.from_language(chunk_size=50, chunk_overlap=0, language=Language.PYTHON)
result = python_splitter.create_documents([PYTHON_CODE])
print(result)
