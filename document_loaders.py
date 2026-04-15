from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("https://www.langchain.com/")
content = loader.load()
print(content)
