from langchain_classic.indexes import SQLRecordManager, index
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_classic.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection_name = "my_docs"
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
namespace = "my_docs_namespace"

vector_store = PGVector(
    embeddings=embeddings_model,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
    )

record_manager = SQLRecordManager(
    namespace=namespace,
    db_url=connection
)

# Create the schema if it doesn't exist
record_manager.create_schema()

# Create documents
docs = [
    Document(page_content='there are cats in the pond', metadata={"id": 1, "source": "cats.txt"}),
    Document(page_content='ducks are also found in the pond', metadata={"id": 2, "source": "ducks.txt"})
]

# Index the documents
index_1 = index(docs_source=docs,
                record_manager=record_manager,
                vector_store=vector_store,
                cleanup="incremental", # prevent duplicate documents)
                source_id_key="source", # use the source field as the source_id
                )

index_2 = index(
docs,
record_manager,
vector_store,
cleanup="incremental",
source_id_key="source",
)

docs[0].page_content = "I just modified this document!"

index_3 = index(
docs,
record_manager,
vector_store,
cleanup="incremental",
source_id_key="source",
)

print("Index attempt 1:", index_1)
print("Index attempt 2:", index_2)
print("Index attempt 3:", index_3)


