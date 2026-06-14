from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain_classic.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

# replace this with the connection details of your db
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOpenAI(model="gpt-4", temperature=0)
# convert question to sql query
write_query = create_sql_query_chain(llm, db)
# Execute SQL query
execute_query = QuerySQLDatabaseTool(db=db)
# combined
chain = write_query | execute_query
# invoke the chain
chain.invoke("How many employees are there?")
