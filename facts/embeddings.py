"""
Split the facts document into chunks of a given size, build the corresponding embeddings by calling an OpenAI
embedding model (for very tiny cost, few uses should stay free with a free tier) and store them in a vector
database (here, a ChromaDB instance).

This is meant to be run only when the facts document gets updates, or when the vector database needs to be (re)created.
"""
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma


# Load the API key for OpenAI:
load_dotenv()

# Load the facts and split them into chunks that will help answering user questions:
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,
)
loader = TextLoader("facts.txt")
chunks = loader.load_and_split(text_splitter=text_splitter)

# Create embeddings from these chunks and store them in a ChromaDB instance
# (ChromaDB uses a SQLite database, so there is nothing to install outside of pip):
embedder = OpenAIEmbeddings()
db = Chroma.from_documents(
    documents=chunks, embedding=embedder, persist_directory="emb"
)

# Demonstrate usage of the vector database:
results = db.similarity_search_with_score(
    query="What is an interesting fact about the English language?", k=5
)
for result in results:
    print("\n----------------------------------------------------------------------\n")
    print("Score:", result[1])
    print("Chunk:")
    print(result[0].page_content)

print("\n----------------------------------------------------------------------\n")
