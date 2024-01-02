"""
Application answering a question thanks to OpenAI, with the help of a list of facts
stored in a vector database. These facts are to be preferred over OpenAI's made up
answers.
"""
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from redundant_filter_retriever import RedundantFilterRetriever


# Load the API key for OpenAI:
load_dotenv()

# Load the facts and split them into chunks that will help answering user questions:
embedder = OpenAIEmbeddings()
db = Chroma(persist_directory="emb", embedding_function=embedder)
# retriever = db.as_retriever()

# Replace the standard chroma retriever with one that excludes duplicates (more
# precisely, "too close" embeddings). Duplicates would appear if we ran several times
# the script creating embeddings from the facts document, this retriever is a way to
# get rid of them (the vector store does not prevent from inserting several times the
# same embedding):
retriever = RedundantFilterRetriever(embedder=embedder, chroma=db)

# Instantiate a pre-built high-level chain template matching our expectations,
# instead of building the chain ourselves:
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(), retriever=retriever, chain_type="stuff"
)

# Run it:
result = chain.run("What is an interesting fact about the English language?")
print(result)
