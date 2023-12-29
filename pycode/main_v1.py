from dotenv import load_dotenv

from langchain.llms import OpenAI

# Load the OpenAI API key defined in a .env file:
load_dotenv()

# Create a client for the OpenAI API:
llm = OpenAI()

# Send a prompt to the API and print the received generated text:
print(llm("Write a very very short poem"))
