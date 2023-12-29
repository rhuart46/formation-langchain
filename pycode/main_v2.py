from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Load the OpenAI API key defined in a .env file:
load_dotenv()

# Create a first chain (not really chaining models yet...):
code_chain = LLMChain(
    llm=OpenAI(),
    prompt=PromptTemplate(
        template="Write a {language} function that will {task}.",
        input_variables=["language", "task"],
    ),
)

# Call the chain in order to render and submit the prompt to OpenAI and print the received results:
results = code_chain({"language": "Python", "task": "return a list of numbers"})
print("Generated code:")
print(results["text"])
