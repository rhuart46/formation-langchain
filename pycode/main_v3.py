from argparse import ArgumentParser

from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Load the OpenAI API key defined in a .env file:
load_dotenv()

# Create the CLI:
parser = ArgumentParser(
    description=(
        "Generate the code of a function achieving the requested task in the requested language, "
        "using an LLM model from Open AI"
    )
)
parser.add_argument(
    "-t",
    "--task",
    help="description of what the code should achieve",
    default="return a list of numbers",
)
parser.add_argument(
    "-l",
    "--language",
    help="the programming language in which the code should be written",
    default="Python",
)
args = parser.parse_args()


# Create a first chain (not really chaining models yet...):
code_chain = LLMChain(
    llm=OpenAI(),
    prompt=PromptTemplate(
        template="Write a {language} function that will {task}.",
        input_variables=["language", "task"],
    ),
)

# Call the chain in order to render and submit the prompt to OpenAI and print the received results:
results = code_chain({"language": args.language, "task": args.task})
print("Generated code:")
print(results["text"])
