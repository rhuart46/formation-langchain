from argparse import ArgumentParser

from dotenv import load_dotenv

from langchain.chains import LLMChain, SequentialChain
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
llm = OpenAI()
code_generation_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Write a {language} function that will {task}.",
        input_variables=["language", "task"],
    ),
    output_key="code",
)
test_generation_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        template="Write a test for the following {language} code:\n\n{code}",
        input_variables=["language", "code"],
    ),
    output_key="test",
)
full_chain = SequentialChain(
    chains=[code_generation_chain, test_generation_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"],
)

# Call the chain in order to render and submit the prompt to OpenAI and print the received results:
results = full_chain({"language": args.language, "task": args.task})
print("Generated code:\n")
print(results["code"])
print("\nGenerated test:\n")
print(results["test"])
