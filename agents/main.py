"""
Application answering a question requiring knowledge stored in a database with ChatGPT.
"""
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from tools.sql import run_query_tool


# Load the API key for OpenAI:
load_dotenv()

# ...
chat = ChatOpenAI()
prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# ...
tools = [run_query_tool]
agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)

# ...
agent_executor("How many users are in the database?")
