"""
This version aims at adding a feature to v1 for limiting the size of the text sent to OpenAI,
in the case of very long conversations. But currently this feature doesn't work well with FileChatMessageHistory,
so we have to remove it and that is why I decided to make a second version of the code.
"""
from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


load_dotenv()

chat_model_client = ChatOpenAI(verbose=True)

# ConversationSummaryMemory aks a LLM model to summarize the ton of text of the conversation so that it can then be
# sent as a system message (think of it as a context) to the chat model, and the continuation of the conversation will
# just need the new human message:
chat_memory = ConversationSummaryMemory(
    llm=chat_model_client,
    memory_key="messages",
    return_messages=True,
)
chat_prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

# LLM responses will not show the difference. In order to see it, we enable the verbose mode so that we can
# see each time the generated system message. We must enable this mode in the chat model configuration too.
chat_chain = LLMChain(llm=chat_model_client, prompt=chat_prompt, memory=chat_memory, verbose=True)

# WARNING: this version sends more requests and if you are limited by your free tier subscription as I do (3 RPM),
#   you will hit the limit more quickly...

while True:
    human_message = input(">> ")
    result = chat_chain({"content": human_message})
    print(result["text"])
