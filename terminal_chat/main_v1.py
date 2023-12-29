"""
Creation of a chatbot to be used in the terminal.
After pycode, this exercise aims at understanding how to make our program able to use previous messages in its
responses. We are then able to have a conservation with the LLM model.
"""
from dotenv import load_dotenv

# Note from the course:
# langchain.llms stores completion models, assumed by LangChain (with good reasons) to be the natural standard.
# When we want to use chat models (which are actually tweaked completion models), we need to import a different class
# for the LLM, from langchain.chat_models.
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)


# Load the API key for OpenAI:
load_dotenv()

# Instantiate a memory zone to be embedded in the future chain and make the prompt template system include the
# saved messages as another input variable, to be inserted in a place defined by a placeholder:
chat_memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"),
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
# Note 1: think of it like we prepare to submit "message_1 \n response_1 \n ... \n my_new_message" every time.
# Note 2: in order to show that LangChain is to be seen as a rich toolbox, we added a feature that may not be super
#   useful in real use cases but exists: the ability to very easily store and reload conversation history with
#   FileChatMessageHistory and a simple JSON file name, so that we can continue the conversation after a restart.

# Build now the chain using a chat model and remembering the whole conversation:
chat_model_client = ChatOpenAI()
chat_chain = LLMChain(llm=chat_model_client, prompt=chat_prompt, memory=chat_memory)

# Interactive chat loop:
while True:
    human_message = input(">> ")
    result = chat_chain({"content": human_message})
    print(result["text"])
