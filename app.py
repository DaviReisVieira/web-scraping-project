import os
import time
import openai
import random
import logging

import chainlit as cl
from chainlit.types import ThreadDict

from typing import List
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


from profile_chat import runnable_moveisAqui


from dotenv import load_dotenv

load_dotenv()


MAX_RETRIES = int(os.getenv("MAX_RETRIES", 5))
BACKOFF_IN_SECONDS = float(os.getenv("BACKOFF_IN_SECONDS", 1))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")


logger = logging.getLogger(__name__)


def select_runnable_profile() -> cl.ChatProfile:
    return runnable_moveisAqui


def backoff(attempt: int) -> float:
    return BACKOFF_IN_SECONDS * 2**attempt + random.uniform(0, 1)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)

    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "USER_MESSAGE":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    await select_runnable_profile()()


@cl.on_chat_start
async def start_chat():

    await cl.Avatar(
        name="Imóveis Aqui",
        url="https://picsum.photos/300",
    ).send()

    await cl.Avatar(
        name="Error",
        url="https://cdn-icons-png.flaticon.com/512/8649/8649595.png"
    ).send()

    await cl.Avatar(
        name="You",
        url="https://media.architecturaldigest.com/photos/5f241de2c850b2a36b415024/master/w_1600%2Cc_limit/Luke-logo.png"
    ).send()

    await select_runnable_profile()()

@cl.on_message
async def main(message: cl.Message):
    memory = cl.user_session.get("memory") 

    chain: ConversationalRetrievalChain = cl.user_session.get(
        "chain")

    cb = cl.AsyncLangchainCallbackHandler()

    res = None

    for attempt in range(MAX_RETRIES):
        try:
            res = await chain.ainvoke(message.content, callbacks=[cb])
            break
        except openai._exceptions.APITimeoutError:
            wait_time = backoff(attempt)
            logger.exception(
                f"OpenAI API timeout occurred. Waiting {wait_time} seconds and trying again.")
            time.sleep(wait_time)
            pass
        except openai._exceptions.APIError:
            wait_time = backoff(attempt)
            logger.exception(
                f"OpenAI API error occurred. Waiting {wait_time} seconds and trying again.")
            time.sleep(wait_time)
            pass
        except Exception as e:
            logger.exception(f"A non retriable error occurred. {e}")
            break

    answer = res["answer"]
    source_documents: List[Document] = res["source_documents"]

    text_elements: List[cl.Text] = []

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"Ref. {source_idx}"
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )
    #     source_names = [text_el.name for text_el in text_elements]

    #     if source_names:
    #         answer += f"\nFontes: {', '.join(source_names)}"
    #     else:
    #         answer += "\nNenhuma fonte encontrada."

    await cl.Message(author="Imóveis Aqui", content=answer, elements=text_elements).send()
 
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(answer)