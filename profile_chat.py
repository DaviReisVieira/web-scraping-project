import os

import chainlit as cl

from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import CSVLoader


def process_file():
    loader = CSVLoader(file_path='data/resultados_limpos.csv',
                   csv_args={"delimiter": ";"},
                   encoding='utf-8',
    )

    docs = []
    for doc in loader.load():
        docs.append(doc)
    return docs


def get_vectorstore():
    db_path = 'vectorstore/db_data'

    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-large'
    )

    if os.path.exists(db_path):
        print('Loading FAISS database')
        vectorstore = FAISS.load_local(
            db_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore


    text_chunks = []
    docs = process_file()
    for doc in docs:
        text_chunks.append(doc.page_content)

    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    vectorstore.save_local(db_path)
    return vectorstore


async def runnable_moveisAqui():
    msg = cl.Message(
        content=f"Carregando meu conhecimento...⏳", disable_feedback=True, author="Imóveis Aqui"
    )
    await msg.send()

    docsearch = await cl.make_async(get_vectorstore)()

    message_history = ChatMessageHistory()

    prompt = """
    Você é um assistente que ajuda os clientes a buscarem imóveis. \
    Forneça uma resposta conversacional. \
    Caso não saiba a resposta, diga 'Desculpe, não sei como te ajudar..😔'. \
    Traga todas as informações possíveis para ajudar o cliente. \
    Seja educado e prestativo. \
    
    Contexto:
    {context}

    Pergunta:
    {question}

    Resposta:
    """

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    memory.chat_memory.add_ai_message(
        "Olá! Sou o Imóveis Aqui! No que posso te ajudar? 🤖")

    cl.user_session.set("memory", memory)

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-4-0125-preview",
                   temperature=0.1, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(
            search_kwargs={"k": 5}
        ),
        memory=cl.user_session.get("memory"),
        return_source_documents=True,
    )

    promt_template = PromptTemplate(
        input_variables=["context", "question"], template=prompt
    )

    chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(
        prompt=promt_template
    )

    cl.user_session.set("chain", chain)

    msg.content = f"Olá! Sou o Imóveis Aqui! No que posso te ajudar? 🤖"
    msg.disable_feedback = False
    await msg.update()