from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
import cassio
import chainlit as cl
import os
from cassio.config import check_resolve_session,check_resolve_keyspace

cassio.init(
    token=os.environ['ASTRA_DB_APPLICATION_TOKEN'],
    database_id=os.environ['ASTRA_DB_DATABASE_ID'],
    keyspace=os.environ.get('ASTRA_DB_KEYSPACE'),
)

@cl.on_chat_start
async def on_chat_start():
    model = ChatOpenAI(streaming=True)
    vectorstore = Cassandra(
    embedding=OpenAIEmbeddings(),
    session=check_resolve_session(None),
    keyspace=check_resolve_keyspace(None),
    table_name='ptc_openai_en',
    )
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a customer service of a ecommerce store and you are asked to pick products for a customer.Include the product description when responding with the list of product recommendation.All the responses should be the same language as the user used",
            ),
            ("human", "{question}"),
        ]
    )
    #chain = ConversationalRetrievalChain(llm=model, context=retriever, prompt=prompt, output_parser=StrOutputParser())
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")  

    res = await chain.invoice(message.content)

    await cl.Message(content=res).send()
