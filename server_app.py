from fastapi import FastAPI
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langserve import add_routes

## May be useful later
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_community.document_transformers import LongContextReorder
from functools import partial
from operator import itemgetter
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS


HOME = os.getcwd()
load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

## TODO: Make sure to pick your LLM and do your prompt engineering as necessary for the final assessment
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")
instruct_llm = ChatNVIDIA(model="meta/llama-3.2-3b-instruct")

chat_prompt = ChatPromptTemplate.from_template(
    "You are a document chatbot. Help the user as they ask questions about documents."
    " User messaged just asked you a question: {input}\n\n"
    " The following information may be useful for your response: "
    " Document Retrieval:\n{context}\n\n"
    " (Answer only from retrieval. Only cite sources that are used. Make your response conversational)"
    "\n\nUser Question: {input}"
)

docstore = FAISS.load_local(
    f"{HOME}/docstore_index", embedder, allow_dangerous_deserialization=True
)


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

## PRE-ASSESSMENT: Run as-is and see the basic chain in action

add_routes(
    app,
    instruct_llm,
    path="/basic_chat",
)

## ASSESSMENT TODO: Implement these components as appropriate

add_routes(
    app,
    RunnableLambda(lambda x: chat_prompt | instruct_llm | StrOutputParser()),
    path="/generator",
)

add_routes(
    app,
    RunnableLambda(lambda x: docstore.as_retriver()),
    path="/retriever",
)

## Might be encountered if this were for a standalone python file...
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9012)
