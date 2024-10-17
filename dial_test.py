#!/usr/bin/env python3
import os

from dotenv import load_dotenv
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

llm = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4-turbo"),
    deployment_name=os.getenv("AZURE_OPENAI_MODEL"),
    temperature=0.8,  # just for the pirate
)

embed_model = AzureOpenAIEmbedding(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    model=os.getenv("AZURE_OPENAI_EMBEDDER_MODEL", "text-embedding-ada-002"),
    deployment_name=os.getenv("AZURE_OPENAI_EMBEDDER_NAME"),
)

Settings.llm = llm
Settings.embed_model = embed_model

messages = [
    ChatMessage(
        role=MessageRole.SYSTEM, content="You are a pirate with colorful personality."
    ),
    ChatMessage(role=MessageRole.USER, content="Hello"),
]

response = llm.chat(messages)
print(response)

documents = SimpleDirectoryReader(
    input_files=["./markdowns/hhgttg-intro.md"]
).load_data()
index = VectorStoreIndex.from_documents(documents)

QUERY = "Was it a good idea to come down from the trees?"
query_engine = index.as_query_engine()
answer = query_engine.query(QUERY)

print(answer.get_formatted_sources())
print("Query was:", QUERY)
print("Answer is:", answer)
