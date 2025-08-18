import os
import nltk

# Set NLTK data directory FIRST
os.environ['NLTK_DATA'] = '/scratch1/mfp5696/nltk_data'
nltk.data.path.append('/scratch1/mfp5696/nltk_data')
os.makedirs('/scratch1/mfp5696/nltk_data', exist_ok=True)

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("Current Time =", current_time)
# get_ipython().system('pip install llama-index ipywidgets')
# get_ipython().system('pip install llama-index llama-index-packs-raptor llama-index-vector-stores-chroma')
# get_ipython().system('pip install --upgrade transformers')
from llama_index.packs.raptor import RaptorPack

#import sys
#sys.setrecursionlimit(5000) 

# optionally download the pack to inspect/modify it yourself!
#from llama_index.core.llama_pack import download_llama_pack
#RaptorPack = download_llama_pack("RaptorPack", "./raptor_pack")

import nest_asyncio
nest_asyncio.apply()
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_dir="/scratch1/mfp5696/250713_raptor/250710_documents_truncated_20",
    recursive=True,
)

documents = []
for docs in reader.iter_data():
    for doc in docs:
        documents.append(doc)

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
import gc
import torch
# Add this before creating RAPTOR
torch.cuda.empty_cache()
gc.collect()

from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding



#Creates a database that persists on disk
client = chromadb.PersistentClient(path="./raptor_paper_db")

#Creates a named collection "raptor"
collection = client.get_or_create_collection("raptor")

#LlamaIndex wrapper around Chroma, provides unified interface for vector operations
vector_store = ChromaVectorStore(chroma_collection=collection)

selected_model = "llama3.1:8b"

# Replace the RaptorPack initialization ------------------------------------------------------------------------------------------------------------------------------------------

raptor_pack = RaptorPack(
    documents, 
    embed_model=OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
    ),  
    llm=Ollama(
        model=selected_model,
        base_url="http://localhost:11434",
        temperature=0.1,
        context_window=4096,
        request_timeout=120.0,
    ),
    vector_store=vector_store,  # used for storage
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="collapsed",  # sets default mode
    transformations=[
        SentenceSplitter(chunk_size=400, chunk_overlap=50)
    ],  # transformations applied for ingestion
)

nodes = raptor_pack.run("What countries have diesel submarines?", mode="collapsed")
print(len(nodes))
print(nodes[0].text)

nodes = raptor_pack.run(
    "What countries have diesel submarines?", mode="tree_traversal"
)
print(len(nodes))
print(nodes[0].text)

from llama_index.packs.raptor import RaptorRetriever

# Update the retriever-------------------------------------------------------------------------------- 

retriever = RaptorRetriever(
    [],
    embed_model=OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    ),  # used for embedding clusters
    llm=Ollama(
        model=selected_model,
        base_url="http://localhost:11434",
        temperature=0.1,
        context_window=4096,
        request_timeout=120.0,
    ),
    vector_store=vector_store,  # used for storage
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="tree_traversal",  # sets default mode
)

# if using a default vector store
# retriever.persist("./persist")
# retriever = RaptorRetriever.from_persist_dir("./persist", ...)


# ## Query Engine-------------------------------------------------------------------------------- 

from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(
    retriever, 
    llm=Ollama(
        model=selected_model,
        base_url="http://localhost:11434",
        temperature=0.1,
        context_window=4096,
        request_timeout=120.0,
    )
)

# main-------------------------------------------------------------------------------- 



response = query_engine.query("What countries have diesel submarines?")
print(str(response))





now = datetime.now()
current_time = now.strftime("%Y-%m-%d %H:%M:%S")
print("Current Time =", current_time)
