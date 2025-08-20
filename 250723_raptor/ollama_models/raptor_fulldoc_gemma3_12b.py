import os
import nltk
import csv
import io
import time
import json
os.environ['NLTK_DATA'] = '/scratch1/mfp5696/nltk_data'
nltk.data.path.append('/scratch1/mfp5696/nltk_data')
os.makedirs('/scratch1/mfp5696/nltk_data', exist_ok=True)
from datetime import datetime
import gc
import torch
torch.cuda.empty_cache()
gc.collect()
import nest_asyncio
nest_asyncio.apply()

from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.packs.raptor import RaptorPack
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.packs.raptor import RaptorRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# get_ipython().system('pip install llama-index ipywidgets')
# get_ipython().system('pip install llama-index llama-index-packs-raptor llama-index-vector-stores-chroma')
# get_ipython().system('pip install --upgrade transformers')

# optionally download the pack to inspect/modify it yourself!
#from llama_index.core.llama_pack import download_llama_pack
#RaptorPack = download_llama_pack("RaptorPack", "./raptor_pack")

#from llama_index.vector_stores.chroma import ChromaVectorStore
#import chromadb

def main():


    #Reading documents from the directory
    reader = SimpleDirectoryReader(
    input_dir="/scratch1/mfp5696/250713_raptor/documents",
    recursive=True,
    )

    documents = []
    for docs in reader.iter_data():
        for doc in docs:
            documents.append(doc)

    # Replace the ChromaDB setup with Qdrant setup:
    client = QdrantClient(url="http://localhost:6333")
    aclient = AsyncQdrantClient(url="http://localhost:6333")

    vector_store = QdrantVectorStore(
        client=client,
        aclient=aclient,
        collection_name="raptor",
        dimension=768,  # nomic-embed-text dimension
    )

    #select model
    selected_model = "gemma3:12b"


    #RaptorPack initialization 
    start = time.time()
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
            request_timeout=259200.0,
        ),
        vector_store=vector_store,  # used for storage
        similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
        mode="collapsed",  # sets default mode
        transformations=[
            SentenceSplitter(chunk_size=400, chunk_overlap=50)
        ],  # transformations applied for ingestion
    )
    total_time = time.time() - start
    print(f"RaptorPack initialized in {total_time:.2f} seconds")


    # This is used to used to show the retrieved text chunks, very useful
    # nodes = raptor_pack.run("What countries have diesel submarines?", mode="collapsed")
    # print(len(nodes))
    # print(nodes[0].text)

    # nodes = raptor_pack.run(
    #     "What countries have diesel submarines?", mode="tree_traversal"
    # )
    # print(len(nodes))
    # print(nodes[0].text)


    # Update the retriever
    start = time.time()
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
            request_timeout=259200.0,
        ),
        vector_store=vector_store,  # used for storage
        similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
        mode="collapsed",  # sets default mode
    )
    total_time = time.time() - start
    print(f"RaptorRetriever initialized in {total_time:.2f} seconds")

    # if using a default vector store
    # retriever.persist("./persist")
    # retriever = RaptorRetriever.from_persist_dir("./persist", ...)


    #Query Engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever, 
        llm=Ollama(
            model=selected_model,
            base_url="http://localhost:11434",
            temperature=0.1,
            context_window=4096,
            request_timeout=259200.0,
        )
    )


    # main query
    start = time.time()
    qa_pairs = []
    with open ("/scratch1/mfp5696/250713_raptor/truth_set_v1_new.csv", "r", encoding='utf-8-sig') as f:
        read = csv.DictReader(f)
        for row in read:
            qa_pairs.append({
                'question': row['Question'].strip(),
                'answer': row['Answer'].strip().strip('"'),
            })

    for qa in qa_pairs:
        response = query_engine.query(qa['question'] + " Answer concisely.")
        qa['respond'] = str(response).strip()
   
    with io.open (f"/scratch1/mfp5696/250713_raptor/250723_raptor/raptor_fulldoc_gemma3_12b.json", "w") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    total_time = time.time() - start
    print(f"Queries completed in {total_time:.2f} seconds")

    

if __name__ == "__main__":

    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Current Time =", current_time)

    main()

    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Current Time =", current_time)