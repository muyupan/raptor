
# get_ipython().run_line_magic('pip', 'install llama-index-llms-huggingface')
# get_ipython().run_line_magic('pip', 'install llama-index-embeddings-huggingface')
# get_ipython().system('pip install llama-index ipywidgets')
# get_ipython().system('pip install llama-index llama-index-packs-raptor llama-index-vector-stores-chroma')
# get_ipython().system('pip install --upgrade transformers')
#!pip install llama-index-embeddings-huggingface
from llama_index.packs.raptor import RaptorPack

import sys
sys.setrecursionlimit(5000) 

# optionally download the pack to inspect/modify it yourself!
# from llama_index.core.llama_pack import download_llama_pack
# RaptorPack = download_llama_pack("RaptorPack", "./raptor_pack")
import nest_asyncio
nest_asyncio.apply()
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_dir="/storage/home/mfp5696/vxk_group/250630_nlp_hallucination/250630_raptor/250710_documents_20",
    recursive=True,
)

documents = []
for docs in reader.iter_data():
    for doc in docs:
        documents.append(doc)

# reader = SimpleDirectoryReader(
#     input_dir="/storage/home/mfp5696/vxk_group/250630_nlp_hallucination/250630_raptor/documents",
#     recursive=True,
# )
# documents = reader.load_data()

# # Take 70%
# num_docs = int(len(documents) * 0.7)
# documents = documents[:num_docs]

# print(len(documents))

# from llama_index.core.node_parser import SentenceSplitter
# Splits documents into smaller chunks

from llama_index.core.node_parser import SentenceSplitter
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import gc
import torch
# Add this before creating RAPTOR
torch.cuda.empty_cache()
gc.collect()

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM
import chromadb

#Creates a database that persists on disk
client = chromadb.PersistentClient(path="./raptor_paper_db")

#Creates a named collection "raptor"
collection = client.get_or_create_collection("raptor")

#LlamaIndex wrapper around Chroma, provides unified interface for vector operations
vector_store = ChromaVectorStore(chroma_collection=collection)

#Llama models
LLAMA2_7B = "meta-llama/Llama-2-7b-hf"
LLAMA2_7B_CHAT = "meta-llama/Llama-2-7b-chat-hf"
LLAMA2_13B = "meta-llama/Llama-2-13b-hf"
LLAMA2_13B_CHAT = "meta-llama/Llama-2-13b-chat-hf"
LLAMA2_70B = "meta-llama/Llama-2-70b-hf"
LLAMA2_70B_CHAT = "meta-llama/Llama-2-70b-chat-hf"

selected_model = LLAMA2_7B_CHAT

raptor_pack = RaptorPack(
    documents, 
    embed_model=HuggingFaceEmbedding(
        model_name="intfloat/e5-base-v2",
        query_instruction="query: ", # used for embedding queries E5 models
        text_instruction="passage: "
    ),  # used for embedding clusters
    #llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),  # used for generating summaries
    llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.1},
    #query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto",
    # change these settings below depending on your GPU
    #model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
),
    vector_store=vector_store,  # used for storage
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="collapsed",  # sets default mode
    transformations=[
        SentenceSplitter(chunk_size=400, chunk_overlap=50)
        #SentenceSplitter(chunk_size=2000, chunk_overlap=200)
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

retriever = RaptorRetriever(
    [],
    # embed_model=OpenAIEmbedding(
    #     model="text-embedding-3-small"
    # ),  # used for embedding clusters
    # llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),  # used for generating summaries
    embed_model=HuggingFaceEmbedding(
        model_name="intfloat/e5-base-v2",
        query_instruction="query: ", # used for embedding queries E5 models
        text_instruction="passage: "
    ),  # used for embedding clusters
    #llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1),  # used for generating summaries
    llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.1},
    #query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto",
    # change these settings below depending on your GPU
    #model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
),
    vector_store=vector_store,  # used for storage
    similarity_top_k=2,  # top k for each layer, or overall top-k for collapsed
    mode="tree_traversal",  # sets default mode
)

# if using a default vector store
# retriever.persist("./persist")
# retriever = RaptorRetriever.from_persist_dir("./persist", ...)


# ## Query Engine

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever, llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.1},
    #query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto",
    # change these settings below depending on your GPU
    #model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
)
)

response = query_engine.query("What countries have diesel submarines?")
print(str(response))

