from llama_index.packs.raptor import RaptorPack
import os
import nest_asyncio
from llama_index.core import SimpleDirectoryReader
import glob

from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import chromadb
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/data/hallucination/HF_models/Llama-3.3-70B-Instruct")

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


llm = HuggingFaceLLM(
    model_name="/data/hallucination/HF_models/Llama-3.3-70B-Instruct",
    model_kwargs={
        "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
        # "quantization_config": quantization_config
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.9,
    },
    tokenizer_name="/data/hallucination/HF_models/Llama-3.3-70B-Instruct",
    stopping_ids=stopping_ids,
)

nest_asyncio.apply()


folder = "/data/hallucination/documents"
txt_files = glob.glob(f"{folder}/*.txt")


documents = SimpleDirectoryReader(input_files=txt_files).load_data()
client = chromadb.PersistentClient(path="./raptor_paper_db")
collection = client.get_or_create_collection("raptor")

vector_store = ChromaVectorStore(chroma_collection=collection)

raptor_pack = RaptorPack(
    documents,
    embed_model=HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2'),  # used for embedding clusters
    llm=llm,  # used for generating summaries
    vector_store=vector_store,  # used for storage
    similarity_top_k=20,  # top k for each layer, or overall top-k for collapsed
    mode="collapsed",  # sets default mode
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    ],  # transformations applied for ingestion
    verbose = True
)
