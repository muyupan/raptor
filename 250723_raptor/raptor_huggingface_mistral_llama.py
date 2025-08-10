import os
import nltk
import csv
import io
import time
import json
import glob
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
from transformers import AutoTokenizer, AutoModelForCausalLM

from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.packs.raptor import RaptorPack
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.packs.raptor import RaptorRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# get_ipython().system('pip install llama-index ipywidgets')
# get_ipython().system('pip install llama-index llama-index-packs-raptor llama-index-vector-stores-chroma')
# get_ipython().system('pip install --upgrade transformers')

# optionally download the pack to inspect/modify it yourself!
#from llama_index.core.llama_pack import download_llama_pack
#RaptorPack = download_llama_pack("RaptorPack", "./raptor_pack")



#Reading documents from the directory
folder = "/data/hallucination/250726_muyu/250713_raptor/250710_documents_5"
txt_files = glob.glob(f"{folder}/*.txt")
documents = SimpleDirectoryReader(input_files=txt_files).load_data()

# Replace the ChromaDB setup with Qdrant setup:
client = QdrantClient(url="http://qdrant:6333")
aclient = AsyncQdrantClient(url="http://qdrant:6333")

vector_store = QdrantVectorStore(
    client=client,
    aclient=aclient,
    collection_name="raptor",
    dimension=768,  # nomic-embed-text dimension
)



small_model_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


small_llm = HuggingFaceLLM(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    model_kwargs={
        #"torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
		"quantization_config": quantization_config  
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.9,
    },
    tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    stopping_ids=stopping_ids,
)

nest_asyncio.apply()


thinking_model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]


thinking_llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    model_kwargs={
        #"torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
		"quantization_config": quantization_config  
    },
    generate_kwargs={
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.9,
    },
    tokenizer_name="meta-llama/Llama-3.3-70B-Instruct",
    stopping_ids=stopping_ids,
)

nest_asyncio.apply()

#RaptorPack initialization 
start = time.time()
raptor_pack = RaptorPack(
    documents, 
    embed_model=(model_name='sentence-transformers/all-MiniLM-L6-v2'),
    llm=small_llm,
    vector_store=vector_store,  # used for storage
    similarity_top_k=10,  # top k for each layer, or overall top-k for collapsed
    mode="collapsed",  # sets default mode
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    ],  # transformations applied for ingestion
)
total_time = time.time() - start
print(f"RaptorPack initialized in {total_time:.2f} seconds")


# This is used to used to show the retrieved text chunks, very useful
# nodes = raptor_pack.run("What countries have diesel submarines?", mode="collapsed")
# for i in range(len(nodes)):
    
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
    embed_model=(model_name='sentence-transformers/all-MiniLM-L6-v2'),
    llm=small_llm,
    vector_store=vector_store,  # used for storage
    similarity_top_k=10,  # top k for each layer, or overall top-k for collapsed
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
    llm=thinking_llm,
)

    
if __name__ == "__main__":

    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Current Time =", current_time)

    # main query
    start = time.time()
    qa_pairs = []
    with open ("/data/hallucination/250726_muyu/250713_raptor/truth_set_v1_new.csv", "r", encoding='utf-8-sig') as f:
        read = csv.DictReader(f)
        for row in read:
            qa_pairs.append({
                'question': row['Question'].strip(),
                'answer': row['Answer'].strip().strip('"'),
            })

    for qa in qa_pairs:
        response = query_engine.query(qa['question'] + " Answer concisely.")
        qa['respond'] = str(response).strip()
        qa['contexts'] = [node.text for node in response.source_nodes]

    with io.open (f"/data/hallucination/250726_muyu/250713_raptor/250723_raptor/deepseek_llama.json", "w") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    total_time = time.time() - start
    print(f"Queries completed in {total_time:.2f} seconds")

    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Current Time =", current_time)
