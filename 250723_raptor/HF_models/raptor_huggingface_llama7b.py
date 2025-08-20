import os
import csv
import io
import time
import json
import glob
from datetime import datetime
import gc
import torch
import nest_asyncio

from transformers import AutoTokenizer, BitsAndBytesConfig
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.packs.raptor import RaptorPack
from llama_index.core import SimpleDirectoryReader, Settings, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("num gpus =", torch.cuda.device_count())

# --- housekeeping ---
torch.cuda.empty_cache()
gc.collect()
nest_asyncio.apply()

# --- docs ---
folder = "/scratch1/mfp5696/250713_raptor/documents"
txt_files = glob.glob(f"{folder}/*.txt")
documents = SimpleDirectoryReader(input_files=txt_files).load_data()

# --- embedding model (384-dim) ---
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

# --- qdrant ---
client = QdrantClient(url="http://127.0.0.1:6336")
aclient = AsyncQdrantClient(url="http://127.0.0.1:6336")
# IMPORTANT: dimension must match the embedding model (MiniLM-L6-v2 = 384)
vector_store = QdrantVectorStore(
    client=client,
    aclient=aclient,
    collection_name="raptor",
    dimension=384,
)

# --- quantization (optional but recommended) ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# --- small model (for RAPTOR summaries/clustering) ---
SMALL_MODEL = "meta-llama/Llama-2-7b-chat-hf"
small_tok = AutoTokenizer.from_pretrained(SMALL_MODEL, trust_remote_code=True)
small_stop_ids = [small_tok.eos_token_id]
try:
    eot_id = small_tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != small_tok.unk_token_id:
        small_stop_ids.append(eot_id)
except Exception:
    pass

small_llm = HuggingFaceLLM(
    model_name=SMALL_MODEL,
    tokenizer_name=SMALL_MODEL,
    device_map="balanced_low_0",              
    model_kwargs={
        "quantization_config": quantization_config,
        "trust_remote_code": True,
    },
    generate_kwargs={"do_sample": True, "temperature": 0.1, "top_p": 0.9},
    stopping_ids=small_stop_ids,
)

# --- answering model ---
THINK_MODEL = "meta-llama/Llama-2-7b-chat-hf"
think_tok = AutoTokenizer.from_pretrained(THINK_MODEL, trust_remote_code=True)
think_stop_ids = [think_tok.eos_token_id]
try:
    eot_id = think_tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot_id, int) and eot_id != think_tok.unk_token_id:
        think_stop_ids.append(eot_id)
except Exception:
    pass

thinking_llm = HuggingFaceLLM(
    model_name=THINK_MODEL,
    tokenizer_name=THINK_MODEL,
    device_map="balanced_low_0",               
    model_kwargs={
        "quantization_config": quantization_config,
        "trust_remote_code": True,
    },
    generate_kwargs={"do_sample": True, "temperature": 0.1, "top_p": 0.9},
    stopping_ids=think_stop_ids,
)

# --- RAPTOR pack (use the pack's query_engine directly) ---
start = time.time()
raptor_pack = RaptorPack(
    documents=documents,
    embed_model=embed_model,
    llm=small_llm,               # used for building summaries/hierarchy
    vector_store=vector_store,   # storage
    similarity_top_k=10,
    mode="collapsed",
    transformations=[SentenceSplitter(chunk_size=2048, chunk_overlap=48)],
)
print(f"RaptorPack built in {time.time() - start:.2f}s")

synth = get_response_synthesizer(llm=thinking_llm, response_mode="compact")
query_engine = RetrieverQueryEngine(retriever=raptor_pack.retriever, response_synthesizer=synth)

if __name__ == "__main__":
    now = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print("Current Time =", now)

    # load QA
    qa_pairs = []
    truth_path = "/scratch1/mfp5696/250713_raptor/truth_set_v1_new.csv"
    with open(truth_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qa_pairs.append(
                {"question": row["Question"].strip(), "answer": row["Answer"].strip().strip('"')}
            )

    # query
    start = time.time()
    for qa in qa_pairs:
        prompt = qa["question"].rstrip() + " Answer concisely."
        response = query_engine.query(prompt)
        qa["respond"] = str(response).strip()
        # source nodes may not always be present; guard it
        try:
            qa["contexts"] = [node.text for node in (response.source_nodes or [])]
        except Exception:
            qa["contexts"] = []

    out_path = "/scratch1/mfp5696/250713_raptor/250723_raptor/raptor_huggingface_llama7b.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"Queries completed in {time.time() - start:.2f} seconds")
    now = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    print("Current Time =", now)

