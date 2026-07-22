import os
from dotenv import load_dotenv
import pymupdf  # pdf reader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from redisvl.index import SearchIndex
from classes import ChunkMetadata

# Loading Env Vars
load_dotenv()
redis_url = os.environ.get("REDIS_URL")

folder_path = Path("DataMarine")  # folder containing the PDF files

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,     # for continued context
    length_function=len,    # measures by char count. Can also set to token_len
)

for file_path in folder_path.glob("*.pdf"):
    doc = pymupdf.open(file_path)  # example


