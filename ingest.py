import os
from dotenv import load_dotenv
import pymupdf  # pdf reader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from redisvl.index import SearchIndex

# Loading Env Vars
load_dotenv()
redis_url = os.environ.get("REDIS_URL")

folder_path = Path("DataMarine")  # folder containing the PDF files

for file_path in folder_path.glob("*.pdf"):
    doc = pymupdf.open(file_path)  # example