import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pymupdf  # pdf reader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from redisvl.index import SearchIndex
from classes import ChunkMetadata
from redisvl.index import SearchIndex

# Extract, transform to ChunkMetadata, embed to vector, load to schema and upload to Redis

# Loading Env Vars
load_dotenv()
redis_url = os.environ.get("REDIS_URL")
openai_api_key = os.environ.get("OPEN_AI_API_KEY")

client = AsyncOpenAI(api_key=openai_api_key)
folder_path = Path("DataMarine")  # folder containing the PDF files

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,     # for continued context
    length_function=len,    # measures by char count. Can also set to token_len
)

def get_index(redis_url: str) -> SearchIndex:
    index = SearchIndex.from_yaml("index.yaml")
    index.connect(redis_url)
    index.create(overwrite=False)  # no-op if index already exists
    return index

def extract_text_from_pdf(file_path: Path) -> str:
    doc = pymupdf.open(file_path)
    raw_text = "\n".join(page.get_text() for page in doc)  # extract text from all pages
    doc.close()
    return raw_text

async def extract_metadata(chunk_text: str) -> ChunkMetadata:
    response = await client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract structured metadata about marine biology content "
                    "from the given text chunk. Only extract what's explicitly "
                    "stated — leave fields empty/unknown rather than guessing."
                ),
            },
            {"role": "user", "content": chunk_text},
        ],
        response_format=ChunkMetadata,
    )
    return response.choices[0].message.parsed

async def embed_chunk(chunk_text: str) -> list[float]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk_text,
    )
    return response.data[0].embedding

#combine extracting metadata and embedding into one function to run concurrently
async def process_chunk(source_doc: str, chunk_index: int, chunk_text: str) -> dict:
    #uses asyncio.gather to run both functions concurrently for efficiency
    metadata, embedding = await asyncio.gather(
        extract_metadata(chunk_text),
        embed_chunk(chunk_text),
    )

    #formatting into payload that matches schema
    payload = {
        "chunk_id": f"{source_doc}_{chunk_index}",
        "content": chunk_text,
        "source_doc": source_doc,
        "chunk_index": chunk_index,
        "species_name": metadata.species_name,
        "taxonomic_class": metadata.taxonomic_class or "",
        "habitat_type": metadata.habitat_type.value if metadata.habitat_type else "",
        "conservation_status": metadata.conservation_status.value,
        "geographic_region": metadata.geographic_region or "",
        "topic_category": metadata.topic_category.value if metadata.topic_category else "",
        "publication_year": metadata.publication_year or 0,
        "embedding": embedding,
    }
    return payload

async def ingest_pdf(source_doc: str, pdf_path: str) -> list[dict]:
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = splitter.split_text(raw_text)

    # creating list of coroutines for processing each chunk concurrently
    tasks = [
        process_chunk(source_doc, idx, chunk_text) for idx, chunk_text in enumerate(chunks)
    ]
    payloads = await asyncio.gather(*tasks)
    print(f"  {source_doc}: {len(payloads)} chunks processed")
    return payloads


async def main():
    index = get_index(redis_url)

    all_payloads = []
    for pdf_path in folder_path.glob("*.pdf"):
        source_doc = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Processing {source_doc}...")
        payloads = await ingest_pdf(source_doc, pdf_path)
        all_payloads.extend(payloads)

    print(f"Loading {len(all_payloads)} total chunks into Redis...")
    index.load(all_payloads, id_field="chunk_id")
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())


