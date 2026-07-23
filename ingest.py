import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pymupdf  # pdf reader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from redisvl.index import SearchIndex
from classes import ChunkMetadata
import random

# Extract, transform to ChunkMetadata, embed to vector, load to schema and upload to Redis

# Loading Env Vars
load_dotenv(override=True)
redis_url = os.environ.get("REDIS_URL")
openai_api_key = os.environ.get("OPEN_AI_API_KEY")

client = AsyncOpenAI(api_key=openai_api_key)
folder_path = Path("DataMarine")  # folder containing the PDF files

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,     # for continued context
    length_function=len,    # measures by char count. Can also set to token_len
)

semaphore = asyncio.Semaphore(5)

async def retry_request(func, retries=5):
    for attempt in range(retries):
        try:
            return await func()

        except Exception as e:
            # Only retry rate limit errors
            if "429" in str(e):
                wait_time = (2 ** attempt) + random.random()

                print(f"Rate limited. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)

            else:
                raise e

    raise Exception("Request failed after retries")

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

    async def request():
        return await client.chat.completions.parse(
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

    response = await retry_request(request)

    return response.choices[0].message.parsed

async def embed_chunk(chunk_text: str) -> list[float]:

    async def request():
        return await client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk_text,
        )
    response = await retry_request(request)

    return response.data[0].embedding

#combine extracting metadata and embedding into one function to run concurrently
async def process_chunk(source_doc: str, chunk_index: int, chunk_text: str) -> dict:
    #uses asyncio.gather to run both functions concurrently for efficiency
    async with semaphore:
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

    print(f"{source_doc}: {len(chunks)} chunks")
    # creating list of coroutines for processing each chunk concurrently
    tasks = [
        process_chunk(source_doc, idx, chunk_text) for idx, chunk_text in enumerate(chunks)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    payloads = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Chunk failed: {result}")
        else:
            payloads.append(result)

    print(f"  {source_doc}: {len(payloads)} chunks processed")
    return payloads


async def main():
    index = get_index(redis_url)
    i = 0
    for pdf_path in folder_path.glob("*.pdf"):
        i += 1
        source_doc = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Processing {source_doc}...")
        payloads = await ingest_pdf(source_doc, pdf_path)
        index.load(payloads, id_field="chunk_id")

    print(f"Loaded {i} total pdfs into Redis...")
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())


