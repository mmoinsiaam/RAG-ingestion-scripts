import os
from dotenv import load_dotenv
import pymupdf  # pdf reader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from redisvl.index import SearchIndex
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class HabitatType(str, Enum):
    REEF = "reef"
    PELAGIC = "pelagic"
    BENTHIC = "benthic"
    ESTUARINE = "estuarine"
    POLAR = "polar"
    DEEP_SEA = "deep_sea"


class ConservationStatus(str, Enum):
    LC = "LC"  # Least Concern
    NT = "NT"  # Near Threatened
    VU = "VU"  # Vulnerable
    EN = "EN"  # Endangered
    CR = "CR"  # Critically Endangered
    EX = "EX"  # Extinct
    UNKNOWN = "unknown"  # chunk doesn't state a status

class TopicCategory(str, Enum):
    BEHAVIOR = "behavior"
    PHYSIOLOGY = "physiology"
    CONSERVATION = "conservation"
    TAXONOMY = "taxonomy"
    ECOLOGY = "ecology"
    REPRODUCTION = "reproduction"

class ChunkMetadata(BaseModel):
    """Structured fields the LLM extracts per chunk. Passed as the
    response_format schema to OpenAI's structured outputs."""

    species_name: list[str] = Field(
        default_factory=list,
        description="Species names explicitly mentioned in this chunk, common or scientific name."
    )
    taxonomic_class: Optional[str] = Field(
        default=None,
        description="Taxonomic class if identifiable, e.g. Elasmobranchii, Cephalopoda, Actinopterygii."
    )
    habitat_type: Optional[HabitatType] = None
    conservation_status: ConservationStatus = ConservationStatus.UNKNOWN
    geographic_region: Optional[str] = Field(
        default=None,
        description="Ocean basin or named region, e.g. North Atlantic, Coral Triangle."
    )
    topic_category: Optional[TopicCategory] = None
    publication_year: Optional[int] = Field(
        default=None,
        description="Year the source paper was published, if stated."
    )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,     # for continued context
    length_function=len,
)

# Loading Env Vars
load_dotenv()
redis_url = os.environ.get("REDIS_URL")

folder_path = Path("DataMarine")  # folder containing the PDF files

for file_path in folder_path.glob("*.pdf"):
    doc = pymupdf.open(file_path)  # example


