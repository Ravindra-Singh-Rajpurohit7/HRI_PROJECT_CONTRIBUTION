import json
import time
import shutil

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("GOOGLE_API_KEY")
DB_PATH = "./chroma_db"


# -------------------------
# LOAD JSON
# -------------------------

with open(
    "expanded_memories.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# -------------------------
# DELETE OLD DB
# -------------------------

try:
    shutil.rmtree(DB_PATH)
    print("Old Chroma DB removed")
except:
    pass


# -------------------------
# EMBEDDINGS
# -------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    google_api_key=API_KEY
)


# -------------------------
# BUILD DOCUMENTS
# -------------------------

documents = []

for topic_name, memories in data.items():

    print(f"\nTopic: {topic_name}")

    for memory in memories:

        full_text = "\n".join(
            memory["text"]
        )

        doc = Document(
            page_content=full_text,
            metadata={
                "id": memory["id"],
                "topic": topic_name
            }
        )

        documents.append(doc)

        print(
            f"Added -> {memory['id']}"
        )


print(
    f"\nTotal Documents: {len(documents)}"
)


# -------------------------
# CREATE VECTORSTORE
# -------------------------

vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)


# -------------------------
# SAFE INSERT
# -------------------------

BATCH_SIZE = 20

for i in range(
    0,
    len(documents),
    BATCH_SIZE
):

    batch = documents[
        i:i + BATCH_SIZE
    ]

    vectorstore.add_documents(
        batch
    )

    print(
        f"Inserted {i + len(batch)} / {len(documents)}"
    )

    # Gemini embedding limit avoid
    time.sleep(15)


print(
    "\nChroma DB Created Successfully."
)