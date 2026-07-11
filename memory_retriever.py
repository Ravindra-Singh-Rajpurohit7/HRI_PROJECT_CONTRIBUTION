from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Yeh add karo top pe
import os
from dotenv import load_dotenv
load_dotenv()

# Constructor me yeh badlo:
class MemoryRetriever:
    def __init__(self, db_path, api_key):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=os.environ.get("GOOGLE_API_KEY")  # hardcoded key ki jagah
        )

        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

    def retrieve_memory(self, user_query):

        docs = self.vectorstore.similarity_search_with_score(
            user_query,
            k=1
        )

        best_doc, score = docs[0]

        memory_id = best_doc.metadata["id"]

        # ------------------------
        # Split memory into sentences
        # ------------------------

        paragraph = []

        raw_text = (
            best_doc.page_content
            .replace("\n", " ")
        )

        raw_sentences = raw_text.split(".")

        for sent in raw_sentences:

            sent = sent.strip()

            if len(sent) > 5:
                paragraph.append(sent)

        if len(paragraph) == 0:

            paragraph = [
                best_doc.page_content
            ]

        # ------------------------
        # Similarity
        # ------------------------

        similarity = 1 / (1 + score)

        # ------------------------
        # Find best sentence
        # ------------------------

        query_words = set(
            user_query.lower().split()
        )

        best_sentence = paragraph[0]
        best_sentence_score = 0

        for sent in paragraph:

            sent_words = set(
                sent.lower().split()
            )

            overlap = len(
                query_words.intersection(
                    sent_words
                )
            )

            current_score = (
                similarity
                + overlap * 0.05
            )

            if current_score > best_sentence_score:

                best_sentence_score = current_score
                best_sentence = sent

        # ------------------------
        # DEBUG PRINTS
        # ------------------------

      

        return (
            memory_id,
            paragraph,
            best_sentence,
            best_sentence_score,
            similarity,
            docs,
            best_doc.page_content
        )