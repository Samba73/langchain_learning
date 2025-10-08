from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
from ollama import embed
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership skills.",
    "MS Dhoni is a former Indian cricketer and captain of the Indian national team, known for his calm demeanor and finishing abilities.",
    "Sachin Tendulkar is a legendary Indian cricketer, often referred to as the 'God of Cricket' for his numerous records and contributions to the sport.",
    "Rohit Sharma is an Indian cricketer known for his elegant batting style and ability to score big hundreds.",
    "Anil Kumble is a former Indian cricketer and one of the greatest leg-spinners in the history of cricket.",
    "Jasprit Bumrah is an Indian cricketer known for his unique bowling action and ability to bowl yorkers at will."
]
query = "Who is Jasprit Bumrah?"

doc_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)
print(f"doc embdeddings value: {doc_embeddings[5]}")
similarities = cosine_similarity([query_embedding], doc_embeddings)
print(f"Similarities: {similarities}")
most_similar_doc_index = np.argmax(similarities)
print(f"Most Similar doc index: {most_similar_doc_index}")
print("Most similar document:", documents[most_similar_doc_index])