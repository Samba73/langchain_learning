from langchain_huggingface import HuggingFaceEndpointEmbeddings
from torch import embedding

embedding = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2")
# sentences = ["This is an example sentence", "Each sentence is converted"]
text = "Delhi is the capital of India."
# embeddings = embedding.embed_documents(sentences)
embeddings = embedding.embed_query(text)
# print(embeddings[0])
# print("===========")
# print(embeddings[1])
print(embeddings)