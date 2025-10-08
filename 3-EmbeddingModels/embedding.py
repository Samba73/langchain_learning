from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"
              , "to a vector using Hugging Face models."]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)