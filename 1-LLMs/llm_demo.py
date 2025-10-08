from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """ Question: {question}
 Answer: Let's think step by step. """
prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.2")
chain = prompt | model
response = chain.invoke({"question": "What is the capital of France?"})
print(response)