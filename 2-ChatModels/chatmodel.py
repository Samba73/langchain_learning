from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

template = """ Question: {question}
 """
prompt = ChatPromptTemplate.from_template(template)
model = ChatOllama(model="llama3.2", temperature=0)
chain = prompt | model
response = chain.invoke({"question": "What is the capital of France?"})
print(response.content)