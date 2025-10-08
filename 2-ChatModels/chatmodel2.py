from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that translates {input_language} to {output_language}.",
        ),
        ("human", "{input}"),
    ]
)

model = ChatOllama(model="llama3.2", temperature=0)
chain = prompt | model
response = chain.invoke(
    {
        "input_language": "English",
        "output_language": "Tamil",
        "input": "I love my country.",
    }
)

print(response.content)
