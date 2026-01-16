from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA
from load_vectorstore import load_vectorstore
from prompt import get_prompt

load_dotenv()

# Load vector database
vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10}
)

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

# Prompt
prompt = get_prompt()

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

print("🔬 Lab Report LLM Tester")
print("Type 'exit' to quit.\n")

while True:
    question = input("Ask a question: ")
    if question.lower() == "exit":
        break

    response = qa_chain.invoke({
    "question": question,
    "query": question
})


    print("\nAnswer:\n", response["result"])

    if "source_documents" in response:
        print("\nSources:")
        for doc in response["source_documents"]:
            print("-", doc.metadata.get("test", "unknown"))
