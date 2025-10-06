from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_xai import ChatXAI

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Replace with a secure key

# --- Document & Chain Initialization ---

# 1. Load the PDF (make sure the file is at static)
all_docs = []
pdf_folder = os.path.join("static", "uploads")
# Iterate over all files in the folder
for filename in os.listdir(pdf_folder):
    # Check if the file is a PDF (case-insensitive)
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        loader = PDFPlumberLoader(pdf_path)
        # loader.load() returns a list of document objects; extend the list
        all_docs.extend(loader.load())

# 3. Create the vector store from documents
index_dir = "faiss_index"  # Initialize the embedder
embedder = HuggingFaceEmbeddings()

if os.path.exists(index_dir):
    # Load the existing vector store
    vector = FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)
    print("Loaded vector store from disk.")
else:
    # 2. Split into chunks using SemanticChunker (which uses HuggingFaceEmbeddings)
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(all_docs)
    # Create the vector store from the documents
    vector = FAISS.from_documents(documents, embedder)
    # Save the vector store locally for future use
    vector.save_local(index_dir)
    print("Created and saved new vector store.")

retriever = vector.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# initialize the GROK llm 
llm= llm = ChatXAI(
    model="grok-2-latest",
    temperature=0.01,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="API HERE"
)

# 5. Define the prompt for QA
prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.

Context: {context}

Question: {question}

Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

# 6. Build the LLMChain
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)

# 7. Define a document prompt for combining retrieved documents
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent: {page_content}\nsource: {source}",
)

# 8. Create a StuffDocumentsChain to combine the documents
combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
)

# 9. Create the RetrievalQA chain with source documents returned
qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
)


def get_response(question):
    response = qa(question)
    answer_text = response.get("result", "I don't know")
    print("the response form the chain is ", response)
    pdf_url = None
    if response.get("source_documents") and len(response["source_documents"]) > 0:
        doc = response["source_documents"][0]
        metadata = doc.metadata
        source_doc = metadata.get("source", "")
        page_num = metadata.get("page", 0)
        # Normalize the path: convert backslashes to forward slashes.
        normalized_source = source_doc.replace("\\", "/")
        # Remove any leading "static/" if present.
        if normalized_source.lower().startswith("static/"):
            normalized_source = normalized_source[len("static/") :]
        # Generate a relative URL using Flask's static route.
        pdf_url = url_for("static", filename=normalized_source)
        # Append the page fragment.
        pdf_url = f"{pdf_url}#page={page_num+1}"
        print("returned urls is", pdf_url)
    return answer_text, pdf_url


# --- Flask Routes ---


# Landing page route
@app.route("/")
def landing():
    return render_template("landing.html")


# Chat page route
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        question = request.form.get("question", "")
        answer, pdf_url = get_response(question)
        return render_template(
            "chat.html", question=question, answer=answer, pdf_url=pdf_url
        )
    else:
        return render_template("chat.html", question="", answer="", pdf_url="")


if __name__ == "__main__":
    app.run(debug=False)
