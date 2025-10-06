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
from langchain_perplexity import ChatPerplexity

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your-secret-key-here"  # Replace with a secure key


# 1. Load PDFs from static/uploads folder
all_docs = []
pdf_folder = os.path.join("static", "uploads")

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        loader = PDFPlumberLoader(pdf_path)
        all_docs.extend(loader.load())

# 2. Initialize the embedder (✅ Explicit model name)
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index_dir = "faiss_index"

# 3. Create or load FAISS vector store
if os.path.exists(index_dir):
    vector = FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)
    print("Loaded vector store from disk.")
else:
    # ✅ Fix here — pass model_name explicitly to SemanticChunker
    text_splitter = SemanticChunker(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    documents = text_splitter.split_documents(all_docs)

    vector = FAISS.from_documents(documents, embedder)
    vector.save_local(index_dir)
    print("Created and saved new vector store.")

# 4. Build retriever
retriever = vector.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# 5. Initialize Perplexity LLM
llm = ChatPerplexity(
    model="sonar-deep-research",
    temperature=0.01,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="API HERE"
)

# 6. Define prompt
prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say "I don't know" but don't make up an answer.

Context: {context}

Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

# 7. Build LLMChain
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)

# 8. Define document combination logic
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent: {page_content}\nsource: {source}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
)

# 9. Create RetrievalQA chain
qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
)

# --- Helper Function ---
def get_response(question):
    response = qa(question)
    answer_text = response.get("result", "I don't know")
    print("Response from the chain:", response)

    pdf_url = None
    if response.get("source_documents"):
        doc = response["source_documents"][0]
        metadata = doc.metadata
        source_doc = metadata.get("source", "")
        page_num = metadata.get("page", 0)

        normalized_source = source_doc.replace("\\", "/")
        if normalized_source.lower().startswith("static/"):
            normalized_source = normalized_source[len("static/"):]
        pdf_url = url_for("static", filename=normalized_source)
        pdf_url = f"{pdf_url}#page={page_num + 1}"
        print("Returned PDF URL:", pdf_url)

    return answer_text, pdf_url

# --- Flask Routes ---
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        question = request.form.get("question", "")
        answer, pdf_url = get_response(question)
        return render_template("chat.html", question=question, answer=answer, pdf_url=pdf_url)
    else:
        return render_template("chat.html", question="", answer="", pdf_url="")

if __name__ == "__main__":
    app.run(debug=False)
