from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from pinecone import Pinecone
import tempfile, os
from fastapi import Body
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from app.db import db
from bson import ObjectId
router = APIRouter(prefix="/Rag", tags=["RAG & Domains"])


domains_collection = db["domains"]
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# ------------------ Helper ------------------
def process_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDFLoader"""
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()
    return "\n".join([page.page_content for page in pages])
def serialize(doc):
    doc["_id"] = str(doc["_id"])
    return doc
# ------------------ Upload PDF ------------------
@router.post("/upload-pdf/")
async def upload_pdf(domain_id: str = Form(...), file: UploadFile = File(...)):
    """
    Upload PDF -> Extract Text -> Generate Embeddings -> Store in Pinecone by Domain ID
    """

    # Validate PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Convert domain_id string to ObjectId
    try:
        domain = domains_collection.find_one({"_id": ObjectId(domain_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid domain_id format")

    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found.")

    index_name = domain.get("name")
    print("Index Name:", index_name)
    if not index_name:
        raise HTTPException(status_code=500, detail="No index assigned to this domain.")

    # Init Pinecone vector store for this domain
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name

    try:
        # Extract text
        document_text = process_pdf(tmp_path)
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF.")

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(document_text)

        # Store in Pinecone
        vector_store.add_texts(
            texts=texts,
            metadatas=[{"domain_id": domain_id, "source": file.filename}] * len(texts),
        )

        return {"status": "success", "message": f"PDF '{file.filename}' indexed under domain {domain_id}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    """
    Upload PDF -> Extract Text -> Generate Embeddings -> Store in Pinecone by Domain ID
    """

    # Validate PDF
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Find domain & index name from Mongo
    domain = domains_collection.find_one({"_id": domain_id})
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found.")

    index_name = domain.get("index_name")
    if not index_name:
        raise HTTPException(status_code=500, detail="No index assigned to this domain.")

    # Init Pinecone vector store for this domain
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name

    try:
        # Extract text
        document_text = process_pdf(tmp_path)
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF.")

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(document_text)

        # Store in Pinecone
        vector_store.add_texts(
            texts=texts,
            metadatas=[{"domain_id": domain_id, "source": file.filename}] * len(texts),
        )

        return {"status": "success", "message": f"PDF '{file.filename}' indexed under domain {domain_id}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


from fastapi import Form
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

@router.post("/chat-direct/")
async def chat_direct(
    domain_id: str = Form(...),
    query: str = Form(...)
):
    """
    Direct Pinecone query -> Retrieve top matches -> Generate Arabic answer using LLM
    """

    # Get domain index
    try:
        domain = domains_collection.find_one({"_id": ObjectId(domain_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid domain_id format")

    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found.")

    index_name = domain.get("name")
    index = pc.Index(index_name)

    # Create embedding vector for the query
    query_vector = embeddings.embed_query(query)

    # Query Pinecone directly
    try:
        query_response = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {str(e)}")

    # Extract context text and create Document objects
    matches = query_response.get("matches", [])
    if not matches:
        return {"status": "success", "answer": "لا أعلم", "sources": []}

    # Create Document objects from matches
    documents = []
    sources = []
    for match in matches:
        text = match["metadata"].get("text", "")
        source = match["metadata"].get("source")
        if text:
            documents.append(Document(page_content=text, metadata=match["metadata"]))
        if source:
            sources.append(source)

    # Arabic Prompt
    prompt_template = """
Use the following content from the documents to answer the question below in detail and in the same language.
If the answer is not available in the content, simply say "I don't know."

Question:
{question}

Context:
{context}

Answer:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    # Build chain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Get LLM response with Document objects
    response = qa_chain.invoke({"context": documents, "question": query})
    final_answer = response if isinstance(response, str) else str(response)

    return {
        "status": "success",
        "query": query,
        "answer": final_answer.strip(),
        "sources": list(set(filter(None, sources)))
    }
