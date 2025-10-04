from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from pinecone import Pinecone
import tempfile, os
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