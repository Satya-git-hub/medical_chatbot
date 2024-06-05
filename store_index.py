from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

index_name = os.environ.get('INDEX')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

#Creating Embeddings for Each of The Text Chunks & storing
vectorstore_from_docs = PineconeVectorStore.from_documents(
        text_chunks,
        index_name=index_name,
        embedding=embeddings
    )