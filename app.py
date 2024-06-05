from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import *
import os 
#from threadpoolctl import threadpool_limits

app = Flask(__name__)

load_dotenv()

# Set environment variables to mitigate conflicts
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["KMP_INIT_AT_FORK"] = "FALSE"

index_name = os.environ.get('INDEX')
pinecone_api_key = os.environ.get('PINECONE_API_KEY')

embeddings = download_hugging_face_embeddings()

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

#Prompt template creation
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

#LLM object creation
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

#QA object created 
qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    #with threadpool_limits(limits=1, user_api='openmp'):
    return render_template("chat.html")

@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print("Response: ", result["result"])
    return str(result["result"])

if __name__ == "__main__":
    #with threadpool_limits(limits=1, user_api='openmp'):
    app.run(host = "0.0.0.0", port = 8080, debug = True)