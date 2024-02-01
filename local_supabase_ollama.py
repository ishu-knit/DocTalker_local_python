# # (successfully run and give accurtate answer probelm is it is anable to understand )


from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS    
from langchain_community.vectorstores import SupabaseVectorStore

from langchain.chains.question_answering import load_qa_chain
import os   
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from supabase import create_client, Client

supabase_client = create_client("https://oxkmkprtwkuiewajaqvx.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im94a21rcHJ0d2t1aWV3YWphcXZ4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDM4Mzc3MDEsImV4cCI6MjAxOTQxMzcwMX0.vWKFLz2r_RzuJ_ZnsMFvp0srckuMcjCiB7RkfhlGj_c")

ollama_embeddings = OllamaEmbeddings(model="luffy",base_url = 'http://127.0.0.1:11434')
model = Ollama(model="luffy",base_url = 'http://127.0.0.1:11434')

doc_reader = PyPDFLoader("impromptu-rh.pdf")
docs = doc_reader.load_and_split()[0:5]
print("pdf load")

# create model
from langchain_community.llms import Ollama
model = Ollama(model="luffy",base_url = 'http://127.0.0.1:11434')
# make vectorstore
# default is documents and match_documents for table_name , query_name respectively
# vectorstore = SupabaseVectorStore.from_documents(docs,ollama_embeddings,client=supabase_client,table_name="documents , query_name="match_documents")
# load database
vectorstore = SupabaseVectorStore(
            client=supabase_client,
            embedding=ollama_embeddings,
            table_name="documents",
            query_name="match_documents",
        )

query = "when was hiroshima attacked ?"
matched_docs = vectorstore.similarity_search(query,k=3)

print(matched_docs)
# retreival qa
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
print("query")
# query and its answer
query = input("type query:--")
ans =chain.invoke({"query":query})
print("ans:-->", ans)
# important  here internet is necessary becoz it used rpc function of the supabase to compare embeddings




# (successfully run and give  answer according to context )
#  using flask 

from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS 
from langchain_community.vectorstores import   SupabaseVectorStore
from langchain.chains.question_answering import load_qa_chain
import os   
from supabase import create_client
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from httpx import WriteTimeout
app = Flask(__name__)


# Initialize models
ollama_embeddings = OllamaEmbeddings(model="luffy",base_url = 'http://127.0.0.1:11434')
model = Ollama(model="luffy",base_url = 'http://127.0.0.1:11434',temperature=1)


doc_reader = PyPDFLoader("impromptu-rh.pdf")
docs = doc_reader.load_and_split()[0:5]

print("pdf load")
# Initialize client
supabase_client = create_client("https://oxkmkprtwkuiewajaqvx.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im94a21rcHJ0d2t1aWV3YWphcXZ4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDM4Mzc3MDEsImV4cCI6MjAxOTQxMzcwMX0.vWKFLz2r_RzuJ_ZnsMFvp0srckuMcjCiB7RkfhlGj_c")
vectorstore = SupabaseVectorStore(
                client=supabase_client,
                embedding=ollama_embeddings,
                table_name="documents",
                query_name="match_documents",
            )

retrieval_chain = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
# embedding
@app.route('/api/text_emd', methods=['POST'])
def create_embedding():
    try:
        data = request.get_json()
        text = data.get('text')
        # create_embedding + store in supabase
        vectorstore = SupabaseVectorStore.from_documents(docs, ollama_embeddings, client=supabase_client , table_name="documents")
        return "created!!"
    except WriteTimeout as e:
        # Handle timeout exception
        print(f"Write timeout error: {e}")
        return "Write timeout error"
    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")
        return "An error occurred"



# query
# API for querying (completed!!)
@app.route('/api/query', methods=['POST'])
def query():
    # extract complete json data
    data = request.get_json()
    # extract "query" key from json
    query_text = data.get('query')
    
     # Invoke the retrieval chain with the query
    result = retrieval_chain.invoke({"query": query_text})    
    return jsonify({'result': result})
    # return "ishu here"




if __name__ == '__main__':
    app.run(debug=True)

