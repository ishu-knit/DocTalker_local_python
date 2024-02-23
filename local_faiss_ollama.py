# (successfully run + giving ans to text only not other info)
# (successfully run + giving ans to text only not other info)

from langchain_community.vectorstores import FAISS    
import os   
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama


ollama_embeddings = OllamaEmbeddings(model="zephyr",base_url = 'http://127.0.0.1:11434')
model = Ollama(model="zephyr",base_url = 'http://127.0.0.1:11434')

doc_reader = PyPDFLoader("pdfs/impromptu-rh.pdf")

docs = doc_reader.load_and_split()

# create model
from langchain_community.llms import Ollama
model = Ollama(model="zephyr",base_url = 'http://127.0.0.1:11434')
# make vectorstore
vectorstore = FAISS.from_documents(docs,ollama_embeddings)
print("vectore stored")
# retreival qa
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(model,retriever=vectorstore.as_retriever())

query = input("type your query:- ")
while query:
# query and its answer
    ans =chain.run({"query":query})
    print("ans:-->", ans)
    query = input("type your query to continue / press enter to end conversation :- ")
