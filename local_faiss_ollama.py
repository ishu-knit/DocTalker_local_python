# (successfully run + giving ans to text only not other info)
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS    
from langchain.chains.question_answering import load_qa_chain
import os   
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
ollama_embeddings = OllamaEmbeddings(model="zephyr",base_url = 'http://127.0.0.1:11434')
model = Ollama(model="zephyr",base_url = 'http://127.0.0.1:11434')

doc_reader = PyPDFLoader("impromptu-rh.pdf")

docs = doc_reader.load_and_split()[0:5]
print("pdf load")

# create model
from langchain_community.llms import Ollama
model = Ollama(model="zephyr",base_url = 'http://127.0.0.1:11434')
# make vectorstore
vectorstore = FAISS.from_documents(docs,ollama_embeddings)
print("vectore stored")
# retreival qa
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(model,retriever=vectorstore.as_retriever())

print("query")

# query and its answer
query = "who is the shakur khan ?"
ans =chain.run({"query":query})


print("ans:-->", ans)