# # (successfully run and give accurtate answer probelm is it is anable to understand )
# from PyPDF2 import PdfReader
# from langchain_openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# # from langchain_community.vectorstores import FAISS    
# from langchain_community.vectorstores import SupabaseVectorStore

# from langchain.chains.question_answering import load_qa_chain
# import os   
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama
# from supabase import create_client, Client


# supabase_client = create_client("https://oxkmkprtwkuiewajaqvx.supabase.co", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im94a21rcHJ0d2t1aWV3YWphcXZ4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDM4Mzc3MDEsImV4cCI6MjAxOTQxMzcwMX0.vWKFLz2r_RzuJ_ZnsMFvp0srckuMcjCiB7RkfhlGj_c")

# ollama_embeddings = OllamaEmbeddings(model="zephyr",base_url = 'http://127.0.0.1:11434')
# model = Ollama(model="zephyr",base_url = 'http://127.0.0.1:11434')

# doc_reader = PyPDFLoader("impromptu-rh.pdf")
# docs = doc_reader.load_and_split()[0:5]
# print("pdf load")

# # create model
# from langchain_community.llms import Ollama
# model = Ollama(model="zephyr",base_url = 'http://127.0.0.1:11434')
# # make vectorstore
# vectorstore = SupabaseVectorStore.from_documents(docs,ollama_embeddings,client=supabase_client)
# # load database
# # vectorstore = SupabaseVectorStore(
# #             client=supabase_client,
# #             embedding=ollama_embeddings,
# #             table_name="documents",
# #             query_name="match_documents",
# #         )
# print("vectore stored")
# # retreival qa
# from langchain.chains import RetrievalQA
# chain = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
# print("query")
# # query and its answer
# query = input("type query:--")
# ans =chain.invoke({"query":query})

# print("ans:-->", ans)




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
# vectorstore = SupabaseVectorStore.from_documents(docs,ollama_embeddings,client=supabase_client)
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
# from langchain.chains import RetrievalQA
# chain = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())
# print("query")
# # query and its answer
# query = input("type query:--")
# ans =chain.invoke({"query":query})


# print("ans:-->", ans)
