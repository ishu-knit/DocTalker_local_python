# successfully run 

from langchain_community.llms import Ollama
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS    
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever


                            #  pdf reader
                            # 1st way  

# doc_reader = PdfReader('unit4.pdf')
# raw_text = ''
# lt=[]
# for i, page in enumerate(doc_reader.pages):
#     text = page.extract_text()
#     if text:
#         lt.append(text)
#         raw_text += text
#         print("index--->",i,text)

# text_splitter = CharacterTextSplitter(
#     separator = "\n",
#     chunk_size = 1000,
#     chunk_overlap  = 200, #striding over the text
#     length_function = len,
# )

# texts = text_splitter.split_text(raw_text)

                        # PdfReader 
                        # 2nd way 
                        #active

doc_reader = PyPDFLoader("unit4.pdf")
docs = doc_reader.load_and_split()


                        # embedding model  
ollama_embedding = OllamaEmbeddings(model="luffy",base_url = 'http://127.0.0.1:11434')

                        # create embedding
vectorstore = FAISS.from_documents(docs, ollama_embedding)

                        # store database (optional)
# vectorstore.save_local("database_docs")


                         #load the database (if stored previously)
# vectorstore = FAISS.load_local("database_docs",ollama_embedding)

                                #query
query = "what is this book about ?"

                                # chain
model = Ollama(model="luffy",base_url = 'http://127.0.0.1:11434')


retriever = VectorStoreRetriever(vectorstore=vectorstore)
chain = RetrievalQA.from_llm(model, retriever=retriever)

                                    # ans
ans =chain.invoke({"query":query})
print(ans)



#  important points


# 1.>
    #  vectorstore = FAISS.from_documents(docs, ollama_embedding)
    # if you have a string  like docs= "my name is ishu" then select 1st way pdf reader
    #  and also replace from_documents to from_texts
    # then no of vectors create is no of char+space this string have 
    # here no of vectors = 12
    # but if you have docs = pdf then 1st select 2nd way  pdf reader 
    # and also  use from_documents in faiss vectorstore

# 2.> 
    # dimension  4095 due to use of zephyr model instead of text_embedding-->1539