# *****important instruction in the end *****


# to download the pdf from internet 

# import requests
# url = "https://www.impromptubook.com/wp-content/uploads/2023/03/impromptu-rh.pdf"
# response = requests.get(url)
# with open("impromptu-rh.pdf", "wb") as pdf_file:
#     pdf_file.write(response.content)
# print("Download complete.")


from langchain_community.llms import Ollama
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS    
from langchain.chains.question_answering import load_qa_chain
import os   


# location of the pdf .
doc_reader = PdfReader('impromptu-rh.pdf')


# to extract text from the pdf
raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text


# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200, #striding over the text
    length_function = len,
)

# raw text into list of smaller chunks
texts = text_splitter.split_text(raw_text)

# embeddings
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()

# create model
from langchain_community.llms import Ollama
model = Ollama(model="zephyr",base_url = 'http://127.0.0.1:11434')

# make vectorstore
vectorstore = FAISS.from_texts(texts,embeddings)

# retreival qa
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(model, retriever=vectorstore.as_retriever())

# query and its answer
query = "who is the author of this book ?"
ans =chain.run({"query":query})

print("ans:-->", ans)



# it dont use similarity search from FAISS

# how to run 
# this uses ollama which run locally on a computer 
# to run do the following
# in terminal type 

#       wsl --user root  -d ubuntu 
#       ollama serve
#       python3 qa.py

# this file is completed in its own 
