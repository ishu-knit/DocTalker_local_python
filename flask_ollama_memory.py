# successfully run 
# flask + locally ollama + Memory + retrivalQA   
# in postman 
#   link :--  POST  http://127.0.0.1:5000/api/query for query 
#  body {   
#     // "query":"give me 5 question that i created for the exam  in points",
#     "text":"hello"
# }



from langchain_community.vectorstores import   SupabaseVectorStore
import os   
from supabase import create_client
from flask import Flask, request, jsonify

from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory , ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()



app = Flask(__name__)



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize models
ollama_embeddings = OllamaEmbeddings(model="zephyr",base_url = 'http://127.0.0.1:11434')


doc_reader = PyPDFLoader("./pdfs/prompt.pdf")
docs = doc_reader.load_and_split()[0:5]
url = "https://oxkmkprtwkuiewajaqvx.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im94a21rcHJ0d2t1aWV3YWphcXZ4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDM4Mzc3MDEsImV4cCI6MjAxOTQxMzcwMX0.vWKFLz2r_RzuJ_ZnsMFvp0srckuMcjCiB7RkfhlGj_c"
model = Ollama(model="zephyr",base_url = 'http://127.0.0.1:11434',temperature=0)



# Initialize client
supabase_client = create_client(url, key)


vectorstore = SupabaseVectorStore(
                client=supabase_client,
                embedding=ollama_embeddings,
                table_name="documents",
                query_name="match_documents",
            )

template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
------
<context>{context}</context>

------
<history> {history} </history>
------

{question}
Answer:
"""


prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)


# memory type 1
# memory = ConversationBufferMemory(
#             memory_key="history",
#             input_key="question")

# memory type-2
memory = ConversationSummaryMemory(llm=model,
                                    return_messages=True,
                                    memory_key="history",
                                    input_key="question")

qa = RetrievalQA.from_chain_type(
    llm= ChatOllama(model="zephyr"),
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": memory,
    }
)





# embedding
@app.route('/api/load', methods=['POST'])
def load_pdf():

    # to load all pdf from a single pdf folder name "pdfs"
    pdf = PyPDFDirectoryLoader("pdfs")
    docs = pdf.load_and_split()[0:6]

    data = request.get_json()
    text = data.get('text')

    # create_embedding + store in supabase
    vectorstore = SupabaseVectorStore.from_documents(docs, ollama_embeddings, client=supabase_client , table_name="documents")

    return "created!!"





# query
# API for querying (completed!!)
@app.route('/api/query', methods=['POST'])
def query():
    # extract complete json data
    data = request.get_json()
    # extract "query" key from json
    query_text = data.get('query')
    
     # Invoke the retrieval chain with the query
    # result = retrieval_chain.invoke({"query": query_text})
    ans = qa.invoke({"query": query_text})
    print("ans------------------->>>>",ans)

    
    return jsonify({'result': ans})


if __name__ == '__main__':
    app.run(debug=True)
