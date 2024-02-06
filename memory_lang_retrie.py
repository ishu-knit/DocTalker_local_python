# # successfully run 

# ****************************************openai************************************************

# from langchain_community.vectorstores import   SupabaseVectorStore
# import os   
# from supabase import create_client
# from flask import Flask, request, jsonify
# from langchain.chat_models import ChatOpenAI
# from langchain.memory import ConversationBufferMemory
# from langchain import PromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# load_dotenv()



# app = Flask(__name__)


# OPENAI_API_KEY = "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"

# # # Initialize models
# ollama_embeddings = OllamaEmbeddings(model="luffy",base_url = 'http://127.0.0.1:11434')


# doc_reader = PyPDFLoader("impromptu-rh.pdf")
# docs = doc_reader.load_and_split()[0:5]
# url = "https://oxkmkprtwkuiewajaqvx.supabase.co"
# key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im94a21rcHJ0d2t1aWV3YWphcXZ4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDM4Mzc3MDEsImV4cCI6MjAxOTQxMzcwMX0.vWKFLz2r_RzuJ_ZnsMFvp0srckuMcjCiB7RkfhlGj_c"



# # # Initialize client
# supabase_client = create_client(url, key)

# os.getenv("OPENAI_API_KEY")

# vectorstore = SupabaseVectorStore(
#                 client=supabase_client,
#                 embedding=ollama_embeddings,
#                 table_name="documents",
#                 query_name="match_documents",
#             )

# template = """
# Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
# ------
# <context>{context}</context>

# ------
# <history> {history} </history>
# ------

# {question}
# Answer:
# """


# prompt = PromptTemplate(
#     input_variables=["history", "context", "question"],
#     template=template,
# )

# qa = RetrievalQA.from_chain_type(
#     llm=ChatOpenAI(),
#     chain_type='stuff',
#     retriever=vectorstore.as_retriever(),
#     verbose=True,
#     chain_type_kwargs={
#         "verbose": True,
#         "prompt": prompt,
#         "memory": ConversationBufferMemory(
#             memory_key="history",
#             input_key="question"),
#     }
# )



# # query
# # API for querying (completed!!)
# @app.route('/api/query', methods=['POST'])
# def query():
#     # extract complete json data
#     data = request.get_json()
#     # extract "query" key from json
#     query_text = data.get('query')
    
#      # Invoke the retrieval chain with the query
#     # result = retrieval_chain.invoke({"query": query_text})
#     ans = qa.run({"query": query_text})
#     print("ans------------------->>>>",ans)

    
#     return jsonify({'result': ans})


# if __name__ == '__main__':
#     app.run(debug=True)



# *****************************************************************ollama ********************************




# from langchain_community.vectorstores import   SupabaseVectorStore
# import os   
# from supabase import create_client
# from flask import Flask, request, jsonify
# from langchain.chat_models import ChatOpenAI
# from langchain_community.chat_models import ChatOllama
# from langchain.memory import ConversationBufferMemory
# from langchain import PromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv
# load_dotenv()



# app = Flask(__name__)



# OPENAI_API_KEY = "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"

# # # Initialize models
# ollama_embeddings = OllamaEmbeddings(model="luffy",base_url = 'http://127.0.0.1:11434')


# doc_reader = PyPDFLoader("impromptu-rh.pdf")
# docs = doc_reader.load_and_split()[0:5]
# url = "https://oxkmkprtwkuiewajaqvx.supabase.co"
# key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im94a21rcHJ0d2t1aWV3YWphcXZ4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MDM4Mzc3MDEsImV4cCI6MjAxOTQxMzcwMX0.vWKFLz2r_RzuJ_ZnsMFvp0srckuMcjCiB7RkfhlGj_c"



# # # Initialize client
# supabase_client = create_client(url, key)

# os.getenv("OPENAI_API_KEY")

# vectorstore = SupabaseVectorStore(
#                 client=supabase_client,
#                 embedding=ollama_embeddings,
#                 table_name="documents",
#                 query_name="match_documents",
#             )

# template = """
# Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
# ------
# <context>{context}</context>

# ------
# <history> {history} </history>
# ------

# {question}
# Answer:
# """


# prompt = PromptTemplate(
#     input_variables=["history", "context", "question"],
#     template=template,
# )

# qa = RetrievalQA.from_chain_type(
#     llm= ChatOllama(model="luffy"),
#     chain_type='stuff',
#     retriever=vectorstore.as_retriever(),
#     verbose=True,
#     chain_type_kwargs={
#         "verbose": True,
#         "prompt": prompt,
#         "memory": ConversationBufferMemory(
#             memory_key="history",
#             input_key="question"),
#     }
# )



# # query
# # API for querying (completed!!)
# @app.route('/api/query', methods=['POST'])
# def query():
#     # extract complete json data
#     data = request.get_json()
#     # extract "query" key from json
#     query_text = data.get('query')
    
#      # Invoke the retrieval chain with the query
#     # result = retrieval_chain.invoke({"query": query_text})
#     ans = qa.invoke({"query": query_text})
#     print("ans------------------->>>>",ans)

    
#     return jsonify({'result': ans})


# if __name__ == '__main__':
#     app.run(debug=True)
