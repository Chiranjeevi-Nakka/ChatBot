import json
import os
import uuid
from openai import AzureOpenAI
from azure.search.documents import SearchClient
# from azure.search.documents.models import QueryType
# from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import RawVectorQuery
from azure.core.credentials import AzureKeyCredential
# from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
# from azure.cognitiveservices.language.textanalytics.models import TextDocumentInput
# from sqlalchemy.orm import Session
# import model
from dotenv import load_dotenv
import time
load_dotenv()


AZURE_COGNITIVE_SEARCH_SERVICE_NAME = os.getenv("AZURE_COGNITIVE_SEARCH_SERVICE_NAME")
AZURE_COGNITIVE_SEARCH_INDEX_NAME = os.getenv("AZURE_COGNITIVE_SEARCH_INDEX_NAME")
AZURE_COGNITIVE_SEARCH_API_KEY = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY")
AZURE_COGNITIVE_SEARCH_ENDPOINT = os.getenv("AZURE_COGNITIVE_SEARCH_ENDPOINT")
azure_credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
# OPENAI_API_VERSION = "2024-06-01"

EMBEDDING_MODEL_Large  = os.getenv("EMBEDDING_MODEL_Large")
# EMBEDDING_MODEL_Ada =  os.getenv("EMBEDDING_MODEL_Ada")

# GPT35 = os.getenv("GPT35")
GPT4 = os.getenv("GPT4")

client = AzureOpenAI(
  api_key = OPENAI_API_KEY,  
  api_version = OPENAI_API_VERSION,
  azure_endpoint = OPENAI_API_ENDPOINT
)

system_message_query_generation_for_retriver = "Assistant is a large language model."

def generate_embeddings_azure_openai(text = " ",model = os.getenv("EMBEDDING_MODEL_Large")):
    response = client.embeddings.create(
        input = text,
        model= model
    )
    return response.data[0].embedding

def call_gpt_model(model= GPT4,
                                  messages= [],
                                  temperature=0.1,
                                  max_tokens = 700,
                                  stream = True):


    response_obj = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature = temperature,
                                              max_tokens = max_tokens,
                                              stream= stream,seed=999)

    return response_obj

def generate_query_for_retriver(user_query = " ",messages = [],model= GPT4):

    start = time.time()
    user_message = summary_prompt_template = """Chat History:
    {chat_history}

    Question:
    {question}

    Search query:"""

    user_message = user_message.format(chat_history=str(messages), question=user_query)

    chat_conversations_for_query_generation_for_retriver = [{"role" : "system", "content" : system_message_query_generation_for_retriver}]
    chat_conversations_for_query_generation_for_retriver.append({"role": "user", "content": user_message })

    response = call_gpt_model(messages = chat_conversations_for_query_generation_for_retriver,stream = False,model= model)
    response = response.choices[0].message.content

    return response


class retrive_similiar_docs : 

    def __init__(self,query = " ", retrive_fields = ["id","content", "filepath","url","title"],
                      ):
        if query:
            self.query = query

        self.search_client = SearchClient(AZURE_COGNITIVE_SEARCH_ENDPOINT, AZURE_COGNITIVE_SEARCH_INDEX_NAME, azure_credential)
        self.retrive_fields = retrive_fields
    
    def text_search(self,top = 2):
        results = self.search_client.search(search_text= self.query,
                                select=self.retrive_fields,top=top)
        
        return results
        

    def pure_vector_search(self, k = 2, vector_field = 'contentVector',query_embedding = []):

        vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=vector_field)

        results = self.search_client.search( search_text=None,  vector_queries= [vector_query],
                                            select=self.retrive_fields)

        return results
        
    def hybrid_search(self,top = 2, k = 2,vector_field = "contentVector",query_embedding = []):
        
        vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=vector_field)
        results = self.search_client.search(search_text=self.query,  vector_queries= [vector_query],
                                                select=self.retrive_fields,top=top)  

        return results


def get_similiar_content(user_query = " ",
                      search_type = "hybrid",top = 2, k =2):

    retrive_docs = retrive_similiar_docs(query = user_query)


    if search_type == "hybrid":
        start = time.time()
        vector_of_search_query = generate_embeddings_azure_openai(user_query)

        start = time.time()
        r = retrive_docs.hybrid_search(top = top, k=k, query_embedding = vector_of_search_query)

        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append("Title : " +doc["title"] + "\n" "URL : " + doc["url"] + "\ncontent : " + doc["content"])
            sources.append("ID: "+doc["id"] +" : " + doc["filepath"])
        similiar_docs = "\n\n\n".join(similiar_doc)

        source = " ".join(sources)

    return similiar_docs,source
    


system_message = "Assistant is a large language model."

chat_conversations_global_message = [{"role" : "system", "content" : system_message}]


def generate_response_with_memory(memory_messages, user_query = " ", model=GPT4, stream=False):

    query_for_retriver = generate_query_for_retriver(user_query=user_query,messages = memory_messages,model=model)
    
    similiar_docs,sources = get_similiar_content(query_for_retriver)
    user_content = user_query + " \nSOURCES:\n" + similiar_docs

    chat_conversations_to_send = chat_conversations_global_message + memory_messages + [{"role":"user","content" : user_content}]
    
    response_from_model = call_gpt_model(messages = chat_conversations_to_send,model=model)
    
    #sources = " ".join(sources)
    return response_from_model,query_for_retriver,sources

def pdf_service(query_string, memory_messages):
    # if model == "GPT-3.5":
    #     model_to_use  = GPT35

    # if model == "GPT-4":
    #     model_to_use  = GPT4
    
    # if model == "GPT-4-Turbo":
    #     model_to_use  = GPT4

    response,query_for_retriver,sources = generate_response_with_memory(memory_messages, user_query= query_string,stream=True,model=GPT4)

    full_response = " "
    for chunk in response:
        if len(chunk.choices) >0:
            if str(chunk.choices[0].delta.content) != "None":                     
                for char in chunk.choices[0].delta.content:
                    # print("response----->",char)
                    full_response += char
                    print(full_response)
                    time.sleep(0.01) 
                yield full_response