import json
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters.latex import LatexTextSplitter
from openai import OpenAI
from duckduckgo_search import DDGS



load_dotenv()
openai_key = os.getenv("openai_key")
os.environ["OPENAI_API_KEY"] = openai_key

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model = ChatOpenAI(model="gpt-4o", temperature=0.3)

app = FastAPI()
origins = ["http://localhost:3000", "http://44.215.20.161:3015"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers="*",
)

# Set your API keys
client = OpenAI(api_key=openai_key)


def chunk_json(strdata):
    """Convert a JSON object into small chunks of key-value pairs."""
    latex_splitter = LatexTextSplitter(chunk_size=512)
    docs = latex_splitter.create_documents([strdata])
    chunks = [doc.page_content for doc in docs]

    return chunks

# function for the google  serach
def  google_search(query):
    try:
        # google search
        results = DDGS().text(query, max_results=5)
        return results
    except Exception as e:
        return False


def storeOnFaiss(strjosn):
    chunks = chunk_json(strjosn)
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def QAWithFaiss(query):
    db = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    docs = db.similarity_search(query=query, k=2)
    text = [i.page_content for i in docs]

    #  function for the  googel search
    results=google_search(query=query)

    # final  context
    context = text+results


    prompt = f"""
        You are an assistant. Use the following information to answer the query:
        {context}

        Attention: if the user query is not related with  medical  sector then response with:- I'm here to assist you with questions related to Meda Medical Dashboard . Could you please rephrase your question or provide more context? If your query is outside my expertise, I recommend reaching out to the appropriate resource for further help.
        Query: {query}
        Answer:
        """
    response = model.invoke(prompt).content
    return response


# Endpoint to insert data into Pinecone
@app.post("/insert")
async def insert_data(request: Request):
    """
    Insert data into Pinecone.
    :param doc_id: Unique ID for the document
    :param data: JSON data to store
    """
    jsondata = await request.json()
    try:
        strdata: str = json.dumps(jsondata)
        storeOnFaiss(strdata)
        return {"message": "Data inserted successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"{e}, data type is:" + type(jsondata)
        )


# Endpoint to query the chatbot
@app.post("/meda-chatbot")
async def chatbot(request: Request):
    """
    Chatbot endpoint to handle user queries.
    :param query: User query
    """
    body = await request.json()
    query = body.get("question")
    try:
        # Search Pinecone for relevant data
        answer = QAWithFaiss(query)

        return JSONResponse(content={"answer": answer}, status_code=200)
        # return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
