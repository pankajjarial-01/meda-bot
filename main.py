import os
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
openai_key = os.getenv("openai_key")
pinecone_key = os.getenv("pinecone_key")
index_name = os.getenv("index_name")
app = FastAPI()
# Set your API keys
client = OpenAI(api_key=openai_key)
pc = Pinecone(api_key=pinecone_key)
index_name = index_name
# Check if the Pinecone index exists, otherwise create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name, dimension=1536, spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )  # 1536 is the dimensionality of text-embedding-ada-002
index = pc.Index(index_name)


# Function to chunk JSON data into key-value pairs
def chunk_json(json_obj):
    """Convert a JSON object into small chunks of key-value pairs."""
    chunks = []
    for key, value in json_obj.items():
        if isinstance(value, dict):
            chunks.extend(chunk_json(value))  # Recursively process nested JSON
        else:
            # Add key-value pairs as text
            chunks.append(f"{key}: {str(value)}")
    return chunks


# Function to generate embeddings using OpenAI's API
def get_embedding(text):
    """Generate embeddings using OpenAI's API."""
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding


# Function to store chunks in Pinecone
def store_chunks_in_pinecone(chunks, doc_id):
    """Store chunk embeddings in Pinecone."""
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        metadata = {"chunk": chunk, "doc_id": doc_id}
        index.upsert([(f"{doc_id}_{i}", embedding, metadata)])


# Function to search Pinecone for relevant chunks
def search_pinecone_with_query(query):
    """Search Pinecone for relevant chunks using query embedding."""
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=5,  # Adjust based on your needs
        include_metadata=True,
    )
    return [match["metadata"]["chunk"] for match in results["matches"]]


# Function to generate a meaningful response using OpenAI GPT
def generate_response(query, relevant_chunks):
    """Construct a meaningful response using GPT."""
    context = "\n".join(relevant_chunks)
    prompt = f"""
    You are an assistant. Use the following information to answer the query:
    {context}
    
    Query: {query}
    Answer:
    """
    completion = client.chat.completions.create(
        model="gpt-4o",  # Use the desired GPT model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return completion.choices[0].message


# Endpoint to insert data into Pinecone
@app.post("/insert")
async def insert_data(request: Request):
    """
    Insert data into Pinecone.
    :param doc_id: Unique ID for the document
    :param data: JSON data to store
    """
    body = await request.json()
    doc_id = body.get("id")
    data = body.get("data")
    try:
        doc_id == uuid4()
        chunks = chunk_json(data)
        store_chunks_in_pinecone(chunks, doc_id)
        return {"message": "Data inserted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to query the chatbot
@app.post("/meda-bot")
async def chatbot(request: Request):
    """
    Chatbot endpoint to handle user queries.
    :param query: User query
    """
    body = await request.json()
    query = body.get("query")
    try:
        # Search Pinecone for relevant data
        relevant_chunks = search_pinecone_with_query(query)

        # Generate a meaningful response
        response = generate_response(query, relevant_chunks)

        return JSONResponse(content={"response": response.content})
        # return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
