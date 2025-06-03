# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.encoders import jsonable_encoder
# from pydantic import BaseModel
# import uvicorn
# import os
# import shutil
# from typing import List
# import chromadb
# from chromadb.config import Settings
# import traceback
# from sentence_transformers import SentenceTransformer
# from ctransformers import AutoModelForCausalLM
# import PyPDF2

# app = FastAPI(title="AskMyNotes API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# chroma_client = None
# embedding_model = None
# llm = None

# model_name = "all-MiniLM-L12-v2"

# @app.on_event("startup")
# async def load_resources():
#     global chroma_client, embedding_model, llm
#     try:
#         chroma_client = chromadb.Client(Settings(
#             persist_directory="./data/chroma",
#             is_persistent=True
#         ))
#         embedding_model = SentenceTransformer(model_name)
#         llm = AutoModelForCausalLM.from_pretrained(
#             "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
#             model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#             model_type="mistral",
#             context_length=4096,
#             gpu_layers=0
#         )
#     except Exception as e:
#         print(f"Error loading resources: {e}")
#         print(traceback.format_exc())

# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     return JSONResponse(
#         status_code=500,
#         content=jsonable_encoder({
#             "success": False,
#             "detail": f"An error occurred: {str(exc)}"
#         })
#     )

# @app.get("/")
# async def health_check():
#     return {"success": True, "status": "healthy"}

# try:
#     chroma_client = chromadb.Client(Settings(
#         persist_directory="./data/chroma",
#         is_persistent=True
#     ))
# except Exception as e:
#     print(f"Error initializing ChromaDB: {e}")
#     print(traceback.format_exc())
#     chroma_client = None

# def get_embedding_model():
#     global embedding_model
#     if embedding_model is None:
#         embedding_model = SentenceTransformer(model_name)
#     return embedding_model

# def get_llm():
#     global llm
#     if llm is None:
#         llm = AutoModelForCausalLM.from_pretrained(
#             "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
#             model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#             model_type="mistral",
#             context_length=4096,
#             gpu_layers=0
#         )
#     return llm

# os.makedirs("./data/chroma", exist_ok=True)
# os.makedirs("./data/uploads", exist_ok=True)

# class QuestionRequest(BaseModel):
#     question: str
#     collection_name: str

# def process_text_chunks(text: str, chunk_size: int = 1000) -> List[str]:
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     current_size = 0

#     for word in words:
#         current_size += len(word) + 1
#         if current_size > chunk_size:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = [word]
#             current_size = len(word)
#         else:
#             current_chunk.append(word)
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         if not chroma_client:
#             raise HTTPException(500, "ChromaDB is not initialized")

#         if not file.filename.lower().endswith(('.txt', '.pdf')):
#             raise HTTPException(400, "Only .txt and .pdf files are supported")

#         collection_name = f"doc_{os.path.splitext(file.filename)[0]}"
#         collection = chroma_client.get_or_create_collection(
#             name=collection_name,
#             metadata={"filename": file.filename}
#         )

#         content = ""
#         if file.filename.lower().endswith('.pdf'):
#             temp_path = f"./data/uploads/{file.filename}"
#             try:
#                 with open(temp_path, "wb") as buffer:
#                     shutil.copyfileobj(file.file, buffer)
#                 await file.close()
#                 with open(temp_path, "rb") as pdf_file:
#                     pdf_reader = PyPDF2.PdfReader(pdf_file)
#                     for page in pdf_reader.pages:
#                         content += page.extract_text() + "\n"
#             finally:
#                 if os.path.exists(temp_path):
#                     os.remove(temp_path)
#         else:
#             content_bytes = await file.read()
#             content = content_bytes.decode()
#             await file.close()

#         chunks = process_text_chunks(content)
#         embedding_model = get_embedding_model()

#         for i, chunk in enumerate(chunks):
#             collection.add(
#                 documents=[chunk],
#                 metadatas=[{"chunk_id": i}],
#                 ids=[f"chunk_{i}"]
#             )

#         return JSONResponse(content={
#             "success": True,
#             "status": "success",
#             "message": f"Document processed and stored in collection: {collection_name}",
#             "collection_name": collection_name,
#             "chunk_count": len(chunks)
#         })

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "success": False,
#                 "detail": f"Error processing file: {str(e)}"
#             }
#         )

# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     try:
#         collection = chroma_client.get_collection(request.collection_name)
#         results = collection.query(
#             query_texts=[request.question],
#             n_results=3
#         )

#         context = "\n".join(results['documents'][0])
#         prompt = f"""<s>[INST] You are a helpful AI assistant. Use the following context to answer the question. 
#         If you cannot find the answer in the context, say so.

#         Context:
#         {context}

#         Question: {request.question}

#         Answer: [/INST]"""

#         llm = get_llm()
#         response = llm(prompt)

#         return {
#             "answer": response,
#             "context": context
#         }

#     except Exception as e:
#         raise HTTPException(500, f"Error processing question: {str(e)}")

# @app.get("/collections")
# async def list_collections():
#     try:
#         if not chroma_client:
#             raise HTTPException(500, "ChromaDB is not initialized")

#         collections = chroma_client.list_collections()
#         return {
#             "success": True,
#             "collections": [
#                 {
#                     "name": col.name,
#                     "metadata": col.metadata
#                 }
#                 for col in collections
#             ]
#         }
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "success": False,
#                 "detail": f"Error listing collections: {str(e)}"
#             }
#         )

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)





# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.encoders import jsonable_encoder
# from pydantic import BaseModel
# import uvicorn
# import os
# import re
# import shutil
# from typing import List
# import chromadb
# from chromadb.config import Settings
# import traceback
# from sentence_transformers import SentenceTransformer
# from ctransformers import AutoModelForCausalLM
# import PyPDF2

# app = FastAPI(title="AskMyNotes API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# chroma_client = None
# embedding_model = None
# llm = None

# model_name = "all-MiniLM-L12-v2"

# @app.on_event("startup")
# async def load_resources():
#     global chroma_client, embedding_model, llm
#     try:
#         chroma_client = chromadb.Client(Settings(
#             persist_directory="./data/chroma",
#             is_persistent=True
#         ))
#         embedding_model = SentenceTransformer(model_name)
#         llm = AutoModelForCausalLM.from_pretrained(
#             "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
#             model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#             model_type="mistral",
#             context_length=4096,
#             gpu_layers=0
#         )
#     except Exception as e:
#         print(f"Error loading resources: {e}")
#         print(traceback.format_exc())

# @app.exception_handler(Exception)
# async def global_exception_handler(request, exc):
#     return JSONResponse(
#         status_code=500,
#         content=jsonable_encoder({
#             "success": False,
#             "detail": f"An error occurred: {str(exc)}"
#         })
#     )

# @app.get("/")
# async def health_check():
#     return {"success": True, "status": "healthy"}

# try:
#     chroma_client = chromadb.Client(Settings(
#         persist_directory="./data/chroma",
#         is_persistent=True
#     ))
# except Exception as e:
#     print(f"Error initializing ChromaDB: {e}")
#     print(traceback.format_exc())
#     chroma_client = None

# def get_embedding_model():
#     global embedding_model
#     if embedding_model is None:
#         embedding_model = SentenceTransformer(model_name)
#     return embedding_model

# def get_llm():
#     global llm
#     if llm is None:
#         llm = AutoModelForCausalLM.from_pretrained(
#             "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
#             model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#             model_type="mistral",
#             context_length=4096,
#             gpu_layers=0
#         )
#     return llm

# os.makedirs("./data/chroma", exist_ok=True)
# os.makedirs("./data/uploads", exist_ok=True)

# class QuestionRequest(BaseModel):
#     question: str
#     collection_name: str

# def process_text_chunks(text: str, chunk_size: int = 1000) -> List[str]:
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     current_size = 0

#     for word in words:
#         current_size += len(word) + 1
#         if current_size > chunk_size:
#             chunks.append(" ".join(current_chunk))
#             current_chunk = [word]
#             current_size = len(word)
#         else:
#             current_chunk.append(word)
    
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

# @app.post("/upload")
# async def upload_files(files: List[UploadFile] = File(...)):
#     MAX_FILES = 5
#     if len(files) > MAX_FILES:
#         raise HTTPException(400, f"Maximum {MAX_FILES} files are allowed per upload.")
    
#     try:
#         if not chroma_client:
#             raise HTTPException(500, "ChromaDB is not initialized")

#         results = []
#         for file in files:
#             if not file.filename.lower().endswith(('.txt', '.pdf')):
#                 raise HTTPException(400, f"Only .txt and .pdf files are supported: {file.filename}")

#             raw_name = os.path.splitext(file.filename)[0]
#             safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)
#             safe_name = re.sub(r'[_-]{2,}', '_', safe_name)
#             safe_name = safe_name.strip('_-')
#             collection_name = f"doc_{safe_name}"

#             collection = chroma_client.get_or_create_collection(
#                 name=collection_name,
#                 metadata={"filename": file.filename}
#             )

#             content = ""
#             if file.filename.lower().endswith('.pdf'):
#                 temp_path = f"./data/uploads/{file.filename}"
#                 try:
#                     with open(temp_path, "wb") as buffer:
#                         shutil.copyfileobj(file.file, buffer)
#                     await file.close()
#                     with open(temp_path, "rb") as pdf_file:
#                         pdf_reader = PyPDF2.PdfReader(pdf_file)
#                         for page in pdf_reader.pages:
#                             content += page.extract_text() + "\n"
#                 finally:
#                     if os.path.exists(temp_path):
#                         os.remove(temp_path)
#             else:
#                 content_bytes = await file.read()
#                 content = content_bytes.decode()
#                 await file.close()

#             chunks = process_text_chunks(content)
#             embedding_model = get_embedding_model()

#             for i, chunk in enumerate(chunks):
#                 collection.add(
#                     documents=[chunk],
#                     metadatas=[{"chunk_id": i}],
#                     ids=[f"chunk_{i}"]
#                 )

#             results.append({
#                 "filename": file.filename,
#                 "collection_name": collection_name,
#                 "chunk_count": len(chunks)
#             })

#         return JSONResponse(content={
#             "success": True,
#             "status": "success",
#             "message": f"Documents processed and stored.",
#             "details": results
#         })

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "success": False,
#                 "detail": f"Error processing files: {str(e)}"
#             }
#         )
# @app.post("/ask")
# async def ask_question(request: QuestionRequest):
#     try:
#         collection = chroma_client.get_collection(request.collection_name)
#         results = collection.query(
#             query_texts=[request.question],
#             n_results=3
#         )

#         context = "\n".join(results['documents'][0])
#         prompt = f"""<s>[INST] You are a helpful AI assistant. Use the following context to answer the question. 
#         If you cannot find the answer in the context, say so.

#         Context:
#         {context}

#         Question: {request.question}

#         Answer: [/INST]"""

#         llm = get_llm()
#         response = llm(prompt)

#         return {
#             "answer": response,
#             "context": context
#         }

#     except Exception as e:
#         raise HTTPException(500, f"Error processing question: {str(e)}")

# @app.get("/collections")
# async def list_collections():
#     try:
#         if not chroma_client:
#             raise HTTPException(500, "ChromaDB is not initialized")

#         collections = chroma_client.list_collections()
#         return {
#             "success": True,
#             "collections": [
#                 {
#                     "name": col.name,
#                     "metadata": col.metadata
#                 }
#                 for col in collections
#             ]
#         }
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "success": False,
#                 "detail": f"Error listing collections: {str(e)}"
#             }
#         )

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import uvicorn
import os
import shutil
from typing import List
import chromadb
from chromadb.config import Settings
import traceback
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
import PyPDF2

app = FastAPI(title="AskMyNotes API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = None
embedding_model = None
llm = None

model_name = "all-MiniLM-L12-v2"

@app.on_event("startup")
async def load_resources():
    global chroma_client, embedding_model, llm
    try:
        chroma_client = chromadb.Client(Settings(
            persist_directory="./data/chroma",
            is_persistent=True
        ))
        embedding_model = SentenceTransformer(model_name)
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            model_type="mistral",
            context_length=4096,
            gpu_layers=0
        )
    except Exception as e:
        print(f"Error loading resources: {e}")
        print(traceback.format_exc())

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder({
            "success": False,
            "detail": f"An error occurred: {str(exc)}"
        })
    )

@app.get("/")
async def health_check():
    return {"success": True, "status": "healthy"}

try:
    chroma_client = chromadb.Client(Settings(
        persist_directory="./data/chroma",
        is_persistent=True
    ))
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    print(traceback.format_exc())
    chroma_client = None

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(model_name)
    return embedding_model

def get_llm():
    global llm
    if llm is None:
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            model_type="mistral",
            context_length=4096,
            gpu_layers=0
        )
    return llm

os.makedirs("./data/chroma", exist_ok=True)
os.makedirs("./data/uploads", exist_ok=True)

class QuestionRequest(BaseModel):
    question: str

def process_text_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        current_size += len(word) + 1
        if current_size > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    MAX_FILES = 5
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Maximum {MAX_FILES} files are allowed per upload.")
    
    try:
        if not chroma_client:
            raise HTTPException(500, "ChromaDB is not initialized")

        collection_name = "all_docs"
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Combined collection of all uploaded documents"}
        )

        results = []
        chunk_id_counter = 0  # To keep unique chunk ids across files

        for file in files:
            if not file.filename.lower().endswith(('.txt', '.pdf')):
                raise HTTPException(400, f"Only .txt and .pdf files are supported: {file.filename}")

            content = ""
            if file.filename.lower().endswith('.pdf'):
                temp_path = f"./data/uploads/{file.filename}"
                try:
                    with open(temp_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    await file.close()
                    with open(temp_path, "rb") as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        for page in pdf_reader.pages:
                            text = page.extract_text()
                            if text:
                                content += text + "\n"
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                content_bytes = await file.read()
                content = content_bytes.decode()
                await file.close()

            chunks = process_text_chunks(content)
            embedding_model = get_embedding_model()

            # Add chunks with unique IDs to avoid conflicts
            for chunk in chunks:
                collection.add(
                    documents=[chunk],
                    metadatas=[{"source": file.filename}],
                    ids=[f"chunk_{chunk_id_counter}"]
                )
                chunk_id_counter += 1

            results.append({
                "filename": file.filename,
                "chunk_count": len(chunks)
            })

        return JSONResponse(content={
            "success": True,
            "status": "success",
            "message": f"Documents processed and stored in collection: {collection_name}",
            "details": results
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "detail": f"Error processing files: {str(e)}"
            }
        )

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        collection_name = "all_docs"
        collection = chroma_client.get_collection(collection_name)
        results = collection.query(
            query_texts=[request.question],
            n_results=3
        )

        context = "\n".join(results['documents'][0])
        prompt = f"""<s>[INST] You are a helpful AI assistant. Use the following context to answer the question. 
        If you cannot find the answer in the context, say so.

        Context:
        {context}

        Question: {request.question}

        Answer: [/INST]"""

        llm = get_llm()
        response = llm(prompt)

        return {
            "answer": response,
            "context": context
        }

    except Exception as e:
        raise HTTPException(500, f"Error processing question: {str(e)}")

@app.get("/collections")
async def list_collections():
    try:
        if not chroma_client:
            raise HTTPException(500, "ChromaDB is not initialized")

        collections = chroma_client.list_collections()
        return {
            "success": True,
            "collections": [
                {
                    "name": col.name,
                    "metadata": col.metadata
                }
                for col in collections
            ]
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "detail": f"Error listing collections: {str(e)}"
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
