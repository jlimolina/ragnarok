import os
import io
import hashlib
import uuid
from fastapi import FastAPI, UploadFile, File, Form, Request, Response, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

import pdfplumber
import unidecode
import regex as re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from markdown import markdown

load_dotenv()
EMBED_MODEL   = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))
PERSIST_DIR   = os.getenv("PERSIST_DIR", "./vectordb")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "llama3:latest")

os.makedirs(PERSIST_DIR, exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/logo", StaticFiles(directory=".", html=False), name="logo")
templates = Jinja2Templates(directory="templates")

user_histories = {}

def get_ollama_models(selected_model=None):
    try:
        import ollama
        response = ollama.list()
        modelos = response.get("models", [])
        opciones = []
        for model in modelos:
            nombre = getattr(model, "model", None)
            if not nombre:
                continue
            opciones.append({
                "nombre": nombre,
                "selected": (selected_model == nombre)
            })
        if not opciones:
            opciones.append({
                "nombre": OLLAMA_MODEL,
                "selected": True,
                "default": True
            })
        return opciones
    except Exception as e:
        print("[ERROR] Al obtener modelos de Ollama:", str(e))
        return [{"nombre": OLLAMA_MODEL, "selected": True, "error": True}]

# -------- Vistas --------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/files", response_class=HTMLResponse)
async def list_files(request: Request):
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
    data = vectordb._collection.get(include=["metadatas"])
    metas = data.get("metadatas", [])
    files = {}
    for meta in metas:
        h = meta.get("file_hash")
        s = meta.get("source")
        files.setdefault(h, {"source": s, "chunks": 0})
        files[h]["chunks"] += 1
    rows = [
        {"source": v["source"], "hash": h[:8], "chunks": v["chunks"]}
        for h, v in files.items()
    ]
    return templates.TemplateResponse("files.html", {"request": request, "rows": rows})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, files: list[UploadFile] = File(...)):
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)

    results = []
    for file in files:
        try:
            data = await file.read()
            file_hash = hashlib.sha256(data).hexdigest()
            existing = vectordb._collection.get(where={"file_hash": file_hash}, limit=1)
            if existing and existing.get("ids"):
                results.append({
                    "filename": file.filename,
                    "status": "skipped",
                    "detail": "ya indexado"
                })
                file.file.seek(0)
                continue

            try:
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    raw = "\n".join(page.extract_text() or "" for page in pdf.pages)
            except Exception as e:
                raise ValueError(f"Error al leer PDF: {str(e)}")

            if not raw.strip():
                raise ValueError("PDF sin texto extraíble")

            txt = unidecode.unidecode(raw)
            txt = re.sub(r"[ \t]+", " ", txt)
            txt = re.sub(r"\r\n|\r", "\n", txt)
            lines = [l.strip() for l in txt.split("\n") if len(l.strip()) > 3]
            normalized = "\n".join(lines)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            chunks = splitter.split_text(normalized)
            if not chunks:
                raise ValueError("No se pudieron generar chunks del texto extraído.")

            metadatas = [{"file_hash": file_hash, "source": file.filename} for _ in chunks]
            vectordb.add_texts(texts=chunks, metadatas=metadatas)
            results.append({
                "filename": file.filename,
                "status": "indexed",
                "chunks": len(chunks)
            })
            file.file.seek(0)

        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": str(e)
            })

    return templates.TemplateResponse("upload_result.html", {
        "request": request,
        "results": results
    })

@app.get("/search", response_class=HTMLResponse)
async def search_form(request: Request, session_id: str = Cookie(default=None)):
    if not session_id:
        session_id = str(uuid.uuid4())
        user_histories[session_id] = []
        response = RedirectResponse(url="/search")
        response.set_cookie("session_id", session_id, max_age=60*60*24*7)
        return response
    select_modelos = get_ollama_models(OLLAMA_MODEL)
    history = user_histories.get(session_id, [])
    chat_history = [
        {"q": q, "a": a, "a_html": markdown(a, extensions=['fenced_code','tables'])}
        for q, a in history
    ]
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "chat_history": chat_history,
        "select_modelos": select_modelos,
        "OLLAMA_MODEL": OLLAMA_MODEL
    })

@app.post("/search", response_class=HTMLResponse)
async def search_answer(
    request: Request,
    response: Response,
    query: str = Form(...),
    model: str = Form(OLLAMA_MODEL),
    session_id: str = Cookie(default=None)
):
    if not session_id:
        session_id = str(uuid.uuid4())
    history = user_histories.setdefault(session_id, [])
    prompt_context = "\n\n".join(
        [f"Pregunta: {q}\nRespuesta: {a}" for q, a in history[-4:]]
    )
    emb = OllamaEmbeddings(model=EMBED_MODEL)
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
    docs = vectordb.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    context = context[:3000]

    prompt = f"""Eres un asistente RAG. Usa SOLO el contexto proporcionado.
---
Historial de conversación (máximo 4 interacciones previas):
{prompt_context}
---
Pregunta: {query}
---
Contexto RAG:
{context}
---
Respuesta:"""

    try:
        import ollama
        respuesta = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False
        )["response"]
    except Exception as e:
        respuesta = f"Error llamando a Ollama: {e}"

    history.append((query, respuesta))

    select_modelos = get_ollama_models(model)
    chat_history = [
        {"q": q, "a": a, "a_html": markdown(a, extensions=['fenced_code','tables'])}
        for q, a in history
    ]
    resp = templates.TemplateResponse("chat.html", {
        "request": request,
        "chat_history": chat_history,
        "select_modelos": select_modelos,
        "OLLAMA_MODEL": model
    })
    resp.set_cookie("session_id", session_id, max_age=60*60*24*7)
    return resp

@app.get("/reset_chat", response_class=HTMLResponse)
async def reset_chat(session_id: str = Cookie(default=None)):
    if session_id and session_id in user_histories:
        del user_histories[session_id]
    resp = RedirectResponse(url="/search")
    resp.set_cookie("session_id", str(uuid.uuid4()), max_age=60*60*24*7)
    return resp

