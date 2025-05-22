#!/bin/bash

set -e

APP_NAME="ragnarok"
VENV=".venv"
APP_FILE="app.py"
DB_FILE="chat_history.db"
STATIC_DIR="static"
TEMPLATES_DIR="templates"
VECTORDB_DIR="vectordb"
ENV_FILE=".env"
REQ_FILE="requirements.txt"
OLLAMA_MODELS=("gemma3:1b" "nomic-embed-text")
OLLAMA_PORT=11434
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"
PROJECT_DIR="$(pwd)"

# 0. Dependencias mínimas del sistema
sudo apt update
sudo apt install -y curl python3 python3-venv python3-pip

# 1. Instalar ollama si no existe
if ! command -v ollama &> /dev/null; then
    echo ">>> Instalando Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    export PATH="$PATH:$HOME/.ollama/bin"
else
    echo ">>> Ollama ya está instalado."
fi

# 2. Arrancar servicio ollama si no está activo (solo en Linux)
if ! pgrep -x "ollama" > /dev/null; then
    echo ">>> Arrancando el servicio Ollama..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 5
fi

# 3. Descargar los modelos Ollama requeridos
for model in "${OLLAMA_MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        echo ">>> El modelo $model ya está descargado."
    else
        echo ">>> Descargando el modelo $model..."
        ollama pull "$model"
    fi
done

# 4. Crear entorno Python si no existe
if [ ! -d "$VENV" ]; then
    echo ">>> Creando entorno virtual..."
    python3 -m venv $VENV
fi

source $VENV/bin/activate

echo ">>> Actualizando pip..."
pip install --upgrade pip

if [ -f "$REQ_FILE" ]; then
    echo ">>> Instalando dependencias desde $REQ_FILE..."
    pip install -r $REQ_FILE
else
    echo ">>> [AVISO] No se encontró $REQ_FILE, instalando dependencias básicas..."
    pip install fastapi uvicorn[standard] python-multipart pdfplumber unidecode regex markdown langchain langchain-community langchain-ollama langchain-chroma chromadb python-dotenv aiosqlite jinja2
fi

echo ">>> Creando estructura de carpetas..."
mkdir -p "$STATIC_DIR" "$TEMPLATES_DIR" "$VECTORDB_DIR"

# 5. Inicializar base de datos SQLite
echo ">>> Inicializando base de datos SQLite..."
python3 - <<EOF
import aiosqlite
import asyncio

async def init_db():
    async with aiosqlite.connect("$DB_FILE") as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            role TEXT,
            content TEXT
        );
        """)
        await db.commit()

asyncio.run(init_db())
EOF

# 6. Crear archivo .env si no existe
echo ">>> Creando archivo .env si no existe..."
if [ ! -f "$ENV_FILE" ]; then
    cat <<EOL > $ENV_FILE
# Variables de configuración para RAGnarok
EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=512
CHUNK_OVERLAP=64
PERSIST_DIR=./vectordb
OLLAMA_MODEL=gemma3:1b
EOL
    echo ">>> Archivo .env creado."
else
    echo ">>> Archivo .env ya existe, no se modifica."
fi

# 7. Crear servicio systemd
echo ">>> Creando servicio systemd..."

sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=Ragnarok FastAPI Server
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/$VENV/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always
User=$(whoami)
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable $APP_NAME
sudo systemctl restart $APP_NAME

echo ""
echo ">>> ¡Instalación completada y servicio '$APP_NAME' lanzado!"
echo ">>> Puedes comprobar el estado con:"
echo "    sudo systemctl status $APP_NAME"
echo ">>> Accede a: http://TU-IP:8000/"
echo ""
echo "Asegúrate de tener tus carpetas y archivos en:"
echo "- Código Python y $APP_FILE en la raíz"
echo "- Plantillas HTML en $TEMPLATES_DIR/"
echo "- CSS, logos, etc. en $STATIC_DIR/"

