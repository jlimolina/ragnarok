#!/bin/bash

APP_NAME="ragnarok"
SERVICE_FILE="/etc/systemd/system/$APP_NAME.service"
PROJECT_DIR="$HOME/ragnarok"     # Cambia esta ruta si tu proyecto está en otro sitio
OLLAMA_BIN="$HOME/.ollama"
OLLAMA_SERVICE="ollama"

set -e

echo ">>> Parando y quitando servicio systemd de $APP_NAME..."
sudo systemctl stop $APP_NAME || true
sudo systemctl disable $APP_NAME || true
if [ -f "$SERVICE_FILE" ]; then
    sudo rm "$SERVICE_FILE"
    sudo systemctl daemon-reload
    echo ">>> Servicio $APP_NAME eliminado de systemd."
fi

echo ">>> Matando procesos Ollama y eliminando binarios..."
pkill -f ollama || true
[ -d "$OLLAMA_BIN" ] && rm -rf "$OLLAMA_BIN"
sudo rm -rf /usr/local/bin/ollama 2>/dev/null || true

echo ">>> Buscando e intentando borrar instalación global de Ollama (si existe)..."
OLLAMA_GLOBAL=$(which ollama 2>/dev/null || true)
if [ ! -z "$OLLAMA_GLOBAL" ]; then
    sudo rm -f "$OLLAMA_GLOBAL"
fi

echo ">>> Borrando proyecto Ragnarok y bases de datos..."
rm -rf "$PROJECT_DIR"

echo ">>> Desinstalando dependencias de Python (opcional)..."
# Si quieres borrar el entorno virtual, descomenta esta línea
rm -rf "$PROJECT_DIR/.venv"

echo ">>> ¡Desinstalación completada!"

