# utils/backup.py

import os
import io
import zipfile
import datetime

def create_backup_zip(paths_to_backup=None):
    """
    Crea un archivo ZIP en memoria con las rutas indicadas (carpetas/archivos).
    Retorna un BytesIO listo para enviar.
    """
    if paths_to_backup is None:
        paths_to_backup = ['vectordb', 'chat_history.db']

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"backup_{timestamp}.zip"

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as backup_zip:
        for path in paths_to_backup:
            if os.path.isdir(path):
                for folder, _, files in os.walk(path):
                    for file in files:
                        filepath = os.path.join(folder, file)
                        arcname = os.path.relpath(filepath, start=".")
                        backup_zip.write(filepath, arcname)
            elif os.path.isfile(path):
                backup_zip.write(path, os.path.basename(path))
    mem_zip.seek(0)
    return mem_zip, backup_filename
def restore_backup_zip(zip_file, paths_to_restore=None):
    """
    Restaura archivos/carpetas desde un archivo ZIP.
    zip_file: un objeto BytesIO o archivo subido por FastAPI.
    paths_to_restore: opcional, lista de rutas base a restaurar (por defecto, vectordb y chat_history.db).
    """
    if paths_to_restore is None:
        paths_to_restore = ['vectordb', 'chat_history.db']

    with zipfile.ZipFile(zip_file) as zf:
        for member in zf.namelist():
            # Solo extraer archivos que estén dentro de las rutas permitidas
            allow = False
            for p in paths_to_restore:
                if member == p or member.startswith(p + "/") or member.startswith("./" + p):
                    allow = True
            if allow:
                zf.extract(member, ".")

