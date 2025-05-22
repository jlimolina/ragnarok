import io
import docx
import unidecode
import regex as re

def extract_text_from_docx(data: bytes) -> str:
    """
    Extrae y normaliza el texto de un archivo DOCX en formato bytes.
    Devuelve un string plano listo para procesar en chunks.
    Lanza ValueError si no hay texto extraíble.
    """
    try:
        file_stream = io.BytesIO(data)
        doc = docx.Document(file_stream)
        raw = "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        raise ValueError(f"Error al leer DOCX: {str(e)}")

    if not raw.strip():
        raise ValueError("DOCX sin texto extraíble")

    txt = unidecode.unidecode(raw)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\r\n|\r", "\n", txt)
    lines = [l.strip() for l in txt.split("\n") if len(l.strip()) > 3]
    normalized = "\n".join(lines)
    return normalized

