import io
import pdfplumber
import unidecode
import regex as re

def extract_text_from_pdf(data: bytes) -> str:
    """
    Extrae y normaliza el texto de un PDF en formato bytes.
    Devuelve un string plano listo para procesar en chunks.
    Lanza ValueError si no hay texto extraíble.
    """
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
    return normalized

