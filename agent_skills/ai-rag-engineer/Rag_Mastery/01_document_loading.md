# RAG 01 — Document Loading: PDF Nativo, OCR, Multimodal

## Regla de Oro
> El mejor RAG empieza con la mejor extracción de texto.
> Basura entra → basura sale. El loader es la base de todo.

---

## 1. PDF Nativo (Texto Seleccionable)

```python
# loaders/pdf_native_loader.py
from pathlib import Path
from typing import Iterator
from langchain_core.documents import Document
import fitz  # PyMuPDF — superior a pypdf para documentos complejos


class DocumentPDFLoader:
    """
    Loader profesional para PDFs con texto nativo.
    Preserva estructura: número de página, sección, metadata enriquecida.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)

    def load(self) -> list[Document]:
        """Carga el PDF preservando estructura y metadata completa."""
        documents: list[Document] = []

        with fitz.open(str(self.file_path)) as pdf:
            doc_metadata = self._extract_doc_metadata(pdf)

            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text")  # Texto limpio
                blocks = page.get_text("blocks")  # Bloques para análisis estructural

                if not text.strip():
                    continue  # Página vacía o imagen pura → pasar a OCR

                # Metadata enriquecida por página
                metadata = {
                    **doc_metadata,
                    "page": page_num,
                    "total_pages": len(pdf),
                    "source": str(self.file_path),
                    "has_images": len(page.get_images()) > 0,
                    "has_tables": self._detect_tables(blocks),
                }

                documents.append(Document(page_content=text, metadata=metadata))

        return documents

    def _extract_doc_metadata(self, pdf: fitz.Document) -> dict:
        """Extrae metadata del PDF: título, autor, fecha."""
        meta = pdf.metadata or {}
        return {
            "title": meta.get("title", self.file_path.stem),
            "author": meta.get("author", ""),
            "creation_date": meta.get("creationDate", ""),
            "doc_type": "document",
        }

    def _detect_tables(self, blocks: list) -> bool:
        """Heurística simple para detectar tablas por densidad de bloques."""
        return len(blocks) > 10


# ─── Loader con PyPDF como fallback ───────────────────────────────────────────

def load_pdf_with_fallback(file_path: str) -> list[Document]:
    """
    Intenta PyMuPDF primero (mejor calidad).
    Fallback a pypdf si falla.
    """
    try:
        loader = DocumentPDFLoader(file_path)
        docs = loader.load()
        if docs:
            return docs
    except Exception as e:
        print(f"⚠️  PyMuPDF falló ({e}), usando pypdf como fallback")

    from langchain_community.document_loaders import PyPDFLoader
    return PyPDFLoader(file_path).load()
```

---

## 2. PDF Escaneado (OCR Pipeline)

```python
# loaders/ocr_loader.py
from pathlib import Path
from langchain_core.documents import Document
import fitz
import pytesseract
from PIL import Image
import io
import re


class OCRPDFLoader:
    """
    Loader OCR para PDFs escaneados (imágenes).
    Usa pytesseract con configuración optimizada para documentos densos.
    """

    # Configuración OCR optimizada para texto denso
    OCR_CONFIG = r"--oem 3 --psm 6 -l spa"  # OEM 3=LSTM, PSM 6=bloque uniforme, spa=español

    def __init__(self, file_path: str | Path, dpi: int = 300) -> None:
        self.file_path = Path(file_path)
        self.dpi = dpi  # 300 DPI mínimo para buena calidad OCR

    def load(self) -> list[Document]:
        documents: list[Document] = []

        with fitz.open(str(self.file_path)) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                # Convertir página a imagen de alta resolución
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")

                # OCR con pytesseract
                image = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(image, config=self.OCR_CONFIG)
                text = self._clean_ocr_text(text)

                if not text.strip():
                    continue

                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(self.file_path),
                        "page": page_num,
                        "total_pages": len(pdf),
                        "extraction_method": "ocr",
                        "ocr_dpi": self.dpi,
                        "doc_type": "document",
                    },
                ))

        return documents

    def _clean_ocr_text(self, text: str) -> str:
        """Limpieza post-OCR para documentos."""
        # Eliminar guiones de separación de palabras
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Normalizar espacios múltiples
        text = re.sub(r" {2,}", " ", text)
        # Eliminar líneas con solo caracteres basura (OCR artifacts)
        lines = [
            line for line in text.split("\n")
            if len(line.strip()) > 3 or not line.strip()
        ]
        return "\n".join(lines)


# ─── Pipeline Unificado: Nativo + OCR ────────────────────────────────────────

class SmartPDFLoader:
    """
    Pipeline inteligente: detecta qué páginas son nativas y cuáles son OCR.
    Ideal para PDFs mixtos (algunas páginas escaneadas, otras nativas).
    """

    TEXT_THRESHOLD = 50  # Chars mínimos para considerar página como nativa

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self._native_loader = DocumentPDFLoader(file_path)
        self._ocr_loader = OCRPDFLoader(file_path)

    def load(self) -> list[Document]:
        """Carga cada página con el método más apropiado."""
        documents: list[Document] = []

        with fitz.open(str(self.file_path)) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                native_text = page.get_text("text").strip()

                if len(native_text) >= self.TEXT_THRESHOLD:
                    # Página con texto nativo
                    doc = Document(
                        page_content=native_text,
                        metadata={
                            "source": str(self.file_path),
                            "page": page_num,
                            "extraction_method": "native",
                            "doc_type": "document",
                        },
                    )
                else:
                    # Página escaneada — usar OCR
                    print(f"   📷 Página {page_num}: usando OCR")
                    ocr_docs = self._ocr_single_page(page, page_num, len(pdf))
                    documents.extend(ocr_docs)
                    continue

                documents.append(doc)

        return documents

    def _ocr_single_page(
        self, page: fitz.Page, page_num: int, total: int
    ) -> list[Document]:
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(image, config=OCRPDFLoader.OCR_CONFIG)

        if not text.strip():
            return []

        return [Document(
            page_content=text,
            metadata={
                "source": str(self.file_path),
                "page": page_num,
                "total_pages": total,
                "extraction_method": "ocr",
                "doc_type": "document",
            },
        )]
```

---

## 3. Loader Universal con Unstructured

```python
# loaders/universal_loader.py
"""
Unstructured es el loader más completo: maneja PDF, DOCX, HTML,
preserva tablas, listas, headers y estructura del documento.
Ideal para documentos legales con estructura compleja.
"""
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document


def load_with_unstructured(file_path: str) -> list[Document]:
    """
    Loader universal que preserva elementos estructurales.
    Detecta: Title, NarrativeText, ListItem, Table, Header, Footer.
    """
    loader = UnstructuredLoader(
        file_path,
        mode="elements",           # Preserva cada elemento por separado
        strategy="hi_res",         # Alta resolución para PDFs complejos
        languages=["spa", "eng"],  # OCR bilingüe
        include_metadata=True,
    )
    elements = loader.load()

    # Enriquecer metadata con tipo de elemento
    for doc in elements:
        doc.metadata["element_type"] = doc.metadata.get("category", "unknown")
        # Marcar artículos legales por su patrón
        if _is_document_section(doc.page_content):
            doc.metadata["is_article"] = True
            doc.metadata["article_number"] = _extract_article_number(doc.page_content)

    return elements


def _is_document_section(text: str) -> bool:
    """Detecta si el texto corresponde a una sección del documento."""
    import re
    return bool(re.match(r"^(Art[íi]culo|ARTÍCULO|Art\.)\s+\d+", text.strip()))


def _extract_article_number(text: str) -> str | None:
    """Extrae el número de artículo del texto."""
    import re
    match = re.search(r"(?:Art[íi]culo|Art\.)\s+(\d+)", text, re.IGNORECASE)
    return match.group(1) if match else None
```

---

## Cuándo Usar Cada Loader

| Situación | Loader recomendado |
|-----------|-------------------|
| PDF digital, texto seleccionable, estructura simple | `DocumentPDFLoader` (PyMuPDF) |
| PDF escaneado o imagen | `OCRPDFLoader` |
| PDF mixto (algunas páginas escaneadas) | `SmartPDFLoader` ✅ |
| Documentos con tablas, headers, listas anidadas | `UnstructuredLoader` con `hi_res` |
| Prototipo rápido | `PyPDFLoader` de LangChain |
| DOCX, HTML, Email, Excel | `UnstructuredLoader` universal |
