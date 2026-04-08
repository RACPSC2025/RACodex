"""
Loader OCR para PDFs escaneados e imágenes directas.

Flujo para PDF escaneado:
  fitz.open() → por cada página → pdf_page_to_image()
  → OCRPreprocessor.preprocess() → EasyOCR.readtext()
  → reconstruir texto con bounding boxes → TextCleaner(ocr_output)
  → LegalChunker

Flujo para imagen directa (JPEG/PNG/TIFF):
  cv2.imread() → OCRPreprocessor.preprocess()
  → EasyOCR.readtext() → reconstruir texto → TextCleaner → LegalChunker

Decisiones de diseño:
  - EasyOCR se inicializa UNA SOLA VEZ (carga modelos ~500MB al primer uso)
    y se reutiliza via singleton. Inicializar en cada llamada añadiría ~8s/doc.
  - La reconstrucción de texto desde bounding boxes preserva el orden de
    lectura columnar (izquierda→derecha, arriba→abajo por bloques de Y).
  - Los confidence scores de EasyOCR se usan para filtrar texto de baja
    calidad (< OCR_CONFIDENCE_THRESHOLD) en lugar de incluirlo como ruido.

Requiere: easyocr, opencv-python-headless, pymupdf
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.ingestion.base import BaseLoader, IngestionError, LoaderUnavailableError
from src.ingestion.processors.legal_chunker import LegalChunker, get_legal_chunker
from src.ingestion.processors.ocr_preprocessor import OCRPreprocessor, get_ocr_preprocessor

if TYPE_CHECKING:
    import numpy as np

log = get_logger(__name__)

# Umbral de confianza: texto con score menor se descarta (es ruido OCR)
_OCR_CONFIDENCE_THRESHOLD = 0.4

# Tolerancia vertical para agrupar palabras en la misma línea (% del alto de página)
_LINE_GROUP_TOLERANCE = 0.015


class OCRLoader(BaseLoader):
    """
    Loader para PDFs escaneados e imágenes directas usando EasyOCR.

    Soporta:
      - application/pdf     (cuando quality_detector marca is_scanned=True)
      - image/jpeg, image/png, image/tiff, image/webp, image/bmp

    El loader inicializa EasyOCR de forma lazy y lo reutiliza entre llamadas.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        use_gpu: bool | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_pages: int = 400,
        render_dpi: int = 300,
        confidence_threshold: float = _OCR_CONFIDENCE_THRESHOLD,
    ) -> None:
        settings = get_settings()
        self._languages = languages or settings.ocr_languages
        self._use_gpu = use_gpu if use_gpu is not None else settings.ocr_gpu
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._max_pages = max_pages
        self._render_dpi = render_dpi
        self._confidence_threshold = confidence_threshold

        # Lazy — se inicializa al primer uso
        self._reader: object | None = None
        self._preprocessor: OCRPreprocessor | None = None
        self._chunker: LegalChunker | None = None

    @property
    def loader_type(self) -> str:
        return "ocr"

    def supports(self, path: Path, mime_type: str) -> bool:
        supported = {
            "application/pdf",
            "image/jpeg",
            "image/png",
            "image/tiff",
            "image/webp",
            "image/bmp",
        }
        return mime_type in supported

    # ── Lazy getters ──────────────────────────────────────────────────────────

    def _get_easyocr(self) -> object:
        """
        Inicializa EasyOCR una sola vez y reutiliza el reader.

        La carga de modelos tarda ~5-10s la primera vez.
        El singleton evita recargar en cada documento.
        """
        if self._reader is None:
            try:
                import easyocr  # noqa: PLC0415
            except ImportError as exc:
                raise LoaderUnavailableError(
                    "easyocr no está instalado.\n"
                    "Ejecuta: pip install easyocr"
                ) from exc

            log.info(
                "easyocr_initializing",
                languages=self._languages,
                gpu=self._use_gpu,
            )
            self._reader = easyocr.Reader(
                self._languages,
                gpu=self._use_gpu,
                verbose=False,
            )
            log.info("easyocr_ready")

        return self._reader

    def _get_preprocessor(self) -> OCRPreprocessor:
        if self._preprocessor is None:
            self._preprocessor = get_ocr_preprocessor()
        return self._preprocessor

    def _get_chunker(self) -> LegalChunker:
        if self._chunker is None:
            self._chunker = LegalChunker(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
            )
        return self._chunker

    def _get_fitz(self) -> object:
        try:
            import fitz  # noqa: PLC0415
            return fitz
        except ImportError as exc:
            raise LoaderUnavailableError(
                "PyMuPDF no está instalado. Ejecuta: pip install pymupdf"
            ) from exc

    # ── Core OCR ──────────────────────────────────────────────────────────────

    def _run_ocr_on_image(self, image: "np.ndarray") -> str:
        """
        Ejecuta EasyOCR sobre una imagen preprocesada y reconstruye el texto.

        Estrategia de reconstrucción:
          1. Ordenar resultados por posición Y (línea vertical)
          2. Agrupar palabras cercanas en la misma línea
          3. Dentro de cada línea, ordenar por X (izquierda → derecha)
          4. Separar líneas con \\n, palabras con espacio

        Args:
            image: ndarray preprocesado (output del OCRPreprocessor).

        Returns:
            Texto reconstruido como string.
        """
        reader = self._get_easyocr()

        results = reader.readtext(image, detail=1, paragraph=False)

        if not results:
            return ""

        # Filtrar por confianza
        filtered = [
            (bbox, text, conf)
            for bbox, text, conf in results
            if conf >= self._confidence_threshold and text.strip()
        ]

        if not filtered:
            log.warning("ocr_low_confidence_all_filtered", threshold=self._confidence_threshold)
            return ""

        # Reconstruir texto ordenado espacialmente
        return self._reconstruct_text(filtered, image.shape[0])

    def _reconstruct_text(
        self,
        ocr_results: list[tuple],
        page_height: int,
    ) -> str:
        """
        Reconstruye texto legible desde resultados de bounding boxes.

        Agrupa resultados en líneas por proximidad vertical, luego
        ordena cada línea de izquierda a derecha.
        """
        # Extraer centro Y de cada bounding box para agrupar en líneas
        items = []
        for bbox, text, conf in ocr_results:
            # bbox: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (4 puntos)
            y_coords = [point[1] for point in bbox]
            x_coords = [point[0] for point in bbox]
            center_y = sum(y_coords) / len(y_coords)
            center_x = sum(x_coords) / len(x_coords)
            items.append((center_y, center_x, text))

        # Ordenar por Y primero
        items.sort(key=lambda x: x[0])

        # Agrupar en líneas con tolerancia
        tolerance = page_height * _LINE_GROUP_TOLERANCE
        lines: list[list[tuple]] = []
        current_line: list[tuple] = []
        current_y = items[0][0] if items else 0

        for center_y, center_x, text in items:
            if abs(center_y - current_y) <= tolerance:
                current_line.append((center_x, text))
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [(center_x, text)]
                current_y = center_y

        if current_line:
            lines.append(current_line)

        # Dentro de cada línea, ordenar por X y unir con espacio
        reconstructed_lines = []
        for line in lines:
            line.sort(key=lambda x: x[0])
            line_text = " ".join(text for _, text in line)
            reconstructed_lines.append(line_text)

        return "\n".join(reconstructed_lines)

    # ── Loaders por tipo de archivo ───────────────────────────────────────────

    def _load_scanned_pdf(self, path: Path) -> tuple[str, int]:
        """
        Extrae texto de un PDF escaneado procesando cada página como imagen.

        Returns:
            (texto_completo, páginas_procesadas)
        """
        fitz = self._get_fitz()
        preprocessor = self._get_preprocessor()
        page_texts: list[str] = []
        pages_processed = 0

        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise IngestionError(
                f"No se pudo abrir el PDF: {path.name}",
                path=path,
                cause=exc,
            ) from exc

        total_pages = len(doc)
        log.info(
            "ocr_pdf_start",
            file=path.name,
            pages=total_pages,
            dpi=self._render_dpi,
        )

        try:
            for page_num, page in enumerate(doc, start=1):
                if page_num > self._max_pages:
                    log.warning("ocr_max_pages_reached", file=path.name, limit=self._max_pages)
                    break

                try:
                    # Renderizar página como imagen
                    image = preprocessor.pdf_page_to_image(page, dpi=self._render_dpi)

                    # Preprocesar para OCR
                    prep_result = preprocessor.preprocess(image)

                    # OCR
                    page_text = self._run_ocr_on_image(prep_result.image)

                    if page_text.strip():
                        page_texts.append(f"[Página {page_num}]\n{page_text}")
                        pages_processed += 1
                    else:
                        log.warning("ocr_empty_page", file=path.name, page=page_num)

                except Exception as exc:
                    log.warning(
                        "ocr_page_failed",
                        file=path.name,
                        page=page_num,
                        error=str(exc),
                    )
                    # Continúa con el resto de páginas

        finally:
            doc.close()

        return "\n\n".join(page_texts), pages_processed

    def _load_image_file(self, path: Path) -> tuple[str, int]:
        """
        Extrae texto de una imagen directa (JPEG, PNG, etc.).

        Returns:
            (texto_extraído, 1)  — siempre 1 página para imágenes
        """
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise LoaderUnavailableError(
                "opencv-python-headless no está instalado."
            ) from exc

        preprocessor = self._get_preprocessor()

        image = cv2.imread(str(path))
        if image is None:
            raise IngestionError(
                f"No se pudo cargar la imagen: {path.name}",
                path=path,
            )

        prep_result = preprocessor.preprocess(image)
        text = self._run_ocr_on_image(prep_result.image)

        if not text.strip():
            raise IngestionError(
                f"No se extrajo texto de la imagen '{path.name}'. "
                "Verifica que la imagen tenga suficiente resolución y contraste.",
                path=path,
            )

        return f"[Página 1]\n{text}", 1

    # ── Interface pública ─────────────────────────────────────────────────────

    def load(self, path: Path) -> list[Document]:
        """
        Carga un PDF escaneado o imagen y retorna chunks como Documents.

        Args:
            path: Ruta al archivo. Debe existir.

        Returns:
            Lista de Documents con texto extraído por OCR y metadata.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            IngestionError: Si el archivo no puede procesarse.
            LoaderUnavailableError: Si EasyOCR u OpenCV no están instalados.
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        # Determinar si es PDF o imagen directa por extensión/MIME
        suffix = path.suffix.lower()
        is_pdf = suffix == ".pdf"

        log.info("ocr_loading", file=path.name, is_pdf=is_pdf)

        if is_pdf:
            raw_text, pages_processed = self._load_scanned_pdf(path)
        else:
            raw_text, pages_processed = self._load_image_file(path)

        if not raw_text.strip():
            raise IngestionError(
                f"OCR no produjo texto en '{path.name}'.",
                path=path,
            )

        # Limpiar y chunkear con perfil OCR
        chunker = self._get_chunker()
        documents = chunker.chunk_with_profile(
            text=raw_text,
            source_path=path,
            loader_type=self.loader_type,
            cleaner_profile="ocr_output",
            add_header=True,
        )

        log.info(
            "ocr_loaded",
            file=path.name,
            pages=pages_processed,
            chunks=len(documents),
        )

        return documents


# ─── Singleton del reader EasyOCR ─────────────────────────────────────────────
# Compartido entre instancias de OCRLoader para evitar re-cargar modelos

_shared_ocr_loader: OCRLoader | None = None


def get_ocr_loader(**kwargs) -> OCRLoader:
    """
    Retorna una instancia de OCRLoader.

    Si no se pasan kwargs, reutiliza el singleton (recomendado para producción).
    Con kwargs, crea una instancia nueva con esa configuración.
    """
    global _shared_ocr_loader  # noqa: PLW0603

    if kwargs:
        return OCRLoader(**kwargs)

    if _shared_ocr_loader is None:
        _shared_ocr_loader = OCRLoader()

    return _shared_ocr_loader
