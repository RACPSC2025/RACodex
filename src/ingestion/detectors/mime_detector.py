"""
Detector de MIME type por contenido de bytes.

Por qué no usar la extensión del archivo:
  - Un PDF con extensión .doc es un PDF, no un Word
  - Archivos subidos por usuarios frecuentemente tienen extensiones incorrectas
  - python-magic lee el magic number de los primeros bytes (igual que `file` en Unix)

Requiere:
  pip install python-magic
  # Linux: apt-get install libmagic1
  # macOS: brew install libmagic
  # Windows: pip install python-magic-bin (incluye la dll)

MIME types que maneja este sistema:
  application/pdf
  application/vnd.openxmlformats-officedocument.wordprocessingml.document  → .docx
  application/msword                                                         → .doc (legacy)
  application/vnd.openxmlformats-officedocument.spreadsheetml.sheet        → .xlsx
  application/vnd.ms-excel                                                  → .xls (legacy)
  image/jpeg, image/png, image/tiff, image/webp                             → imágenes directas
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.logging import get_logger
from src.ingestion.base import UnsupportedFormatError

log = get_logger(__name__)


# ─── MIME types soportados ────────────────────────────────────────────────────

SUPPORTED_MIME_TYPES: frozenset[str] = frozenset({
    "application/pdf",
    # Word
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    # Excel
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    # Imágenes (para PDFs que son fotografías)
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
    "image/bmp",
})

# Alias legibles para logging y mensajes de error
MIME_LABELS: dict[str, str] = {
    "application/pdf": "PDF",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word (.docx)",
    "application/msword": "Word (.doc, legacy)",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel (.xlsx)",
    "application/vnd.ms-excel": "Excel (.xls, legacy)",
    "image/jpeg": "Imagen JPEG",
    "image/png": "Imagen PNG",
    "image/tiff": "Imagen TIFF",
    "image/webp": "Imagen WebP",
    "image/bmp": "Imagen BMP",
}


# ─── Resultado de detección ───────────────────────────────────────────────────

@dataclass(frozen=True)
class MimeDetectionResult:
    """Resultado inmutable de la detección de MIME type."""
    path: Path
    mime_type: str
    is_supported: bool
    label: str

    @property
    def is_pdf(self) -> bool:
        return self.mime_type == "application/pdf"

    @property
    def is_word(self) -> bool:
        return self.mime_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }

    @property
    def is_excel(self) -> bool:
        return self.mime_type in {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        }

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")

    def __str__(self) -> str:
        return f"{self.path.name} → {self.label} ({self.mime_type})"


# ─── Detector ─────────────────────────────────────────────────────────────────

class MimeDetector:
    """
    Detecta el MIME type real de un archivo leyendo sus bytes.

    Singleton-friendly: una instancia puede reutilizarse para N archivos.
    El import de `magic` es lazy para no romper el arranque si la librería
    no está instalada (da un error claro en tiempo de uso, no de import).
    """

    def __init__(self) -> None:
        self._magic_instance: object | None = None

    def _get_magic(self) -> object:
        """Lazy import de python-magic con error descriptivo."""
        if self._magic_instance is None:
            try:
                import magic  # noqa: PLC0415
                self._magic_instance = magic.Magic(mime=True)
            except ImportError as exc:
                raise ImportError(
                    "python-magic no está instalado. "
                    "Ejecuta: pip install python-magic\n"
                    "Linux: sudo apt-get install libmagic1\n"
                    "macOS: brew install libmagic"
                ) from exc
        return self._magic_instance

    def detect(self, path: Path) -> MimeDetectionResult:
        """
        Detecta el MIME type de un archivo por su contenido.

        Args:
            path: Ruta al archivo. Debe existir.

        Returns:
            MimeDetectionResult con el MIME detectado y metadatos.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            UnsupportedFormatError: Si el MIME no está en SUPPORTED_MIME_TYPES.
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        if not path.is_file():
            raise ValueError(f"La ruta no es un archivo: {path}")

        magic_instance = self._get_magic()
        mime_type: str = magic_instance.from_file(str(path))  # type: ignore[attr-defined]

        # Algunos magic devuelven parámetros extra: "text/plain; charset=utf-8"
        mime_type = mime_type.split(";")[0].strip()

        is_supported = mime_type in SUPPORTED_MIME_TYPES
        label = MIME_LABELS.get(mime_type, f"Desconocido ({mime_type})")

        result = MimeDetectionResult(
            path=path,
            mime_type=mime_type,
            is_supported=is_supported,
            label=label,
        )

        log.debug(
            "mime_detected",
            file=path.name,
            mime=mime_type,
            supported=is_supported,
        )

        if not is_supported:
            raise UnsupportedFormatError(
                f"Formato no soportado: {label} para '{path.name}'. "
                f"Formatos aceptados: {', '.join(sorted(SUPPORTED_MIME_TYPES))}",
                path=path,
            )

        return result

    def detect_many(
        self,
        paths: list[Path],
        *,
        skip_unsupported: bool = True,
    ) -> list[MimeDetectionResult]:
        """
        Detecta MIME types de múltiples archivos.

        Args:
            paths: Lista de rutas a analizar.
            skip_unsupported: Si True, omite formatos no soportados con un warning.
                              Si False, lanza UnsupportedFormatError al primer fallo.

        Returns:
            Lista de resultados, solo para archivos soportados si skip_unsupported=True.
        """
        results: list[MimeDetectionResult] = []

        for path in paths:
            try:
                result = self.detect(path)
                results.append(result)
            except UnsupportedFormatError as exc:
                if not skip_unsupported:
                    raise
                log.warning("unsupported_file_skipped", file=str(path), reason=str(exc))
            except FileNotFoundError:
                log.warning("file_not_found_skipped", file=str(path))

        return results


# ─── Instancia por defecto ────────────────────────────────────────────────────

_default_detector: MimeDetector | None = None


def get_mime_detector() -> MimeDetector:
    """Retorna la instancia singleton del detector (lazy init)."""
    global _default_detector  # noqa: PLW0603
    if _default_detector is None:
        _default_detector = MimeDetector()
    return _default_detector


def detect_mime(path: Path | str) -> MimeDetectionResult:
    """
    Función de conveniencia: detecta el MIME de un único archivo.

    Usa la instancia singleton del detector.

    Args:
        path: Ruta al archivo.

    Returns:
        MimeDetectionResult

    Raises:
        FileNotFoundError, UnsupportedFormatError
    """
    return get_mime_detector().detect(Path(path))
