"""
Loader de hojas de cálculo Excel (.xlsx / .xls legacy).

Contexto legal colombiano:
  Las entidades públicas frecuentemente publican tablas normativas en Excel:
  - Tablas de sanciones y multas (Art. X → Sanción → Monto)
  - Listados de EPS/ARL habilitadas
  - Matrices de riesgos laborales
  - Resoluciones con anexos tabulares

Estrategia de conversión sheet → Document:
  Cada hoja de cálculo se procesa de forma independiente.
  Las filas se convierten a texto en dos formatos alternativos:

  1. Formato "fila como párrafo" (default):
     Columna1: Valor1 | Columna2: Valor2 | Columna3: Valor3

  2. Formato "tabla Markdown" (alternativo):
     | Columna1 | Columna2 | Columna3 |
     |----------|----------|----------|
     | Valor1   | Valor2   | Valor3   |

  El formato "fila como párrafo" funciona mejor para RAG porque:
  - Cada fila es un chunk semánticamente completo
  - El LLM ve el nombre de la columna junto al valor
  - Evita chunks de tabla cortados a la mitad

Requiere: openpyxl >= 3.1, pandas >= 2.2
  pip install openpyxl pandas
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.ingestion.base import BaseLoader, IngestionError, LoaderUnavailableError
from src.ingestion.processors.metadata_extractor import get_metadata_extractor

log = get_logger(__name__)

# Número máximo de filas por chunk (para tablas muy largas)
_DEFAULT_ROWS_PER_CHUNK = 20

# Máximo de filas a procesar por hoja (protección de memoria)
_MAX_ROWS_PER_SHEET = 10_000


class ExcelLoader(BaseLoader):
    """
    Loader de hojas de cálculo Excel para normativa y datos legales.

    Convierte cada hoja en uno o varios Documents preservando
    el contexto de columnas en cada chunk.
    """

    def __init__(
        self,
        rows_per_chunk: int = _DEFAULT_ROWS_PER_CHUNK,
        max_rows_per_sheet: int = _MAX_ROWS_PER_SHEET,
        output_format: str = "row_paragraph",   # "row_paragraph" | "markdown_table"
        skip_empty_sheets: bool = True,
        sheet_names: list[str] | None = None,   # None = todas las hojas
    ) -> None:
        """
        Args:
            rows_per_chunk: Filas agrupadas por chunk (mayor = chunks más largos).
            max_rows_per_sheet: Límite de filas por hoja (protección de memoria).
            output_format: "row_paragraph" (recomendado RAG) o "markdown_table".
            skip_empty_sheets: Ignorar hojas sin datos.
            sheet_names: Lista de nombres de hojas a procesar. None = todas.
        """
        self._rows_per_chunk = rows_per_chunk
        self._max_rows = max_rows_per_sheet
        self._output_format = output_format
        self._skip_empty = skip_empty_sheets
        self._sheet_names = sheet_names

    @property
    def loader_type(self) -> str:
        return "excel"

    def supports(self, path: Path, mime_type: str) -> bool:
        return mime_type in {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        }

    def _get_pandas(self) -> object:
        try:
            import pandas as pd  # noqa: PLC0415
            return pd
        except ImportError as exc:
            raise LoaderUnavailableError(
                "pandas no está instalado.\n"
                "Ejecuta: pip install pandas openpyxl"
            ) from exc

    def _get_openpyxl(self) -> object:
        try:
            import openpyxl  # noqa: PLC0415
            return openpyxl
        except ImportError as exc:
            raise LoaderUnavailableError(
                "openpyxl no está instalado.\n"
                "Ejecuta: pip install openpyxl"
            ) from exc

    def _read_sheets(self, path: Path) -> dict[str, object]:
        """
        Lee todas las hojas (o las especificadas) del archivo Excel.

        Returns:
            Dict {nombre_hoja: DataFrame}
        """
        pd = self._get_pandas()

        try:
            # engine="openpyxl" para .xlsx, "xlrd" para .xls legacy
            engine = "openpyxl" if path.suffix.lower() == ".xlsx" else "xlrd"
            xl = pd.ExcelFile(str(path), engine=engine)
        except Exception as exc:
            raise IngestionError(
                f"No se pudo abrir el Excel: {path.name}",
                path=path,
                cause=exc,
            ) from exc

        sheets_to_read = self._sheet_names or xl.sheet_names
        result = {}

        for sheet_name in sheets_to_read:
            try:
                df = pd.read_excel(
                    xl,
                    sheet_name=sheet_name,
                    nrows=self._max_rows,
                    dtype=str,             # todo como string — evita conversiones numéricas
                )
                # Limpiar: reemplazar NaN con string vacío
                df = df.fillna("")
                # Eliminar filas completamente vacías
                df = df[df.any(axis=1)]

                if self._skip_empty and df.empty:
                    log.debug("excel_sheet_skipped_empty", sheet=sheet_name)
                    continue

                result[sheet_name] = df

            except Exception as exc:
                log.warning(
                    "excel_sheet_read_failed",
                    sheet=sheet_name,
                    file=path.name,
                    error=str(exc),
                )

        return result

    def _rows_to_paragraph(
        self,
        df: object,
        columns: list[str],
        rows: list[dict],
    ) -> str:
        """
        Convierte un grupo de filas al formato "Columna: Valor | ...".

        Ejemplo:
          Artículo: 22 | Descripción: Obligaciones del empleador | Sanción: 20 SMLV
          Artículo: 23 | Descripción: Sistema de Gestión SST | Sanción: 50 SMLV
        """
        pd = self._get_pandas()
        lines = []
        for row in rows:
            parts = []
            for col in columns:
                val = str(row.get(col, "")).strip()
                if val and val.lower() not in {"nan", "none", ""}:
                    parts.append(f"{col}: {val}")
            if parts:
                lines.append(" | ".join(parts))
        return "\n".join(lines)

    def _rows_to_markdown_table(
        self,
        columns: list[str],
        rows: list[dict],
    ) -> str:
        """
        Convierte un grupo de filas a tabla Markdown GFM.
        """
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        data_rows = [
            "| " + " | ".join(
                str(row.get(col, "")).strip() for col in columns
            ) + " |"
            for row in rows
        ]
        return "\n".join([header, separator, *data_rows])

    def _sheet_to_documents(
        self,
        df: object,
        sheet_name: str,
        source_path: Path,
        sheet_index: int,
    ) -> list[Document]:
        """
        Convierte un DataFrame a lista de Documents.

        Agrupa filas en chunks de _rows_per_chunk para balancear
        tamaño vs contexto.
        """
        pd = self._get_pandas()
        columns = list(df.columns)
        records = df.to_dict("records")
        extractor = get_metadata_extractor()
        documents: list[Document] = []

        # Dividir en grupos de _rows_per_chunk
        for chunk_start in range(0, len(records), self._rows_per_chunk):
            chunk_rows = records[chunk_start : chunk_start + self._rows_per_chunk]
            chunk_idx = chunk_start // self._rows_per_chunk

            if self._output_format == "markdown_table":
                content = self._rows_to_markdown_table(columns, chunk_rows)
            else:
                content = self._rows_to_paragraph(df, columns, chunk_rows)

            if not content.strip():
                continue

            # Prefijar con nombre de hoja para contexto
            row_range = f"{chunk_start + 1}–{min(chunk_start + self._rows_per_chunk, len(records))}"
            header = f"[Hoja: {sheet_name} | Filas: {row_range} | Fuente: {source_path.name}]\n"
            full_content = header + content

            # Metadata
            meta = extractor.extract(
                chunk_text=full_content,
                source_path=source_path,
                chunk_index=sheet_index * 1000 + chunk_idx,
                loader_type=self.loader_type,
            )
            # Agregar metadata específica de Excel
            meta_dict = meta.to_dict()
            meta_dict["sheet_name"] = sheet_name
            meta_dict["row_start"] = str(chunk_start + 1)
            meta_dict["row_end"] = str(min(chunk_start + self._rows_per_chunk, len(records)))
            meta_dict["columns"] = ", ".join(columns[:10])  # primeras 10 columnas

            documents.append(Document(
                page_content=full_content,
                metadata=meta_dict,
            ))

        return documents

    def load(self, path: Path) -> list[Document]:
        """
        Carga un Excel y retorna sus hojas como Documents.

        Args:
            path: Ruta al .xlsx o .xls. Debe existir.

        Returns:
            Lista de Documents, uno o más por hoja según tamaño.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            IngestionError: Si el archivo no puede procesarse.
            LoaderUnavailableError: Si pandas/openpyxl no están instalados.
        """
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Excel no encontrado: {path}")

        log.info("excel_loading", file=path.name)

        sheets = self._read_sheets(path)

        if not sheets:
            raise IngestionError(
                f"El Excel '{path.name}' no contiene hojas con datos.",
                path=path,
            )

        all_documents: list[Document] = []

        for sheet_idx, (sheet_name, df) in enumerate(sheets.items()):
            log.debug(
                "excel_processing_sheet",
                sheet=sheet_name,
                rows=len(df),
                cols=len(df.columns),
            )
            sheet_docs = self._sheet_to_documents(df, sheet_name, path, sheet_idx)
            all_documents.extend(sheet_docs)

        log.info(
            "excel_loaded",
            file=path.name,
            sheets=len(sheets),
            total_chunks=len(all_documents),
        )

        return all_documents


def get_excel_loader(**kwargs) -> ExcelLoader:
    """Factory function para ExcelLoader."""
    return ExcelLoader(**kwargs)
