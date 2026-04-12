"""
Document Augmentation — Genera preguntas por chunk durante la ingestión.

Implementa procesamiento por LOTES (batching) nativo.
Cada bloque de N chunks se agrupa en una única inferencia al LLM, reduciendo
drásticamente la latencia y los costos de API.

Uso en el pipeline de ingestión:
    from src.ingestion.processors.document_augmenter import augment_documents

    chunks = chunker.chunk(docs)
    augmented = augment_documents(chunks, llm=llm, questions_per_chunk=3, batch_size=5)
    vector_store.add_documents(augmented)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
import traceback

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.logging import get_logger
from src.config.providers import get_llm

log = get_logger(__name__)


# ─── Prompt ──────────────────────────────────────────────────────────────────

BATCH_QUESTION_GENERATION_PROMPT = SystemMessage(
    content=(
        "Eres un experto analizando fragmentos de documentos para generar preguntas "
        "reales que los usuarios buscarían en Google o en un sistema RAG para "
        "encontrar dicha información.\n\n"
        "INSTRUCCIONES CRÍTICAS:\n"
        "1. Recibirás un lote de fragmentos numerados (ej. [CHUNK 0], [CHUNK 1]).\n"
        "2. Genera el número exacto de preguntas solicitado para CADA fragmento.\n"
        "3. Lenguaje natural, sin jerga excesiva ni prefijos.\n"
        "4. Tienes PROHIBIDO generar cualquier texto que no sea un JSON válido.\n\n"
        "FORMATO DE RESPUESTA (Solo JSON, sin bloques de código markdown):\n"
        "{\n"
        '  "results": [\n'
        '    {"chunk_id": 0, "questions": ["pregunta 1?", "pregunta 2?"]},\n'
        '    {"chunk_id": 1, "questions": ["pregunta 1?", "pregunta 2?"]}\n'
        "  ]\n"
        "}"
    )
)


# ─── Función principal (Síncrona) ────────────────────────────────────────────

def augment_documents(
    chunks: list[Document],
    llm=None,
    questions_per_chunk: int = 3,
    batch_size: int = 5,
) -> list[Document]:
    """
    Augmenta cada chunk con preguntas usando Batching Síncrono.
    """
    llm = llm or get_llm(temperature=0.3)
    augmented: list[Document] = list(chunks)  # Copia intocada de los originales

    # Dividir todos los chunks en lotes de tamaño batch_size
    batches = [
        list(enumerate(chunks))[i : i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    log.info(
        "sync_batch_augmentation_start",
        chunks=len(chunks),
        batch_size=batch_size,
        total_batches=len(batches),
    )

    for batch_idx, batch in enumerate(batches):
        try:
            results_dict = _generate_questions_batch(batch, llm, questions_per_chunk)
            # Procesar el resultado procesado
            for local_idx, (global_idx, chunk) in enumerate(batch):
                questions = results_dict.get(local_idx, [])
                for q_text in questions:
                    augmented.append(Document(
                        page_content=q_text,
                        metadata={
                            **chunk.metadata,
                            "augmentation_type": "question",
                            "parent_chunk_index": chunk.metadata.get("chunk_index", global_idx),
                            "is_augmented_question": True,
                        },
                    ))
            
            log.debug("sync_batch_processed", batch=batch_idx+1, total=len(batches))
            
        except Exception as exc:
            log.warning(
                "batch_augmentation_failed",
                batch=batch_idx,
                error=str(exc),
            )

    return augmented


# ─── Versión Async con concurrencia ──────────────────────────────────────────

async def augment_documents_async(
    chunks: list[Document],
    llm=None,
    questions_per_chunk: int = 3,
    batch_size: int = 5,
    max_concurrency: int = 5,
) -> list[Document]:
    """
    Augmenta cada chunk usando Batching Asíncrono hiper-optimiado.
    
    Toma los chunks en lotes de `batch_size` e invoca hasta `max_concurrency` 
    lotes al mismo tiempo usando un semáforo.
    """
    llm = llm or get_llm(temperature=0.3)
    semaphore = asyncio.Semaphore(max_concurrency)

    batches = [
        list(enumerate(chunks))[i : i + batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    log.info(
        "async_batch_augmentation_start",
        chunks=len(chunks),
        batch_size=batch_size,
        max_concurrency=max_concurrency,
        total_batches=len(batches),
    )

    async def _augment_one_batch(batch: list[tuple[int, Document]], b_idx: int) -> list[Document]:
        async with semaphore:
            try:
                results_dict = await _generate_questions_batch_async(batch, llm, questions_per_chunk)
                batch_docs = []
                for local_idx, (global_idx, chunk) in enumerate(batch):
                    questions = results_dict.get(local_idx, [])
                    for q_text in questions:
                        batch_docs.append(Document(
                            page_content=q_text,
                            metadata={
                                **chunk.metadata,
                                "augmentation_type": "question",
                                "parent_chunk_index": chunk.metadata.get("chunk_index", global_idx),
                                "is_augmented_question": True,
                            },
                        ))
                return batch_docs
            except Exception as exc:
                log.warning(
                    "async_batch_augmentation_failed",
                    batch_idx=b_idx,
                    error=str(exc),
                )
                return []

    tasks = [_augment_one_batch(batch, b_idx) for b_idx, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_questions: list[Document] = []
    for result in results:
        if isinstance(result, Exception):
            log.warning("async_augment_exception", error=str(result))
        elif isinstance(result, list):
            all_questions.extend(result)

    augmented = list(chunks) + all_questions

    log.info(
        "async_augmentation_complete",
        original_chunks=len(chunks),
        total_documents=len(augmented),
        questions_added=len(all_questions),
    )

    return augmented


# ─── Helpers (Batching Logic) ────────────────────────────────────────────────

def _build_batch_prompt(batch: list[tuple[int, Document]], n: int) -> str:
    """Construye el string del prompt agrupando múltiples chunks."""
    prompt_lines = [
        f"Genera exactamente {n} preguntas para cada uno de los siguientes {len(batch)} fragmentos."
    ]
    for local_idx, (global_idx, chunk) in enumerate(batch):
        snippet = chunk.page_content[:1500].replace('\n', ' ')
        prompt_lines.append(f"\n[CHUNK {local_idx}]\n{snippet}")
        
    return "\n".join(prompt_lines)


def _parse_batch_response(content: str) -> dict[int, list[str]]:
    """Convierte el JSON (esperado) del LLM en un diccionario local_idx -> preguntas."""
    try:
        raw_json = content.strip()
        
        # Extraer JSON si hay texto extra antes/después
        start = raw_json.find("{")
        end = raw_json.rfind("}") + 1
        if start >= 0 and end > start:
            raw_json = raw_json[start:end]

        data = json.loads(raw_json)
        results = data.get("results", [])
        
        parsed_dict = {}
        for r in results:
            c_id = r.get("chunk_id")
            qs = r.get("questions", [])
            if c_id is not None:
                parsed_dict[int(c_id)] = [str(q).strip() for q in qs if str(q).strip()]
                
        return parsed_dict
    except json.JSONDecodeError as exc:
        log.warning("batch_json_decode_error", content=content[:200], error=str(exc))
        return {}


def _generate_questions_batch(batch: list[tuple[int, Document]], llm, n: int) -> dict[int, list[str]]:
    prompt = _build_batch_prompt(batch, n)
    response = llm.invoke([
        BATCH_QUESTION_GENERATION_PROMPT,
        HumanMessage(content=prompt),
    ])
    return _parse_batch_response(response.content)


async def _generate_questions_batch_async(batch: list[tuple[int, Document]], llm, n: int) -> dict[int, list[str]]:
    prompt = _build_batch_prompt(batch, n)
    response = await llm.ainvoke([
        BATCH_QUESTION_GENERATION_PROMPT,
        HumanMessage(content=prompt),
    ])
    return _parse_batch_response(response.content)


# ─── Integración con pipeline de ingestión ───────────────────────────────────

async def augment_and_index_async(
    chunks: list[Document],
    vector_store,
    llm=None,
    questions_per_chunk: int = 3,
    batch_size: int = 5,
    max_concurrency: int = 5,
) -> dict[str, int]:
    augmented = await augment_documents_async(
        chunks,
        llm=llm,
        questions_per_chunk=questions_per_chunk,
        batch_size=batch_size,
        max_concurrency=max_concurrency,
    )

    originals = [d for d in augmented if not d.metadata.get("is_augmented_question")]
    questions = [d for d in augmented if d.metadata.get("is_augmented_question")]

    vector_store.add_documents(augmented)

    return {
        "original_chunks": len(originals),
        "questions_added": len(questions),
        "total_indexed": len(augmented),
    }


def augment_and_index(
    chunks: list[Document],
    vector_store,
    llm=None,
    questions_per_chunk: int = 3,
    batch_size: int = 5,
) -> dict[str, int]:
    augmented = augment_documents(
        chunks, 
        llm=llm, 
        questions_per_chunk=questions_per_chunk, 
        batch_size=batch_size
    )

    originals = [d for d in augmented if not d.metadata.get("is_augmented_question")]
    questions = [d for d in augmented if d.metadata.get("is_augmented_question")]

    vector_store.add_documents(augmented)

    return {
        "original_chunks": len(originals),
        "questions_added": len(questions),
        "total_indexed": len(augmented),
    }
