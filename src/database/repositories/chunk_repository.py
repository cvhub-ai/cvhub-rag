import hashlib
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from src.models.chunk import Chunk, ChunkType


class ChunkRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def insert_many(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        values = []

        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata["source_block_ids"] = [
                str(block_id) for block_id in chunk.source_block_ids
            ]

            content_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()

            values.append(
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.content,
                    content_hash,
                    chunk.token_count,
                    chunk.chunk_type.value,
                    chunk.page_start,
                    chunk.page_end,
                    chunk.section_title,
                    Jsonb(metadata),
                )
            )

        with self._conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO chunks (
                    id,
                    document_id,
                    chunk_index,
                    content,
                    content_hash,
                    token_count,
                    chunk_type,
                    page_start,
                    page_end,
                    section_title,
                    metadata
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                """,
                values,
            )

    def get_by_document(self, document_id: UUID) -> list[Chunk]:
        with self._conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT *
                FROM chunks
                WHERE document_id = %s
                ORDER BY chunk_index
                """,
                (document_id,),
            )

            rows = cursor.fetchall()

        chunks = []

        for row in rows:
            metadata = dict(row["metadata"])
            source_block_ids = metadata.pop("source_block_ids", [])

            chunk = Chunk(
                id=row["id"],
                document_id=row["document_id"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                token_count=row["token_count"],
                chunk_type=ChunkType(row["chunk_type"]),
                page_start=row["page_start"],
                page_end=row["page_end"],
                section_title=row["section_title"],
                source_block_ids=[UUID(block_id) for block_id in source_block_ids],
                metadata=metadata,
            )

            chunks.append(chunk)

        return chunks

    def get_unembedded_chunks(
        self,
        *,
        model_name: str,
        model_version: str = "default",
        batch_size: int = 100,
    ) -> list[Chunk]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        with self._conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT c.*
                FROM chunks c
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM chunk_embeddings e
                    WHERE e.chunk_id = c.id
                    AND e.model_name = %s
                    AND e.model_version = %s
                )
                ORDER BY c.id
                LIMIT %s
                """,
                (model_name, model_version, batch_size),
            )

            rows = cursor.fetchall()

        return [self._row_to_chunk(row) for row in rows]

    @staticmethod
    def _row_to_chunk(row: dict) -> Chunk:
        metadata = dict(row["metadata"])
        source_block_ids = metadata.pop("source_block_ids", [])

        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            token_count=row["token_count"],
            chunk_type=ChunkType(row["chunk_type"]),
            page_start=row["page_start"],
            page_end=row["page_end"],
            section_title=row["section_title"],
            source_block_ids=[UUID(block_id) for block_id in source_block_ids],
            metadata=metadata,
        )
