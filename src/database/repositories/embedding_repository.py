from typing import Any
from uuid import UUID, uuid4

from pgvector import Vector
from psycopg import Connection
from psycopg.rows import dict_row


class EmbeddingRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def upsert_many(
        self,
        embeddings: list[tuple[UUID, list[float]]],
        *,
        model_name: str,
        model_version: str = "default",
    ) -> None:
        if not embeddings:
            return

        values = []

        for chunk_id, embedding in embeddings:
            if not embedding:
                raise ValueError("Embedding cannot be empty.")

            values.append(
                (
                    uuid4(),
                    chunk_id,
                    model_name,
                    model_version,
                    len(embedding),
                    Vector(embedding),
                )
            )

        with self._conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO chunk_embeddings (
                    id,
                    chunk_id,
                    model_name,
                    model_version,
                    dimension,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (
                    chunk_id,
                    model_name,
                    model_version
                )
                DO UPDATE SET
                    dimension = EXCLUDED.dimension,
                    embedding = EXCLUDED.embedding
                """,
                values,
            )

    def get_by_chunk(
        self,
        chunk_id: UUID,
        *,
        model_name: str,
        model_version: str = "default",
    ) -> dict[str, Any] | None:
        with self._conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT *
                FROM chunk_embeddings
                WHERE chunk_id = %s
                  AND model_name = %s
                  AND model_version = %s
                """,
                (chunk_id, model_name, model_version),
            )

            return cursor.fetchone()
