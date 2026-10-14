import math
from uuid import UUID

from pgvector import Vector
from psycopg import Connection
from psycopg.rows import dict_row

from src.database.search.search_result import (
    SearchResult,
    search_result_from_row,
)


class VectorSearch:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def search(
        self,
        query_vector: list[float],
        *,
        model_name: str,
        model_version: str = "default",
        top_k: int = 200,
        min_similarity: float | None = None,
        document_id: UUID | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if not query_vector:
            raise ValueError("query_vector cannot be empty.")

        if not all(math.isfinite(v) for v in query_vector):
            raise ValueError("query_vector contains invalid values.")

        if not any(v != 0 for v in query_vector):
            raise ValueError("query_vector cannot be a zero vector.")

        if min_similarity is not None and (
            not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1
        ):
            raise ValueError("min_similarity must be between -1 and 1.")

        vector = Vector(query_vector)

        sql = """
            SELECT
                c.id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.token_count,
                c.chunk_type,
                c.page_start,
                c.page_end,
                c.section_title,
                c.metadata,
                1 - (e.embedding <=> %s) AS score
            FROM chunk_embeddings AS e
            JOIN chunks AS c
                ON c.id = e.chunk_id
            WHERE e.model_name = %s
              AND e.model_version = %s
              AND e.dimension = %s
        """

        params = [
            vector,
            model_name,
            model_version,
            len(query_vector),
        ]

        if document_id is not None:
            sql += " AND c.document_id = %s"
            params.append(document_id)

        if min_similarity is not None:
            sql += " AND 1 - (e.embedding <=> %s) >= %s"
            params.extend([vector, min_similarity])

        sql += """
            ORDER BY e.embedding <=> %s ASC, c.id
            LIMIT %s
        """

        params.extend([vector, top_k])

        with self._conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

        return [search_result_from_row(row) for row in rows]
