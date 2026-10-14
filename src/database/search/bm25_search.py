from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row

from src.database.search.search_result import (
    SearchResult,
    search_result_from_row,
)


class BM25Search:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def search(
        self,
        query: str,
        *,
        top_k: int = 300,
        document_id: UUID | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if not query.strip():
            return []

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
                pdb.score(c.id) AS score
            FROM chunks AS c
            WHERE c.content ||| %s
        """

        params: list[str | UUID | int] = [query]

        if document_id is not None:
            sql += " AND c.document_id = %s"
            params.append(document_id)

        sql += """
            ORDER BY pdb.score(c.id) DESC, c.id
            LIMIT %s
        """

        params.append(top_k)

        with self._conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

        return [search_result_from_row(row) for row in rows]
