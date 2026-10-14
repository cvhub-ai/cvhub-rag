from typing import Any
from uuid import UUID

from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


class DocumentRepository:
    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def insert(
        self,
        *,
        document_id: UUID,
        file_name: str,
        file_type: str,
        storage_path: str,
        checksum: str | None = None,
        parser_name: str | None = None,
        parser_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (
                    id,
                    file_name,
                    file_type,
                    storage_path,
                    checksum,
                    parser_name,
                    parser_version,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    document_id,
                    file_name,
                    file_type,
                    storage_path,
                    checksum,
                    parser_name,
                    parser_version,
                    Jsonb(metadata if metadata is not None else {}),
                ),
            )

    def get_by_id(self, document_id: UUID) -> dict[str, Any] | None:
        with self._conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT *
                FROM documents
                WHERE id = %s
                """,
                (document_id,),
            )

            return cursor.fetchone()
