from uuid import uuid4

from src.database.connection import ConnectionFactory
from src.database.unit_of_work import UnitOfWork


def test_unit_of_work_rollback():
    factory = ConnectionFactory()
    document_id = uuid4()

    with UnitOfWork(factory) as uow:
        uow.connection.execute(
            """
            INSERT INTO documents (
                id,
                file_name,
                file_type,
                storage_path
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                document_id,
                "test.pdf",
                "pdf",
                "/test/test.pdf",
            ),
        )

        # No commit

    with factory.create() as conn:
        result = conn.execute(
            "SELECT id FROM documents WHERE id = %s",
            (document_id,),
        ).fetchone()

        assert result is None
