from uuid import uuid4

from src.database.connection import ConnectionFactory
from src.database.repositories.document_repository import DocumentRepository
from src.database.unit_of_work import UnitOfWork


def test_insert_and_get_document():
    factory = ConnectionFactory()
    document_id = uuid4()

    with UnitOfWork(factory) as uow:
        repo = DocumentRepository(uow.connection)

        repo.insert(
            document_id=document_id,
            file_name="test.pdf",
            file_type="pdf",
            storage_path="/test/test.pdf",
            checksum="test_checksum",
            parser_name="DoclingParser",
            parser_version="1.0",
            metadata={"source": "test"},
        )

        document = repo.get_by_id(document_id)

        assert document is not None
        assert document["id"] == document_id
        assert document["file_name"] == "test.pdf"
        assert document["file_type"] == "pdf"
        assert document["status"] == "pending"
        assert document["metadata"] == {"source": "test"}

        # No commit: UnitOfWork will roll back changes.

    with factory.create() as conn:
        repo = DocumentRepository(conn)
        assert repo.get_by_id(document_id) is None
