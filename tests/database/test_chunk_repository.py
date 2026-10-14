from uuid import uuid4

from src.database.connection import ConnectionFactory
from src.database.repositories.chunk_repository import ChunkRepository
from src.database.repositories.document_repository import DocumentRepository
from src.database.unit_of_work import UnitOfWork
from src.models.chunk import Chunk, ChunkType


def test_insert_and_get_chunks():
    factory = ConnectionFactory()

    document_id = uuid4()
    block_id = uuid4()

    chunks = [
        Chunk(
            id=uuid4(),
            document_id=document_id,
            chunk_index=0,
            content="Introduction to Computer Vision.",
            token_count=6,
            chunk_type=ChunkType.TEXT,
            page_start=1,
            page_end=1,
            section_title="Introduction",
            source_block_ids=[block_id],
            metadata={"language": "en"},
        ),
        Chunk(
            id=uuid4(),
            document_id=document_id,
            chunk_index=1,
            content="Object detection is a computer vision task.",
            token_count=9,
            chunk_type=ChunkType.TEXT,
            page_start=2,
            page_end=2,
            section_title="Object Detection",
        ),
    ]

    with UnitOfWork(factory) as uow:
        document_repo = DocumentRepository(uow.connection)
        chunk_repo = ChunkRepository(uow.connection)

        document_repo.insert(
            document_id=document_id,
            file_name="test.pdf",
            file_type="pdf",
            storage_path="/test/test.pdf",
        )

        chunk_repo.insert_many(chunks)

        results = chunk_repo.get_by_document(document_id)

        assert len(results) == 2
        assert results == chunks

        # No commit: automatically rollback

    # Verify that no test data remains
    with factory.create() as conn:
        document_repo = DocumentRepository(conn)
        chunk_repo = ChunkRepository(conn)

        assert document_repo.get_by_id(document_id) is None
        assert chunk_repo.get_by_document(document_id) == []
