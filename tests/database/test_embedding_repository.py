from uuid import uuid4

import pytest

from src.database.connection import ConnectionFactory
from src.database.repositories.chunk_repository import ChunkRepository
from src.database.repositories.document_repository import DocumentRepository
from src.database.repositories.embedding_repository import EmbeddingRepository
from src.database.unit_of_work import UnitOfWork
from src.models.chunk import Chunk, ChunkType


@pytest.fixture
def db_chunk():
    factory = ConnectionFactory()
    document_id = uuid4()
    chunk_id = uuid4()

    with UnitOfWork(factory) as uow:
        document_repo = DocumentRepository(uow.connection)
        chunk_repo = ChunkRepository(uow.connection)

        document_repo.insert(
            document_id=document_id,
            file_name="test.pdf",
            file_type="pdf",
            storage_path="/test/test.pdf",
        )

        chunk_repo.insert_many(
            [
                Chunk(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=0,
                    content="Computer vision introduction.",
                    token_count=4,
                    chunk_type=ChunkType.TEXT,
                )
            ]
        )

        yield uow.connection, chunk_id

        # No commit: automatically rollback


def test_embedding_repository(db_chunk):
    conn, chunk_id = db_chunk
    repo = EmbeddingRepository(conn)

    # Insert embedding
    repo.upsert_many(
        [(chunk_id, [0.1, 0.2, 0.3])],
        model_name="test-model",
    )

    result = repo.get_by_chunk(
        chunk_id,
        model_name="test-model",
    )

    assert result is not None
    assert result["dimension"] == 3
    assert result["model_name"] == "test-model"
    assert result["embedding"].to_list() == pytest.approx([0.1, 0.2, 0.3])

    # Update existing embedding
    repo.upsert_many(
        [(chunk_id, [0.4, 0.5, 0.6])],
        model_name="test-model",
    )

    updated = repo.get_by_chunk(
        chunk_id,
        model_name="test-model",
    )

    assert updated is not None
    assert updated["id"] == result["id"]
    assert updated["embedding"].to_list() == pytest.approx([0.4, 0.5, 0.6])
