from uuid import uuid4

from src.database.connection import ConnectionFactory
from src.database.repositories.chunk_repository import ChunkRepository
from src.database.repositories.document_repository import DocumentRepository
from src.database.repositories.embedding_repository import EmbeddingRepository
from src.database.unit_of_work import UnitOfWork
from src.models.chunk import Chunk, ChunkType


def test_get_unembedded_chunks():
    factory = ConnectionFactory()
    document_id = uuid4()

    chunks = [
        Chunk(
            id=uuid4(),
            document_id=document_id,
            chunk_index=i,
            content=f"Test chunk {i}",
            token_count=4,
            chunk_type=ChunkType.TEXT,
        )
        for i in range(3)
    ]

    with UnitOfWork(factory) as uow:
        document_repo = DocumentRepository(uow.connection)
        chunk_repo = ChunkRepository(uow.connection)
        embedding_repo = EmbeddingRepository(uow.connection)

        # Insert document
        document_repo.insert(
            document_id=document_id,
            file_name="test.pdf",
            file_type="pdf",
            storage_path="/test/test.pdf",
        )

        # Insert chunks
        chunk_repo.insert_many(chunks)

        # Fetch first batch
        batch = chunk_repo.get_unembedded_chunks(
            model_name="test-model",
            batch_size=2,
        )

        assert len(batch) == 2

        # Save embeddings for first batch
        embedding_repo.upsert_many(
            [(chunk.id, [0.1, 0.2, 0.3]) for chunk in batch],
            model_name="test-model",
        )

        # Fetch remaining chunks
        remaining = chunk_repo.get_unembedded_chunks(
            model_name="test-model",
            batch_size=2,
        )

        assert len(remaining) == 1
        assert remaining[0].id not in {c.id for c in batch}

        # Save the final embedding
        embedding_repo.upsert_many(
            [(remaining[0].id, [0.4, 0.5, 0.6])],
            model_name="test-model",
        )

        # All chunks have embeddings
        result = chunk_repo.get_unembedded_chunks(
            model_name="test-model",
        )

        assert result == []

        # No commit: automatically rollback
