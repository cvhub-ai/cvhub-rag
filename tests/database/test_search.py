import math
from uuid import uuid4

import pytest

from src.database.connection import ConnectionFactory
from src.database.repositories.chunk_repository import ChunkRepository
from src.database.repositories.document_repository import DocumentRepository
from src.database.repositories.embedding_repository import EmbeddingRepository
from src.database.search.bm25_search import BM25Search
from src.database.search.vector_search import VectorSearch
from src.database.unit_of_work import UnitOfWork
from src.models.chunk import Chunk, ChunkType


@pytest.fixture
def search_data():
    factory = ConnectionFactory()

    document_id = uuid4()
    other_document_id = uuid4()

    # Unique values to avoid conflicts with existing data
    keyword = f"cvhubsearch{uuid4().hex}"
    model_name = f"test-model-{uuid4().hex}"

    chunks = [
        Chunk(
            id=uuid4(),
            document_id=document_id,
            chunk_index=0,
            content=f"{keyword} PostgreSQL database indexing.",
            token_count=6,
            chunk_type=ChunkType.TEXT,
        ),
        Chunk(
            id=uuid4(),
            document_id=document_id,
            chunk_index=1,
            content=f"{keyword} information retrieval.",
            token_count=5,
            chunk_type=ChunkType.TEXT,
        ),
        Chunk(
            id=uuid4(),
            document_id=document_id,
            chunk_index=2,
            content="Computer vision object detection.",
            token_count=5,
            chunk_type=ChunkType.TEXT,
        ),
    ]

    other_chunk = Chunk(
        id=uuid4(),
        document_id=other_document_id,
        chunk_index=0,
        content=f"{keyword} another document.",
        token_count=5,
        chunk_type=ChunkType.TEXT,
    )

    with UnitOfWork(factory) as uow:
        document_repo = DocumentRepository(uow.connection)
        chunk_repo = ChunkRepository(uow.connection)
        embedding_repo = EmbeddingRepository(uow.connection)

        # Insert documents
        for doc_id in (document_id, other_document_id):
            document_repo.insert(
                document_id=doc_id,
                file_name=f"{doc_id}.pdf",
                file_type="pdf",
                storage_path=f"/test/{doc_id}.pdf",
            )

        # Insert chunks
        chunk_repo.insert_many(chunks + [other_chunk])

        # Insert mock embeddings
        embedding_repo.upsert_many(
            [
                (chunks[0].id, [1.0, 0.0, 0.0]),
                (chunks[1].id, [0.8, 0.6, 0.0]),
                (chunks[2].id, [0.0, 1.0, 0.0]),
                (other_chunk.id, [1.0, 0.0, 0.0]),
            ],
            model_name=model_name,
        )

        yield {
            "conn": uow.connection,
            "document_id": document_id,
            "keyword": keyword,
            "model_name": model_name,
            "chunks": chunks,
        }

        # No commit: automatically rollback


def test_bm25_search(search_data):
    search = BM25Search(search_data["conn"])

    results = search.search(
        query=search_data["keyword"],
        document_id=search_data["document_id"],
        top_k=10,
    )

    # Two chunks contain the keyword
    assert len(results) == 2

    expected_ids = {
        search_data["chunks"][0].id,
        search_data["chunks"][1].id,
    }

    assert {result.chunk.id for result in results} == expected_ids

    # Results belong to the requested document
    assert all(
        result.chunk.document_id == search_data["document_id"] for result in results
    )

    # BM25 scores are valid and sorted descending
    scores = [result.score for result in results]

    assert all(math.isfinite(score) for score in scores)
    assert scores == sorted(scores, reverse=True)

    # Check Top K
    top_result = search.search(
        query=search_data["keyword"],
        document_id=search_data["document_id"],
        top_k=1,
    )

    assert len(top_result) == 1
    assert top_result[0].chunk.id == results[0].chunk.id


def test_vector_search(search_data):
    search = VectorSearch(search_data["conn"])

    results = search.search(
        query_vector=[1.0, 0.0, 0.0],
        model_name=search_data["model_name"],
        document_id=search_data["document_id"],
        top_k=10,
    )

    # All three chunks have embeddings
    assert len(results) == 3

    # First chunk is an exact vector match
    assert results[0].chunk.id == search_data["chunks"][0].id
    assert results[0].score == pytest.approx(1.0)

    # Results sorted by similarity
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)

    # Check minimum similarity
    filtered = search.search(
        query_vector=[1.0, 0.0, 0.0],
        model_name=search_data["model_name"],
        document_id=search_data["document_id"],
        min_similarity=0.75,
        top_k=10,
    )

    assert len(filtered) == 2
    assert all(result.score >= 0.75 for result in filtered)

    # Check Top K
    top_result = search.search(
        query_vector=[1.0, 0.0, 0.0],
        model_name=search_data["model_name"],
        document_id=search_data["document_id"],
        top_k=1,
    )

    assert len(top_result) == 1
    assert top_result[0].chunk.id == search_data["chunks"][0].id


def test_search_no_results(search_data):
    bm25_search = BM25Search(search_data["conn"])
    vector_search = VectorSearch(search_data["conn"])

    # Unmatched keyword
    assert (
        bm25_search.search(
            query="",
            document_id=search_data["document_id"],
        )
        == []
    )

    # Unknown embedding model
    assert (
        vector_search.search(
            query_vector=[1.0, 0.0, 0.0],
            model_name="unknown-model",
            document_id=search_data["document_id"],
        )
        == []
    )


def test_search_invalid_parameters(search_data):
    bm25_search = BM25Search(search_data["conn"])
    vector_search = VectorSearch(search_data["conn"])

    with pytest.raises(ValueError):
        bm25_search.search(
            query="computer vision",
            top_k=0,
        )

    with pytest.raises(ValueError):
        vector_search.search(
            query_vector=[],
            model_name=search_data["model_name"],
        )

    with pytest.raises(ValueError):
        vector_search.search(
            query_vector=[1.0, 0.0, 0.0],
            model_name=search_data["model_name"],
            min_similarity=1.5,
        )
