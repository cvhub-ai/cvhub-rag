-- ============================================================
-- Extensions
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;


-- ============================================================
-- Documents
-- ============================================================

CREATE TABLE documents (
    id UUID PRIMARY KEY,

    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    storage_path TEXT NOT NULL,

    checksum TEXT,

    parser_name TEXT,
    parser_version TEXT,

    status VARCHAR(32) NOT NULL DEFAULT 'pending',

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_checksum
    ON documents(checksum);


-- ============================================================
-- Chunks
-- ============================================================

CREATE TABLE chunks (
    id UUID PRIMARY KEY,

    document_id UUID NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    chunk_index INTEGER NOT NULL,

    content TEXT NOT NULL,
    content_hash TEXT,

    token_count INTEGER,

    chunk_type VARCHAR(64),

    page_start INTEGER,
    page_end INTEGER,

    section_title TEXT,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_chunks_document_index
        UNIQUE (document_id, chunk_index),

    CONSTRAINT chk_chunks_page_range
        CHECK (
            page_start IS NULL
            OR page_end IS NULL
            OR page_end >= page_start
        ),

    CONSTRAINT chk_chunks_token_count
        CHECK (
            token_count IS NULL
            OR token_count >= 0
        )
);

CREATE INDEX idx_chunks_document_id
    ON chunks(document_id);


-- ============================================================
-- BM25 Index
-- ============================================================

CREATE INDEX idx_chunks_bm25
    ON chunks
    USING paradedb (
        id,
        content
    )
    WITH (
        key_field = 'id'
    );


-- ============================================================
-- Chunk Embeddings
-- ============================================================

CREATE TABLE chunk_embeddings (
    id UUID PRIMARY KEY,

    chunk_id UUID NOT NULL
        REFERENCES chunks(id)
        ON DELETE CASCADE,

    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'default',

    dimension INTEGER NOT NULL,

    embedding VECTOR NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_chunk_embedding_model
        UNIQUE (
            chunk_id,
            model_name,
            model_version
        ),

    CONSTRAINT chk_embedding_dimension
        CHECK (
            dimension > 0
            AND vector_dims(embedding) = dimension
        )
);

CREATE INDEX idx_chunk_embeddings_chunk_id
    ON chunk_embeddings(chunk_id);

CREATE INDEX idx_chunk_embeddings_model
    ON chunk_embeddings(
        model_name,
        model_version
    );


-- ============================================================
-- Images
-- ============================================================

CREATE TABLE images (
    id UUID PRIMARY KEY,

    chunk_id UUID NOT NULL
        REFERENCES chunks(id)
        ON DELETE CASCADE,

    page_number INTEGER,
    image_index INTEGER,

    storage_path TEXT NOT NULL,

    mime_type VARCHAR(64),

    width INTEGER,
    height INTEGER,

    caption TEXT,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_images_page_number
        CHECK (
            page_number IS NULL
            OR page_number > 0
        ),

    CONSTRAINT chk_images_size
        CHECK (
            (width IS NULL OR width > 0)
            AND
            (height IS NULL OR height > 0)
        )
);

CREATE INDEX idx_images_chunk_id
    ON images(chunk_id);