# CvHub Database Setup

This guide contains only the commands required to reproduce and run the local CvHub database.

## 1. Create local environment file

```bash
cp .env.example .env
```

Edit `.env` and set the local database password.

Example:

```env
POSTGRES_DB=cvhub-rag
POSTGRES_USER=cvhub
POSTGRES_PASSWORD=your_password
```

## 2. Start the database

```bash
docker compose --env-file .env -f docker/database/docker-compose.yml up -d
```

This starts the PostgreSQL/ParadeDB container.

Check whether it is running:

```bash
docker compose --env-file .env -f docker/database/docker-compose.yml ps
```

The database should show `healthy`.

## 3. Create the database schema

```bash
docker compose --env-file .env -f docker/database/docker-compose.yml exec -T postgres sh -lc 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < migrations/001_initial_schema.sql
```

This creates the CvHub tables and indexes, including:

- `documents`
- `chunks`
- `chunk_embeddings`
- `images`
- BM25 index
- pgvector support

## 4. Open PostgreSQL manually

```bash
docker compose --env-file .env -f docker/database/docker-compose.yml exec postgres sh -lc 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

Useful checks:

```sql
\dt
```

Shows all tables.

```sql
\di
```

Shows all indexes.

Exit:

```sql
\q
```

## 5. Connect with DBeaver

Use:

```text
Host: localhost
Port: 5433
Database: cvhub-rag
Username: cvhub
Password: value from .env
```

## 6. Stop / restart the database

Stop:

```bash
docker compose --env-file .env -f docker/database/docker-compose.yml stop
```

Start again:

```bash
docker compose --env-file .env -f docker/database/docker-compose.yml start
```

Do not use `down -v` unless you intentionally want to delete the local database data.
