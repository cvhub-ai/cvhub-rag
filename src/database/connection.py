import os

import psycopg
from pgvector.psycopg import register_vector
from psycopg import Connection


class ConnectionFactory:
    def __init__(self) -> None:
        self._config = {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5433")),
            "dbname": os.environ["POSTGRES_DB"],
            "user": os.environ["POSTGRES_USER"],
            "password": os.environ["POSTGRES_PASSWORD"],
        }

    def create(self) -> Connection:
        conn = psycopg.connect(
            **self._config,
            connect_timeout=5,
        )

        try:
            register_vector(conn)
        except Exception:
            conn.close()
            raise

        return conn
