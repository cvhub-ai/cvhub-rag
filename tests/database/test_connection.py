# tests/database/test_connection.py

import os

from src.database.connection import ConnectionFactory


def test_create_connection():
    factory = ConnectionFactory()

    with factory.create() as conn:
        assert not conn.closed

        result = conn.execute("SELECT 1").fetchone()
        assert result == (1,)

        result = conn.execute("SELECT current_database()").fetchone()
        assert result == (os.environ["POSTGRES_DB"],)

    assert conn.closed
