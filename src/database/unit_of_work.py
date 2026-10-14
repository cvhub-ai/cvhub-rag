from types import TracebackType
from typing import Self

from psycopg import Connection

from src.database.connection import ConnectionFactory


class UnitOfWork:
    def __init__(self, factory: ConnectionFactory) -> None:
        self._factory = factory
        self._conn: Connection | None = None

    def __enter__(self) -> Self:
        if self._conn is not None:
            raise RuntimeError("UnitOfWork is already active.")

        self._conn = self._factory.create()
        return self

    @property
    def connection(self) -> Connection:
        if self._conn is None:
            raise RuntimeError("UnitOfWork is not active.")

        return self._conn

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._conn is not None:
            try:
                self._conn.rollback()
            finally:
                self._conn.close()
                self._conn = None
