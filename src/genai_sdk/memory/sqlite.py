"""SQLite-backed memory backend for durable local session storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from ..types import Message


class SQLiteMemory:
    """Persist session messages in a local SQLite database."""

    def __init__(self, path: str = ".genai_memory.db"):
        """Initialize storage and create schema if missing."""
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    name TEXT,
                    tool_call_id TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    async def append(self, session_id: str, messages: list[Message]) -> None:
        conn = sqlite3.connect(self.path)
        try:
            conn.executemany(
                """
                INSERT INTO messages(session_id, role, content, name, tool_call_id, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        session_id,
                        m.role,
                        m.content,
                        m.name,
                        m.tool_call_id,
                        m.timestamp.isoformat(),
                        json.dumps(m.metadata),
                    )
                    for m in messages
                ],
            )
            conn.commit()
        finally:
            conn.close()

    async def load(self, session_id: str, limit: int = 20) -> list[Message]:
        conn = sqlite3.connect(self.path)
        try:
            rows = conn.execute(
                """
                SELECT role, content, name, tool_call_id, timestamp, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        finally:
            conn.close()

        rows = list(reversed(rows))
        return [
            Message(
                role=row[0],
                content=row[1],
                name=row[2],
                tool_call_id=row[3],
                timestamp=datetime.fromisoformat(row[4]),
                metadata=json.loads(row[5]),
            )
            for row in rows
        ]

    async def summarize_if_needed(self, session_id: str, budget: int) -> None:
        msgs = await self.load(session_id, limit=10_000)
        if len(msgs) <= budget:
            return
        summary = "\n".join(f"[{m.role}] {m.content}" for m in msgs[:-budget])
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()
        await self.append(session_id, [Message(role="system", content=f"Summary:\n{summary}")] + msgs[-budget:])
