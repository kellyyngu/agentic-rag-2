import threading


class CitationManager:
    """
    Server-lifetime citation registry.

    Maps chunk_id → globally-unique, monotonically-increasing citation ID.
    IDs never reset across queries — the same chunk always maps to the same ID
    for the lifetime of the server process. Thread-safe for concurrent requests.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter: int = 0
        self._chunk_to_id: dict[str, str] = {}

    def get_or_assign(self, chunk_id: str) -> str:
        """Return existing global ID or assign a new sequential one."""
        with self._lock:
            if chunk_id not in self._chunk_to_id:
                self._counter += 1
                self._chunk_to_id[chunk_id] = str(self._counter)
            return self._chunk_to_id[chunk_id]

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._chunk_to_id)
