"""
A small, thread-safe, bounded LRU cache.

Used to cap long-lived in-memory registries (per-session CitationManagers, the
retriever's chunk-content cache) so a long-running process cannot grow without
limit. Standard library only — safe to import anywhere, including offline tests.

Semantics intentionally mirror a dict for the access patterns the callers use:
`get(key, default)`, `cache[key] = value`, `key in cache`, `len(cache)`. On
overflow the least-recently-used entry is evicted. Reads and writes both count
as "use" (move-to-end), so a steadily-accessed key is never evicted out from
under an active session.
"""
import threading
from collections import OrderedDict
from typing import Any, Optional

_MISSING = object()


class LRUCache:
    def __init__(self, max_size: int = 1000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max = max_size
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any, default: Optional[Any] = None) -> Any:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]
            return default

    def __getitem__(self, key: Any) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)  # evict least-recently-used

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            return key in self._data

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    @property
    def max_size(self) -> int:
        return self._max
