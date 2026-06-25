"""
Unit tests for the bounded LRU cache used to cap long-lived registries
(per-session CitationManagers, the retriever chunk cache).

Protects against the unbounded-growth bug: without a cap, one entry per session /
per chunk accumulates forever in a long-running process.
"""
import sys
import os
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from agent.bounded_cache import LRUCache


class TestLRUCache:
    def test_basic_get_set(self):
        c = LRUCache(max_size=10)
        c["a"] = 1
        assert c.get("a") == 1
        assert c["a"] == 1

    def test_missing_key_default(self):
        c = LRUCache(max_size=10)
        assert c.get("nope") is None
        assert c.get("nope", "fallback") == "fallback"

    def test_missing_key_raises_on_index(self):
        c = LRUCache(max_size=10)
        with pytest.raises(KeyError):
            _ = c["absent"]

    def test_contains_and_len(self):
        c = LRUCache(max_size=10)
        c["a"] = 1
        assert "a" in c
        assert "b" not in c
        assert len(c) == 1

    def test_evicts_least_recently_used(self):
        c = LRUCache(max_size=2)
        c["a"] = 1
        c["b"] = 2
        c["c"] = 3                 # exceeds cap → "a" (LRU) evicted
        assert "a" not in c
        assert "b" in c and "c" in c
        assert len(c) == 2

    def test_get_refreshes_recency(self):
        """Reading a key marks it recently-used so it survives the next eviction."""
        c = LRUCache(max_size=2)
        c["a"] = 1
        c["b"] = 2
        assert c.get("a") == 1     # "a" now most-recently-used
        c["c"] = 3                 # "b" is now LRU → evicted, "a" survives
        assert "a" in c
        assert "b" not in c
        assert "c" in c

    def test_set_existing_updates_and_refreshes(self):
        c = LRUCache(max_size=2)
        c["a"] = 1
        c["b"] = 2
        c["a"] = 99                # update + refresh recency
        c["c"] = 3                 # "b" is LRU → evicted
        assert c["a"] == 99
        assert "b" not in c

    def test_never_exceeds_max_size_under_load(self):
        c = LRUCache(max_size=50)
        for i in range(1000):
            c[f"k{i}"] = i
        assert len(c) == 50        # bounded regardless of insert volume

    def test_rejects_invalid_max_size(self):
        with pytest.raises(ValueError):
            LRUCache(max_size=0)

    def test_thread_safe_under_concurrent_writes(self):
        c = LRUCache(max_size=100)

        def worker(base):
            for i in range(200):
                c[f"{base}-{i}"] = i

        threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(c) == 100       # cap held; no corruption / overflow
