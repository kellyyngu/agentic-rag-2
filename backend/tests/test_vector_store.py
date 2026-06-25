"""
Unit tests for deterministic Qdrant point IDs.

The bug this protects against: point IDs were derived from Python's built-in
hash(), which is randomized per process. The same chunk got a different ID after
every restart, so re-ingestion created duplicate vectors instead of upserting.
_point_id must be deterministic, restart-stable, and collision-resistant.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retrieval.vector_store import _point_id


class TestPointId:
    def test_same_chunk_id_same_point_id(self):
        assert _point_id("abc123") == _point_id("abc123")

    def test_different_chunk_ids_differ(self):
        assert _point_id("chunk-a") != _point_id("chunk-b")

    def test_deterministic_known_value(self):
        """A hardcoded expected value proves the ID does not depend on PYTHONHASHSEED.
        If this drifts, IDs changed — existing vectors would be orphaned."""
        import hashlib
        digest = hashlib.sha256("abc123".encode("utf-8")).hexdigest()
        expected = int(digest[:16], 16) & 0x7FFFFFFFFFFFFFFF
        assert _point_id("abc123") == expected

    def test_restart_stability_simulated(self):
        """hash() varies with PYTHONHASHSEED; SHA-256 must not. Recomputing the
        digest independently yields the same ID — i.e. a restart cannot change it."""
        import hashlib
        for cid in ["x", "a-very-long-chunk-id-0000", "deadbeefcafef00d"]:
            digest = hashlib.sha256(cid.encode("utf-8")).hexdigest()
            assert _point_id(cid) == (int(digest[:16], 16) & 0x7FFFFFFFFFFFFFFF)

    def test_within_qdrant_unsigned_range(self):
        """Qdrant point IDs are unsigned ints — must be non-negative and < 2^63."""
        for cid in ["1", "abc", "0" * 16, "f" * 16]:
            pid = _point_id(cid)
            assert 0 <= pid < 2**63

    def test_duplicate_ingestion_identical_id(self):
        """Two independent ingestions of the same chunk_id collapse to one point."""
        first_run = _point_id("doc.pdf:3:somehash")
        second_run = _point_id("doc.pdf:3:somehash")
        assert first_run == second_run

    def test_hex_chunk_id_real_shape(self):
        """chunk_id from _hash_chunk is a 16-char md5 prefix — confirm it maps cleanly."""
        pid = _point_id("a1b2c3d4e5f60718")
        assert isinstance(pid, int)
        assert pid >= 0
