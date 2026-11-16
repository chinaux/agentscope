# -*- coding: utf-8 -*-
# pylint: disable=protected-access,unused-argument
"""Unit tests for the ZvecStore class.

``protected-access`` is disabled because some tests legitimately
reach into store internals to assert what the public API just did:
that a handle rejected on a metric mismatch is never left in
``_collections``, and that ``metric_type`` is normalized before it
reaches the SDK.

``unused-argument`` is disabled because the scan-retry tests install
stand-ins through ``mock.patch.object``, and a stand-in has to keep
the signature of the method it replaces.
"""
import asyncio
import datetime
import enum
import os
import tempfile
import unittest
from contextlib import AsyncExitStack
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.async_case import IsolatedAsyncioTestCase

from utils import AnyString, AnyValue

from agentscope.message import TextBlock
from agentscope.rag import (
    Chunk,
    DocumentSummary,
    KnowledgeBase,
    VectorRecord,
    VectorSearchResult,
    ZvecStore,
)


_ZVEC_AVAILABLE = find_spec("zvec") is not None


class _Level(enum.Enum):
    """Enum metadata value, to cover a type JSON does not store as-is."""

    HIGH = "high"
    LOW = "low"


def _make_record(
    text: str,
    vector: list[float],
    document_id: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
    metadata: dict | None = None,
    source: str | None = None,
) -> VectorRecord:
    """Build a VectorRecord for testing.

    Args:
        text (`str`):
            The chunk text content.
        vector (`list[float]`):
            The embedding vector.
        document_id (`str`):
            The ID of the source document the record belongs to.
        chunk_index (`int`, defaults to ``0``):
            The chunk index within the document.
        total_chunks (`int`, defaults to ``1``):
            The total number of chunks in the document.
        metadata (`dict | None`, optional):
            The chunk metadata.
        source (`str | None`, optional):
            The source filename, defaulting to ``<document_id>.txt``.

    Returns:
        `VectorRecord`:
            The constructed record.
    """
    return VectorRecord(
        vector=vector,
        document_id=document_id,
        chunk=Chunk(
            content=TextBlock(text=text),
            source=f"{document_id}.txt" if source is None else source,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            metadata=metadata or {},
        ),
    )


def _dump_results(results: list[VectorSearchResult]) -> list[dict]:
    """Convert search results into plain dicts for comparison.

    Args:
        results (`list[VectorSearchResult]`):
            The search results to convert.

    Returns:
        `list[dict]`:
            The results as plain dicts.
    """
    return [result.model_dump() for result in results]


def _dump_summaries(summaries: list[DocumentSummary]) -> list[dict]:
    """Sort document summaries by id and convert them into dicts.

    ``list_documents`` returns summaries in unspecified order, so they
    are sorted here to keep whole-structure assertions deterministic.

    Args:
        summaries (`list[DocumentSummary]`):
            The summaries to convert.

    Returns:
        `list[dict]`:
            The summaries as plain dicts, ordered by ``document_id``.
    """
    return [
        summary.model_dump()
        for summary in sorted(summaries, key=lambda item: item.document_id)
    ]


def _expected_chunk(
    text: str,
    document_id: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
    metadata: dict | None = None,
    source: str | None = None,
) -> dict:
    """Build the dumped form of a chunk.

    Args:
        text (`str`):
            The chunk text content.
        document_id (`str`):
            The ID of the source document the chunk belongs to.
        chunk_index (`int`, defaults to ``0``):
            The chunk index within the document.
        total_chunks (`int`, defaults to ``1``):
            The total number of chunks in the document.
        metadata (`dict | None`, optional):
            The chunk metadata.
        source (`str | None`, optional):
            The source filename, defaulting to ``<document_id>.txt``.

    Returns:
        `dict`:
            The expected dumped chunk, with ``AnyString`` in place of
            the generated block id and timestamp.
    """
    return {
        "content": {
            "type": "text",
            "text": text,
            "id": AnyString(),
            "created_at": AnyString(),
            "finished_at": None,
        },
        "source": f"{document_id}.txt" if source is None else source,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "metadata": metadata or {},
    }


def _expected_result(
    score: Any,
    text: str,
    document_id: str,
    chunk_index: int = 0,
    total_chunks: int = 1,
    metadata: dict | None = None,
    source: str | None = None,
) -> dict:
    """Build the dumped form of one search result.

    Args:
        score (`Any`):
            The expected relevance score, or :class:`AnyValue` when the
            score is asserted separately.
        text (`str`):
            The chunk text content.
        document_id (`str`):
            The ID of the source document the chunk belongs to.
        chunk_index (`int`, defaults to ``0``):
            The chunk index within the document.
        total_chunks (`int`, defaults to ``1``):
            The total number of chunks in the document.
        metadata (`dict | None`, optional):
            The chunk metadata.
        source (`str | None`, optional):
            The source filename, defaulting to ``<document_id>.txt``.

    Returns:
        `dict`:
            The expected dumped search result.
    """
    return {
        "score": score,
        "document_id": document_id,
        "chunk": _expected_chunk(
            text,
            document_id,
            chunk_index,
            total_chunks,
            metadata,
            source,
        ),
    }


def _knowledge_base(
    store: ZvecStore,
    metadata_filter: dict | None,
) -> KnowledgeBase:
    """Bind a handle without needing an embedding request for deletion.

    Args:
        store (`ZvecStore`):
            The store under test.
        metadata_filter (`dict | None`):
            The scope the handle is bound to.

    Returns:
        `KnowledgeBase`:
            A handle over the shared collection.
    """
    return KnowledgeBase(
        name="test",
        description="Test knowledge base",
        embedding_model=SimpleNamespace(dimensions=3),
        vector_store=store,
        collection="shared",
        metadata_filter=metadata_filter,
    )


class _ZvecStoreTestBase(IsolatedAsyncioTestCase):
    """Shared temporary-directory fixture for the ZvecStore test cases."""

    async def asyncSetUp(self) -> None:
        """Create a temporary zvec store before each test."""
        self._exit_stack = AsyncExitStack()
        tmpdir = self._exit_stack.enter_context(tempfile.TemporaryDirectory())
        self._tmpdir = Path(tmpdir)
        self.store = await self._exit_stack.enter_async_context(
            ZvecStore(path=str(self._tmpdir / "zvec")),
        )

    async def asyncTearDown(self) -> None:
        """Close the store and remove the temporary directory."""
        await self._exit_stack.aclose()


@unittest.skipUnless(_ZVEC_AVAILABLE, "zvec is required for ZvecStore tests")
class ZvecStoreTest(_ZvecStoreTestBase):
    """The collection lifecycle and basic search test cases."""

    async def test_collection_lifecycle(self) -> None:
        """Collections can be created, checked, and deleted."""
        self.assertEqual(await self.store.has_collection("kb-1"), False)

        await self.store.create_collection("kb-1", dimensions=3)
        self.assertEqual(await self.store.has_collection("kb-1"), True)
        self.assertEqual((self._tmpdir / "zvec" / "kb-1").is_dir(), True)

        # Creating an existing collection is a no-op
        await self.store.create_collection("kb-1", dimensions=3)
        self.assertEqual(await self.store.has_collection("kb-1"), True)

        await self.store.delete_collection("kb-1")
        self.assertEqual(await self.store.has_collection("kb-1"), False)
        self.assertEqual((self._tmpdir / "zvec" / "kb-1").exists(), False)

        # Deleting a missing collection is a no-op too
        await self.store.delete_collection("kb-1")
        self.assertEqual(await self.store.has_collection("kb-1"), False)

    async def test_create_collection_rejects_dimension_mismatch(self) -> None:
        """An existing collection keeps the dimension it was built with."""
        await self.store.create_collection("kb-1", dimensions=3)
        with self.assertRaisesRegex(ValueError, "dimension 3"):
            await self.store.create_collection("kb-1", dimensions=4)

        # The rejected call leaves the usable collection untouched
        await self.store.insert(
            "kb-1",
            [_make_record("still here", [1.0, 0.0, 0.0], "doc-1")],
        )
        self.assertEqual(len(await self.store.list_documents("kb-1")), 1)

    async def test_create_collection_rejects_metric_mismatch(self) -> None:
        """An existing collection keeps the metric it was built with."""
        root = str(self._tmpdir / "metric")
        async with ZvecStore(path=root, metric_type="L2") as first:
            await first.create_collection("kb-1", dimensions=3)

        # The same metric may reopen the same directory
        async with ZvecStore(path=root, metric_type="L2") as same:
            await same.create_collection("kb-1", dimensions=3)

        async with ZvecStore(path=root, metric_type="COSINE") as other:
            with self.assertRaisesRegex(ValueError, "metric L2"):
                await other.create_collection("kb-1", dimensions=3)

    async def test_metric_mismatch_is_caught_on_open(self) -> None:
        """Every entry point rejects a collection built with another metric.

        The application looks an existing knowledge base up through
        ``has_collection`` and only calls ``create_collection`` when it
        creates a new one, so a check bound to that method alone would
        never run against a directory that already exists.
        """
        root = str(self._tmpdir / "metric")
        async with ZvecStore(path=root, metric_type="L2") as first:
            await first.create_collection("kb-1", dimensions=3)
            await first.insert(
                "kb-1",
                [_make_record("doc", [1.0, 0.0, 0.0], "doc-1")],
            )

        async with ZvecStore(path=root, metric_type="COSINE") as other:
            # Nothing is cached by the rejected open, so every call
            # verifies again instead of trusting a stale handle.
            for call in (
                lambda: other.search("kb-1", [1.0, 0.0, 0.0]),
                lambda: other.insert(
                    "kb-1",
                    [_make_record("doc", [1.0, 0.0, 0.0], "doc-1")],
                ),
                lambda: other.list_documents("kb-1"),
                lambda: other.list_chunks("kb-1", "doc-1"),
                lambda: other.create_collection("kb-1", dimensions=3),
            ):
                with self.assertRaisesRegex(ValueError, "metric L2"):
                    await call()
            self.assertNotIn("kb-1", other._collections)

            # The knowledge base still counts as existing, so a
            # misconfigured one can be dropped instead of becoming
            # undeletable.
            self.assertTrue(await other.has_collection("kb-1"))
            await other.delete_collection("kb-1")
            self.assertFalse(await other.has_collection("kb-1"))

    async def test_rejects_unsafe_collection_names(self) -> None:
        """A collection name cannot escape the store directory."""
        names = [
            "../evil",
            "..",
            "a/b",
            "a\\b",
            "",
            ".",
            ".hidden",
            "-leading-dash",
            "_leading-underscore",
            "kb:1",
            "x" * 129,
        ]
        for name in names:
            with self.subTest(name=name):
                for operation in (
                    self.store.has_collection,
                    self.store.delete_collection,
                ):
                    with self.assertRaises(ValueError):
                        await operation(name)
                with self.assertRaises(ValueError):
                    await self.store.create_collection(name, dimensions=3)

        self.assertEqual(sorted(os.listdir(self._tmpdir)), [])

    async def test_insert_and_search(self) -> None:
        """Inserted records are searchable, ordered by similarity."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "Hello world!",
                    [1.0, 0.0, 0.0],
                    document_id="doc-1",
                    chunk_index=0,
                    total_chunks=2,
                ),
                _make_record(
                    "Goodbye world!",
                    [0.0, 1.0, 0.0],
                    document_id="doc-1",
                    chunk_index=1,
                    total_chunks=2,
                ),
            ],
        )

        results = await self.store.search(
            "kb-1",
            query_vector=[1.0, 0.0, 0.0],
            top_k=2,
        )

        # Cosine similarity: identical direction scores 1.0, orthogonal 0.0
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertAlmostEqual(results[1].score, 0.0)
        self.assertEqual(
            _dump_results(results),
            [
                _expected_result(
                    AnyValue(),
                    "Hello world!",
                    "doc-1",
                    chunk_index=0,
                    total_chunks=2,
                ),
                _expected_result(
                    AnyValue(),
                    "Goodbye world!",
                    "doc-1",
                    chunk_index=1,
                    total_chunks=2,
                ),
            ],
        )

    async def test_search_top_k(self) -> None:
        """top_k limits the number of returned results."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(f"chunk-{i}", [1.0, 0.0, 0.0], "doc-1", i, 5)
                for i in range(5)
            ],
        )

        for top_k in (1, 3, 5, 10):
            with self.subTest(top_k=top_k):
                results = await self.store.search(
                    "kb-1",
                    query_vector=[1.0, 0.0, 0.0],
                    top_k=top_k,
                )
                self.assertEqual(len(results), min(top_k, 5))

        with self.assertRaises(ValueError):
            await self.store.search("kb-1", [1.0, 0.0, 0.0], top_k=0)

    async def test_search_requires_an_existing_collection(self) -> None:
        """Searching an absent collection fails instead of creating it."""
        with self.assertRaises(ValueError):
            await self.store.search("missing", [1.0, 0.0, 0.0])
        self.assertEqual(await self.store.has_collection("missing"), False)


@unittest.skipUnless(_ZVEC_AVAILABLE, "zvec is required for ZvecStore tests")
class ZvecMetadataFilterTest(_ZvecStoreTestBase):
    """Verify ``metadata_filter`` handling against the real backend."""

    async def test_search_metadata_filter(self) -> None:
        """search applies the metadata_filter as a native pre-filter."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "A",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    metadata={"kb_scope": "kb-a"},
                ),
                _make_record(
                    "B",
                    [1.0, 0.0, 0.0],
                    "doc-2",
                    metadata={"kb_scope": "kb-b"},
                ),
            ],
        )

        for scope, document_id in (("kb-a", "doc-1"), ("kb-b", "doc-2")):
            with self.subTest(scope=scope):
                results = await self.store.search(
                    "kb-1",
                    query_vector=[1.0, 0.0, 0.0],
                    top_k=5,
                    metadata_filter={"kb_scope": scope},
                )
                self.assertEqual(
                    [
                        (item.document_id, item.chunk.content.text)
                        for item in results
                    ],
                    [(document_id, "A" if scope == "kb-a" else "B")],
                )

        self.assertEqual(
            await self.store.search(
                "kb-1",
                [1.0, 0.0, 0.0],
                metadata_filter={"kb_scope": "kb-c"},
            ),
            [],
        )

    async def test_mixed_scalar_metadata_filter(self) -> None:
        """Combine float equality with the other scalar match types."""
        await self.store.create_collection("kb-1", dimensions=3)
        metadata_filter = {
            "confidence": 0.75,
            "category": "notes",
            "version": 2,
            "active": True,
        }
        records = []
        for index, metadata in enumerate(
            [
                metadata_filter,
                {**metadata_filter, "confidence": 0.5},
                {**metadata_filter, "category": "other"},
                {**metadata_filter, "version": 3},
                {**metadata_filter, "active": False},
            ],
        ):
            records.append(
                _make_record(
                    f"note-{index}",
                    [1.0, 0.0, 0.0],
                    f"doc-{index}",
                    metadata=dict(metadata),
                ),
            )
        await self.store.insert("kb-1", records)

        results = await self.store.search(
            "kb-1",
            [1.0, 0.0, 0.0],
            metadata_filter=metadata_filter,
        )
        self.assertEqual(
            [(item.document_id, item.chunk.metadata) for item in results],
            [("doc-0", metadata_filter)],
        )

    async def test_metadata_filter_is_type_sensitive(self) -> None:
        """A filter value only matches metadata of the same JSON type."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "int",
                    [1.0, 0.0, 0.0],
                    "doc-int",
                    metadata={"v": 10},
                ),
                _make_record(
                    "str",
                    [1.0, 0.0, 0.0],
                    "doc-str",
                    metadata={"v": "10"},
                ),
                _make_record(
                    "float",
                    [1.0, 0.0, 0.0],
                    "doc-float",
                    metadata={"v": 10.0},
                ),
            ],
        )

        cases = [
            ({"v": 10}, "doc-int"),
            ({"v": "10"}, "doc-str"),
            ({"v": 10.0}, "doc-float"),
        ]
        for metadata_filter, document_id in cases:
            with self.subTest(metadata_filter=metadata_filter):
                results = await self.store.search(
                    "kb-1",
                    [1.0, 0.0, 0.0],
                    metadata_filter=metadata_filter,
                )
                self.assertEqual(
                    [item.document_id for item in results],
                    [document_id],
                )

    async def test_float_metadata_filters(self) -> None:
        """Every read operation matches finite floats without rounding."""
        await self.store.create_collection("kb-1", dimensions=3)
        values = [-0.75, 0.0, 0.75]
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    f"note-{index}",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    chunk_index=index,
                    total_chunks=len(values),
                    metadata={"confidence": value},
                )
                for index, value in enumerate(values)
            ],
        )

        for index, value in enumerate(values):
            metadata_filter = {"confidence": value}
            with self.subTest(operation="search", value=value):
                results = await self.store.search(
                    "kb-1",
                    [1.0, 0.0, 0.0],
                    metadata_filter=metadata_filter,
                )
                self.assertEqual(
                    [result.chunk.chunk_index for result in results],
                    [index],
                )
            with self.subTest(operation="list_documents", value=value):
                self.assertEqual(
                    _dump_summaries(
                        await self.store.list_documents(
                            "kb-1",
                            metadata_filter=metadata_filter,
                        ),
                    ),
                    [
                        {
                            "document_id": "doc-1",
                            "source": "doc-1.txt",
                            "chunk_count": 1,
                            "metadata": metadata_filter,
                        },
                    ],
                )
            with self.subTest(operation="list_chunks", value=value):
                chunks = await self.store.list_chunks(
                    "kb-1",
                    "doc-1",
                    metadata_filter=metadata_filter,
                )
                self.assertEqual(
                    [chunk.chunk_index for chunk in chunks],
                    [index],
                )

    async def test_nonfinite_metadata_filter(self) -> None:
        """Reject non-finite floats instead of widening a query."""
        await self.store.create_collection("kb-1", dimensions=3)
        for value in [float("nan"), float("inf"), float("-inf")]:
            metadata_filter = {"confidence": value}
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    await self.store.search(
                        "kb-1",
                        [1.0, 0.0, 0.0],
                        metadata_filter=metadata_filter,
                    )
                with self.assertRaisesRegex(ValueError, "finite"):
                    await self.store.delete(
                        "kb-1",
                        "doc-1",
                        metadata_filter,
                    )
                with self.assertRaisesRegex(ValueError, "finite"):
                    await self.store.list_documents(
                        "kb-1",
                        metadata_filter,
                    )
                with self.assertRaisesRegex(ValueError, "finite"):
                    await self.store.list_chunks(
                        "kb-1",
                        "doc-1",
                        metadata_filter=metadata_filter,
                    )

    async def test_nested_nonfinite_metadata_is_rejected(self) -> None:
        """The finiteness check reaches inside nested metadata values."""
        await self.store.create_collection("kb-1", dimensions=3)
        with self.assertRaisesRegex(ValueError, "finite"):
            await self.store.search(
                "kb-1",
                [1.0, 0.0, 0.0],
                metadata_filter={"range": {"min": 0.0, "max": float("inf")}},
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            await self.store.insert(
                "kb-1",
                [
                    _make_record(
                        "nested",
                        [1.0, 0.0, 0.0],
                        "doc-1",
                        metadata={"scores": [0.5, float("nan")]},
                    ),
                ],
            )

    async def test_unordered_metadata_values_are_rejected(self) -> None:
        """A set hashes differently in every process, so it is refused."""
        await self.store.create_collection("kb-1", dimensions=3)
        for value in ({"a", "b"}, frozenset({"a", "b"})):
            with self.subTest(container=type(value).__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "not stable across processes",
                ):
                    await self.store.search(
                        "kb-1",
                        [1.0, 0.0, 0.0],
                        metadata_filter={"tags": value},
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "not stable across processes",
                ):
                    await self.store.insert(
                        "kb-1",
                        [
                            _make_record(
                                "tagged",
                                [1.0, 0.0, 0.0],
                                "doc-1",
                                metadata={"tags": value},
                            ),
                        ],
                    )

    async def test_nested_metadata_filter_round_trip(self) -> None:
        """Canonical encoding keeps nested values matchable."""
        await self.store.create_collection("kb-1", dimensions=3)
        metadata = {"range": {"min": 1, "max": 5}, "tags": ["a", "b"]}
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "match",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    metadata=metadata,
                ),
            ],
        )
        results = await self.store.search(
            "kb-1",
            [1.0, 0.0, 0.0],
            metadata_filter={"range": {"max": 5, "min": 1}},
        )
        self.assertEqual(
            [result.chunk.content.text for result in results],
            ["match"],
        )

    async def test_list_chunks_filters_on_the_stored_digests(self) -> None:
        """A chunk page filters on the same digests ``search`` uses.

        ``datetime`` and ``Enum`` do not survive the stored ``Chunk``
        JSON unchanged, so a filter recomputed from the decoded chunk
        would drop records the native query still finds.
        """
        await self.store.create_collection("kb-1", dimensions=3)
        moment = datetime.datetime(2024, 1, 1, 0, 0)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "keep",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    chunk_index=0,
                    metadata={"at": moment, "level": _Level.HIGH},
                ),
                _make_record(
                    "drop",
                    [0.0, 1.0, 0.0],
                    "doc-1",
                    chunk_index=1,
                    metadata={
                        "at": datetime.datetime(2025, 1, 1, 0, 0),
                        "level": _Level.LOW,
                    },
                ),
            ],
        )

        for metadata_filter in ({"at": moment}, {"level": _Level.HIGH}):
            with self.subTest(metadata_filter=metadata_filter):
                results = await self.store.search(
                    "kb-1",
                    [1.0, 0.0, 0.0],
                    metadata_filter=metadata_filter,
                )
                self.assertEqual(
                    [result.chunk.content.text for result in results],
                    ["keep"],
                )
                chunks = await self.store.list_chunks(
                    "kb-1",
                    "doc-1",
                    metadata_filter=metadata_filter,
                )
                self.assertEqual(
                    [chunk.content.text for chunk in chunks],
                    ["keep"],
                )

    async def test_metadata_values_cannot_alter_the_filter(self) -> None:
        """Hostile document ids and metadata never widen a query.

        Filter values never reach the native expression as text — they
        are hashed first — so quoting, backslashes, and expression
        syntax inside a value cannot change what is matched.
        """
        await self.store.create_collection("kb-1", dimensions=3)
        hostile = [
            "x' or '1'='1",
            "x') or ('1'='1",
            "back\\slash",
            "trailing\\",
            'double"quote',
            "filter_key contain_all ()",
        ]
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    f"hostile-{index}",
                    [1.0, 0.0, 0.0],
                    document_id=value,
                    metadata={"scope": value},
                )
                for index, value in enumerate(hostile)
            ],
        )
        await self.store.insert(
            "kb-1",
            [_make_record("innocent", [1.0, 0.0, 0.0], "doc-safe")],
        )

        for index, value in enumerate(hostile):
            with self.subTest(value=value):
                results = await self.store.search(
                    "kb-1",
                    [1.0, 0.0, 0.0],
                    top_k=10,
                    metadata_filter={"scope": value},
                )
                self.assertEqual(
                    [item.chunk.content.text for item in results],
                    [f"hostile-{index}"],
                )
                self.assertEqual(
                    len(
                        await self.store.list_documents(
                            "kb-1",
                            {"scope": value},
                        ),
                    ),
                    1,
                )

        # Deleting one hostile document leaves every other record alone
        await self.store.delete("kb-1", hostile[0])
        self.assertEqual(
            len(await self.store.list_chunks("kb-1", hostile[0])),
            0,
        )
        self.assertEqual(
            sorted(
                summary.document_id
                for summary in await self.store.list_documents("kb-1")
            ),
            sorted([*hostile[1:], "doc-safe"]),
        )

    async def test_unicode_round_trip(self) -> None:
        """Non-ASCII text and ids survive insertion and retrieval."""
        await self.store.create_collection("kb-1", dimensions=3)
        text = "中文内容 😀 with \"quotes\", 'apostrophes' and \\ backslashes"
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    text,
                    [1.0, 0.0, 0.0],
                    document_id="doc-中文-😀",
                    source="报告 '2026'.md",
                    metadata={"tenant": "acme 😀", "path": "C:\\tmp\\a"},
                ),
            ],
        )

        results = await self.store.search(
            "kb-1",
            [1.0, 0.0, 0.0],
            metadata_filter={"tenant": "acme 😀", "path": "C:\\tmp\\a"},
        )
        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertEqual(
            _dump_results(results),
            [
                _expected_result(
                    AnyValue(),
                    text,
                    "doc-中文-😀",
                    source="报告 '2026'.md",
                    metadata={"tenant": "acme 😀", "path": "C:\\tmp\\a"},
                ),
            ],
        )
        self.assertEqual(
            _dump_summaries(await self.store.list_documents("kb-1")),
            [
                {
                    "document_id": "doc-中文-😀",
                    "source": "报告 '2026'.md",
                    "chunk_count": 1,
                    "metadata": {"tenant": "acme 😀", "path": "C:\\tmp\\a"},
                },
            ],
        )


@unittest.skipUnless(_ZVEC_AVAILABLE, "zvec is required for ZvecStore tests")
class ZvecStoreDataTest(_ZvecStoreTestBase):
    """Verify deletion, listing, persistence, and concurrency."""

    async def test_delete_by_document_id(self) -> None:
        """Deleting a document removes all of its chunks only."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record("doc1-a", [1.0, 0.0, 0.0], "doc-1", 0, 2),
                _make_record("doc1-b", [1.0, 0.0, 0.0], "doc-1", 1, 2),
                _make_record("doc2-a", [0.0, 1.0, 0.0], "doc-2", 0, 1),
            ],
        )

        await self.store.delete("kb-1", "doc-1")

        self.assertEqual(
            _dump_summaries(await self.store.list_documents("kb-1")),
            [
                {
                    "document_id": "doc-2",
                    "source": "doc-2.txt",
                    "chunk_count": 1,
                    "metadata": {},
                },
            ],
        )
        self.assertEqual(await self.store.list_chunks("kb-1", "doc-1"), [])

        # Deleting an unknown document is a no-op
        await self.store.delete("kb-1", "doc-missing")
        self.assertEqual(len(await self.store.list_documents("kb-1")), 1)

    async def test_delete_with_metadata_filter(self) -> None:
        """A scoped delete only removes chunks inside the scope."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "in-scope",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    0,
                    2,
                    metadata={"tenant": "alpha"},
                ),
                _make_record(
                    "out-of-scope",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    1,
                    2,
                    metadata={"tenant": "beta"},
                ),
            ],
        )

        await self.store.delete("kb-1", "doc-1", {"tenant": "alpha"})

        self.assertEqual(
            [
                chunk.chunk_index
                for chunk in await self.store.list_chunks("kb-1", "doc-1")
            ],
            [1],
        )

    async def test_insert_empty_records(self) -> None:
        """Inserting an empty list is a no-op."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert("kb-1", [])
        self.assertEqual(await self.store.list_documents("kb-1"), [])

    async def test_list_documents_aggregates_by_document_id(self) -> None:
        """list_documents groups chunks by document_id."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "A",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    0,
                    2,
                    metadata={
                        "filename": "alpha.txt",
                        "media_type": "text/plain",
                    },
                    source="alpha.txt",
                ),
                _make_record(
                    "B",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    1,
                    2,
                    metadata={
                        "filename": "alpha.txt",
                        "media_type": "text/plain",
                    },
                    source="alpha.txt",
                ),
                _make_record(
                    "C",
                    [0.0, 1.0, 0.0],
                    "doc-2",
                    0,
                    1,
                    metadata={
                        "filename": "beta.md",
                        "media_type": "text/markdown",
                    },
                    source="beta.md",
                ),
            ],
        )

        self.assertEqual(
            _dump_summaries(await self.store.list_documents("kb-1")),
            [
                {
                    "document_id": "doc-1",
                    "source": "alpha.txt",
                    "chunk_count": 2,
                    "metadata": {
                        "filename": "alpha.txt",
                        "media_type": "text/plain",
                    },
                },
                {
                    "document_id": "doc-2",
                    "source": "beta.md",
                    "chunk_count": 1,
                    "metadata": {
                        "filename": "beta.md",
                        "media_type": "text/markdown",
                    },
                },
            ],
        )

        self.assertEqual(
            _dump_summaries(
                await self.store.list_documents(
                    "kb-1",
                    metadata_filter={"media_type": "text/markdown"},
                ),
            ),
            [
                {
                    "document_id": "doc-2",
                    "source": "beta.md",
                    "chunk_count": 1,
                    "metadata": {
                        "filename": "beta.md",
                        "media_type": "text/markdown",
                    },
                },
            ],
        )

    async def test_list_chunks_pagination_and_isolation(self) -> None:
        """list_chunks pages by chunk_index and isolates documents."""
        await self.store.create_collection("kb-1", dimensions=3)
        records = [
            _make_record(
                f"doc1-chunk{i}",
                [1.0, 0.0, 0.0],
                document_id="doc-1",
                chunk_index=i,
                total_chunks=5,
            )
            # Insert out of order to prove ordering is restored.
            for i in (3, 0, 4, 1, 2)
        ] + [
            _make_record(
                f"doc2-chunk{i}",
                [0.0, 1.0, 0.0],
                document_id="doc-2",
                chunk_index=i,
                total_chunks=2,
            )
            for i in (1, 0)
        ]
        await self.store.insert("kb-1", records)

        page = await self.store.list_chunks("kb-1", "doc-1", offset=0, limit=3)
        self.assertEqual([chunk.chunk_index for chunk in page], [0, 1, 2])
        self.assertEqual(
            [chunk.content.text for chunk in page],
            ["doc1-chunk0", "doc1-chunk1", "doc1-chunk2"],
        )

        tail = await self.store.list_chunks(
            "kb-1",
            "doc-1",
            offset=3,
            limit=30,
        )
        self.assertEqual([chunk.chunk_index for chunk in tail], [3, 4])

        self.assertEqual(
            await self.store.list_chunks("kb-1", "doc-1", offset=5, limit=3),
            [],
        )
        self.assertEqual(
            await self.store.list_chunks("kb-1", "doc-1", limit=0),
            [],
        )
        self.assertEqual(
            await self.store.list_chunks("kb-1", "doc-missing"),
            [],
        )

        other = await self.store.list_chunks("kb-1", "doc-2", limit=30)
        self.assertEqual([chunk.chunk_index for chunk in other], [0, 1])
        self.assertTrue(
            all(chunk.content.text.startswith("doc2") for chunk in other),
        )

    async def test_list_chunks_metadata_filter(self) -> None:
        """list_chunks applies the metadata_filter tenant scoping."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    "scoped",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    metadata={"kb_scope": "kb-a"},
                ),
            ],
        )

        self.assertEqual(
            len(
                await self.store.list_chunks(
                    "kb-1",
                    "doc-1",
                    metadata_filter={"kb_scope": "kb-a"},
                ),
            ),
            1,
        )
        self.assertEqual(
            await self.store.list_chunks(
                "kb-1",
                "doc-1",
                metadata_filter={"kb_scope": "kb-b"},
            ),
            [],
        )

    async def test_list_chunks_deduplicates_reindex_duplicates(self) -> None:
        """Re-indexing the same document never duplicates chunks.

        Stable record ids make the second insert an in-place upsert, so
        a page is never ``[0, 1, 1, 2]`` after a retry.
        """
        await self.store.create_collection("kb-1", dimensions=3)
        records = [
            _make_record(
                f"chunk{i}",
                [1.0, 0.0, 0.0],
                document_id="doc-1",
                chunk_index=i,
                total_chunks=5,
            )
            for i in range(5)
        ]
        # Simulate the index worker crashing after the vector insert
        # and retrying the whole pipeline.
        await self.store.insert("kb-1", records)
        await self.store.insert("kb-1", records)

        self.assertEqual(
            [
                chunk.chunk_index
                for chunk in await self.store.list_chunks(
                    "kb-1",
                    "doc-1",
                    offset=0,
                    limit=5,
                )
            ],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [
                chunk.chunk_index
                for chunk in await self.store.list_chunks(
                    "kb-1",
                    "doc-1",
                    offset=1,
                    limit=3,
                )
            ],
            [1, 2, 3],
        )
        self.assertEqual(
            _dump_summaries(await self.store.list_documents("kb-1")),
            [
                {
                    "document_id": "doc-1",
                    "source": "doc-1.txt",
                    "chunk_count": 5,
                    "metadata": {},
                },
            ],
        )

    async def test_list_chunks_after_reindexing_into_fewer_chunks(
        self,
    ) -> None:
        """A shorter re-index leaves no tail from the previous run."""
        await self.store.create_collection("kb-1", dimensions=3)

        def _records(count: int) -> list:
            return [
                _make_record(
                    f"v{count}-chunk{i}",
                    [1.0, 0.0, 0.0],
                    document_id="doc-1",
                    chunk_index=i,
                    total_chunks=count,
                )
                for i in range(count)
            ]

        await self.store.insert("kb-1", _records(5))
        await self.store.delete("kb-1", "doc-1")
        await self.store.insert("kb-1", _records(2))

        chunks = await self.store.list_chunks("kb-1", "doc-1", limit=30)
        self.assertEqual([chunk.chunk_index for chunk in chunks], [0, 1])
        self.assertEqual(
            [chunk.content.text for chunk in chunks],
            ["v2-chunk0", "v2-chunk1"],
        )

    async def test_persists_records_after_reopen(self) -> None:
        """Records written by one store instance are readable by the next."""
        root = str(self._tmpdir / "persistent")

        async with ZvecStore(path=root) as first_store:
            await first_store.create_collection("kb-1", dimensions=3)
            await first_store.insert(
                "kb-1",
                [
                    _make_record(
                        "persisted",
                        [1.0, 0.0, 0.0],
                        "doc-1",
                        metadata={"tenant": "alpha"},
                    ),
                ],
            )

        # The new instance reopens the same directories: leaving the
        # context above closed every collection handle.
        async with ZvecStore(path=root) as second_store:
            self.assertEqual(await second_store.has_collection("kb-1"), True)
            results = await second_store.search(
                "kb-1",
                [1.0, 0.0, 0.0],
                metadata_filter={"tenant": "alpha"},
            )

        self.assertAlmostEqual(results[0].score, 1.0)
        self.assertEqual(
            _dump_results(results),
            [
                _expected_result(
                    AnyValue(),
                    "persisted",
                    "doc-1",
                    metadata={"tenant": "alpha"},
                ),
            ],
        )

    async def test_batch_size_writes_every_record(self) -> None:
        """Inserts and chunk pages are split across native calls."""
        store = await self._exit_stack.enter_async_context(
            ZvecStore(path=str(self._tmpdir / "batched"), batch_size=2),
        )
        await store.create_collection("kb-1", dimensions=3)
        await store.insert(
            "kb-1",
            [
                _make_record(f"chunk-{i}", [1.0, 0.0, 0.0], "doc-1", i, 5)
                for i in range(5)
            ],
        )

        self.assertEqual(
            [
                chunk.chunk_index
                for chunk in await store.list_chunks("kb-1", "doc-1", limit=5)
            ],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            _dump_summaries(await store.list_documents("kb-1")),
            [
                {
                    "document_id": "doc-1",
                    "source": "doc-1.txt",
                    "chunk_count": 5,
                    "metadata": {},
                },
            ],
        )

        for batch_size in (0, -1):
            with self.subTest(batch_size=batch_size):
                with self.assertRaisesRegex(ValueError, "positive"):
                    ZvecStore(batch_size=batch_size)

    async def test_concurrent_operations_share_one_handle(self) -> None:
        """Concurrent coroutines interleave on one collection handle."""
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [
                _make_record(
                    f"chunk-{i}",
                    [1.0, 0.0, 0.0],
                    "doc-1",
                    i,
                    4,
                    metadata={"phase": "initial"},
                )
                for i in range(4)
            ],
        )

        # Every coroutine sees a deterministic slice: the searches are
        # scoped to the records written above, and the chunk listing is
        # scoped to a document the concurrent inserts never touch.
        searches = [
            self.store.search(
                "kb-1",
                [1.0, 0.0, 0.0],
                top_k=10,
                metadata_filter={"phase": "initial"},
            )
            for _ in range(4)
        ]
        inserts = [
            self.store.insert(
                "kb-1",
                [
                    _make_record(
                        f"extra-{i}",
                        [0.0, 1.0, 0.0],
                        f"doc-extra-{i}",
                        metadata={"phase": "extra"},
                    ),
                ],
            )
            for i in range(4)
        ]
        listings = [
            self.store.list_chunks("kb-1", "doc-1", limit=4) for _ in range(2)
        ]
        outcomes = await asyncio.gather(*searches, *inserts, *listings)

        for results in outcomes[:4]:
            self.assertEqual(len(results), 4)
        for chunks in outcomes[-2:]:
            self.assertEqual(
                [chunk.chunk_index for chunk in chunks],
                [0, 1, 2, 3],
            )

        summaries = await self.store.list_documents("kb-1")
        self.assertEqual(
            sorted(summary.document_id for summary in summaries),
            ["doc-1", *(f"doc-extra-{i}" for i in range(4))],
        )
        self.assertEqual(
            sum(summary.chunk_count for summary in summaries),
            8,
        )

    async def test_second_writer_is_rejected(self) -> None:
        """A live store holds the lock, so a second one cannot write.

        The native engine locks a collection directory per handle
        rather than per process, so the second store is refused even
        though both live in one process.  Nothing is raised when it is
        constructed or entered — the lock is only taken when a
        collection handle is opened — so every entry point has to
        surface the refusal itself instead of one shared open call.
        """
        root = str(self._tmpdir / "single-writer")
        async with ZvecStore(path=root) as first:
            await first.create_collection("kb-1", dimensions=3)
            await first.insert(
                "kb-1",
                [_make_record("doc", [1.0, 0.0, 0.0], "doc-1")],
            )

            async with ZvecStore(path=root) as second:
                for call in (
                    lambda: second.has_collection("kb-1"),
                    lambda: second.search("kb-1", [1.0, 0.0, 0.0]),
                    lambda: second.insert(
                        "kb-1",
                        [_make_record("doc", [0.0, 1.0, 0.0], "doc-2")],
                    ),
                    lambda: second.delete("kb-1", document_id="doc-1"),
                    lambda: second.list_documents("kb-1"),
                    lambda: second.list_chunks("kb-1", "doc-1"),
                    lambda: second.create_collection("kb-1", dimensions=3),
                    lambda: second.delete_collection("kb-1"),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "lock read-write collection",
                    ):
                        await call()

                # A refused open caches nothing, so the store keeps
                # verifying instead of trusting a half-open handle.
                self.assertEqual(second._collections, {})

            # The refused writer left the live one usable and its data
            # intact.
            self.assertEqual(len(await first.list_documents("kb-1")), 1)
            results = await first.search("kb-1", [1.0, 0.0, 0.0], top_k=5)
            self.assertAlmostEqual(results[0].score, 1.0)
            self.assertEqual(
                _dump_results(results),
                [_expected_result(AnyValue(), "doc", "doc-1")],
            )

        # Once the writer is gone the same path accepts a new store.
        async with ZvecStore(path=root) as third:
            self.assertEqual(await third.has_collection("kb-1"), True)
            self.assertEqual(len(await third.list_documents("kb-1")), 1)


@unittest.skipUnless(_ZVEC_AVAILABLE, "zvec is required for ZvecStore tests")
class ZvecScanRetryTest(IsolatedAsyncioTestCase):
    """Verify a scan survives the repair window an upsert opens.

    The native engine refuses to create a payload iterator while an
    upsert-triggered optimize runs, and asks the caller to retry.  The
    SDK exposes no error code for that, so these cases pin the retry by
    feeding the real message instead of waiting for native timing.
    """

    async def asyncSetUp(self) -> None:
        """Create one collection holding a single document."""
        self._exit_stack = AsyncExitStack()
        tmpdir = self._exit_stack.enter_context(tempfile.TemporaryDirectory())
        self.store = await self._exit_stack.enter_async_context(
            ZvecStore(path=str(Path(tmpdir) / "zvec")),
        )
        await self.store.create_collection("kb-1", dimensions=3)
        await self.store.insert(
            "kb-1",
            [_make_record("only", [1.0, 0.0, 0.0], "doc-1")],
        )

    async def asyncTearDown(self) -> None:
        """Close the store created by a test."""
        await self._exit_stack.aclose()

    async def test_transient_iterator_error_is_retried(self) -> None:
        """A refused iterator is retried until the scan goes through."""
        real_scan = self.store._scan_documents
        calls: list[int] = []

        def flaky(
            handle: Any,
            keys: set[str],
        ) -> list[DocumentSummary]:
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError(
                    "create_iterator is not allowed while a maintenance "
                    "operation (optimize, schema DDL, close or destroy) "
                    "is running; retry later",
                )
            return real_scan(handle, keys)

        with mock.patch.object(self.store, "_scan_documents", flaky):
            summaries = await self.store.list_documents("kb-1")

        self.assertEqual(
            [summary.document_id for summary in summaries],
            ["doc-1"],
        )
        self.assertEqual(len(calls), 3)

    async def test_retry_gives_up_after_the_budget(self) -> None:
        """A window that never closes surfaces the native error."""
        calls: list[int] = []

        def always_busy(
            handle: Any,
            keys: set[str],
        ) -> list[DocumentSummary]:
            calls.append(1)
            raise RuntimeError(
                "create_iterator is not allowed while a maintenance "
                "operation is running; retry later",
            )

        with mock.patch.object(self.store, "_scan_documents", always_busy):
            with self.assertRaisesRegex(
                RuntimeError,
                "create_iterator is not allowed",
            ):
                await self.store.list_documents("kb-1")

        self.assertEqual(len(calls), self.store._SCAN_RETRY_ATTEMPTS)

    async def test_other_runtime_errors_are_not_retried(self) -> None:
        """Only the maintenance window is worth another attempt."""
        calls: list[int] = []

        def broken(
            handle: Any,
            keys: set[str],
        ) -> list[DocumentSummary]:
            calls.append(1)
            raise RuntimeError("segment file is corrupt")

        with mock.patch.object(self.store, "_scan_documents", broken):
            with self.assertRaisesRegex(RuntimeError, "corrupt"):
                await self.store.list_documents("kb-1")

        self.assertEqual(len(calls), 1)


@unittest.skipUnless(_ZVEC_AVAILABLE, "zvec is required for ZvecStore tests")
class ZvecConfigValidationTest(IsolatedAsyncioTestCase):
    """Verify constructor and query-parameter validation."""

    async def asyncSetUp(self) -> None:
        """Create a temporary directory for the configuration cases."""
        self._exit_stack = AsyncExitStack()
        tmpdir = self._exit_stack.enter_context(tempfile.TemporaryDirectory())
        self._tmpdir = Path(tmpdir)

    async def asyncTearDown(self) -> None:
        """Close every store created by a test."""
        await self._exit_stack.aclose()

    async def test_metric_type_is_case_insensitive(self) -> None:
        """A lower-case metric is normalized before it reaches the SDK."""
        root = str(self._tmpdir / "zvec")
        store = ZvecStore(path=root, metric_type="cosine")
        self.assertEqual(store._metric_type, "COSINE")
        async with store:
            await store.create_collection("kb-1", dimensions=3)

        # The upper-cased value is what the reopened collection is
        # compared against, so re-opening must not report a mismatch.
        async with ZvecStore(path=root, metric_type="CoSiNe") as reopened:
            await reopened.create_collection("kb-1", dimensions=3)

    async def test_rejects_unknown_metric_type(self) -> None:
        """An unsupported metric fails at construction, not at open."""
        with self.assertRaisesRegex(ValueError, "Unsupported metric_type"):
            ZvecStore(path=str(self._tmpdir / "zvec"), metric_type="HAMMING")

    async def test_search_rejects_non_positive_top_k(self) -> None:
        """``top_k`` is validated here instead of by the native call."""
        store = await self._exit_stack.enter_async_context(
            ZvecStore(path=str(self._tmpdir / "zvec")),
        )
        for top_k in (0, -1):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(
                    ValueError,
                    "top_k must be positive",
                ):
                    # The collection does not exist: the parameter
                    # check runs before any native call.
                    await store.search(
                        "missing",
                        [1.0, 0.0, 0.0],
                        top_k=top_k,
                    )


@unittest.skipUnless(_ZVEC_AVAILABLE, "zvec is required for ZvecStore tests")
class ZvecMetricScoreTest(IsolatedAsyncioTestCase):
    """Verify score normalization for every supported metric."""

    async def asyncSetUp(self) -> None:
        """Create a temporary directory for the metric collections."""
        self._exit_stack = AsyncExitStack()
        tmpdir = self._exit_stack.enter_context(tempfile.TemporaryDirectory())
        self._tmpdir = Path(tmpdir)

    async def asyncTearDown(self) -> None:
        """Close every store created by a test."""
        await self._exit_stack.aclose()

    async def _store(self, metric_type: str) -> ZvecStore:
        """Open one store per metric inside the shared temporary directory.

        Args:
            metric_type (`str`):
                The zvec metric the store is built with.

        Returns:
            `ZvecStore`:
                The entered store, closed by ``asyncTearDown``.
        """
        return await self._exit_stack.enter_async_context(
            ZvecStore(
                path=str(self._tmpdir / metric_type.lower()),
                metric_type=metric_type,
            ),
        )

    async def test_cosine_scores_are_cosine_similarity(self) -> None:
        """The cosine distance is mapped back to a similarity."""
        store = await self._store("COSINE")
        await store.create_collection("kb-1", dimensions=3)
        await store.insert(
            "kb-1",
            [
                _make_record("same", [1.0, 0.0, 0.0], "doc-same"),
                _make_record("aside", [0.0, 1.0, 0.0], "doc-aside"),
                _make_record("opposite", [-1.0, 0.0, 0.0], "doc-opposite"),
            ],
        )

        results = await store.search("kb-1", [1.0, 0.0, 0.0], top_k=3)

        self.assertEqual(
            [
                (result.document_id, round(result.score, 6))
                for result in results
            ],
            [("doc-same", 1.0), ("doc-aside", 0.0), ("doc-opposite", -1.0)],
        )

    async def test_ip_scores_are_returned_unchanged(self) -> None:
        """The inner product is already higher-is-better."""
        store = await self._store("IP")
        await store.create_collection("kb-1", dimensions=3)
        await store.insert(
            "kb-1",
            [
                _make_record("double", [2.0, 0.0, 0.0], "doc-double"),
                _make_record("single", [1.0, 0.0, 0.0], "doc-single"),
                _make_record("aside", [0.0, 1.0, 0.0], "doc-aside"),
            ],
        )

        results = await store.search("kb-1", [1.0, 0.0, 0.0], top_k=3)

        # A distance-style normalization would have reversed this order
        self.assertEqual(
            [
                (result.document_id, round(result.score, 6))
                for result in results
            ],
            [("doc-double", 2.0), ("doc-single", 1.0), ("doc-aside", 0.0)],
        )

    async def test_l2_scores_are_negated(self) -> None:
        """The squared L2 distance is negated to rank nearer first."""
        store = await self._store("L2")
        await store.create_collection("kb-1", dimensions=3)
        await store.insert(
            "kb-1",
            [
                _make_record("near", [1.0, 0.0, 0.0], "doc-near"),
                _make_record("far", [10.0, 0.0, 0.0], "doc-far"),
            ],
        )

        results = await store.search("kb-1", [0.0, 0.0, 0.0], top_k=2)

        self.assertEqual(
            [
                (result.document_id, round(result.score, 6))
                for result in results
            ],
            [("doc-near", -1.0), ("doc-far", -100.0)],
        )


@unittest.skipUnless(_ZVEC_AVAILABLE, "zvec is required for ZvecStore tests")
class ZvecScopedDeleteTest(IsolatedAsyncioTestCase):
    """Verify scope isolation against the real embedded backend."""

    async def asyncSetUp(self) -> None:
        """Insert chunks spanning documents and two independent scopes."""
        self._exit_stack = AsyncExitStack()
        tmpdir = self._exit_stack.enter_context(tempfile.TemporaryDirectory())
        self.store = await self._exit_stack.enter_async_context(
            ZvecStore(path=str(Path(tmpdir) / "zvec")),
        )
        await self.store.create_collection("shared", dimensions=3)
        self.scope = {"tenant": "alpha", "project": 2}
        records = []
        for document_id, index, metadata in (
            ("shared-doc", 0, self.scope),
            ("shared-doc", 1, {"tenant": "beta", "project": 2}),
            ("shared-doc", 2, {"tenant": "alpha", "project": 3}),
            ("alpha-doc", 0, self.scope),
            ("beta-doc", 0, {"tenant": "beta", "project": 2}),
        ):
            records.append(
                _make_record(
                    f"{document_id}-{index}",
                    [1.0, 0.0, 0.0],
                    document_id,
                    chunk_index=index,
                    total_chunks=3,
                    metadata=dict(metadata),
                ),
            )
        await self.store.insert("shared", records)

    async def asyncTearDown(self) -> None:
        """Release the temporary collection."""
        await self._exit_stack.aclose()

    async def test_foreign_document_is_unchanged(self) -> None:
        """A scoped handle cannot remove an invisible foreign document."""
        before = await self.store.list_chunks("shared", "beta-doc")
        await _knowledge_base(self.store, self.scope).delete_document(
            "beta-doc",
        )
        after = await self.store.list_chunks("shared", "beta-doc")
        self.assertEqual(before, after)
        self.assertEqual(len(after), 1)

    async def test_delete_requires_document_and_every_scope_field(
        self,
    ) -> None:
        """Delete only chunks matching both the document and all scope keys."""
        await _knowledge_base(self.store, self.scope).delete_document(
            "shared-doc",
        )
        chunks = await self.store.list_chunks("shared", "shared-doc")
        self.assertEqual([chunk.chunk_index for chunk in chunks], [1, 2])
        self.assertEqual(
            len(await self.store.list_chunks("shared", "alpha-doc")),
            1,
        )
        await _knowledge_base(self.store, self.scope).delete_document(
            "alpha-doc",
        )
        self.assertEqual(
            await self.store.list_chunks("shared", "alpha-doc"),
            [],
        )

    async def test_unscoped_handle_deletes_all_document_chunks(self) -> None:
        """The default handle deletes documents across the whole collection."""
        await _knowledge_base(self.store, None).delete_document("shared-doc")
        self.assertEqual(
            await self.store.list_chunks("shared", "shared-doc"),
            [],
        )
        self.assertEqual(len(await self.store.list_documents("shared")), 2)
