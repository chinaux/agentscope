# -*- coding: utf-8 -*-
"""Zvec implementation of the vector store backend.

Built on the official `zvec <https://github.com/alibaba/zvec>`_ Python
SDK, an embedded vector database engine.  A collection is a
self-contained directory on local disk, opened in-process: no server to
deploy, no network hop per query, which fits library-mode RAG where the
index can just live next to the application.

The SDK is synchronous, so every call is offloaded with
:func:`asyncio.to_thread` to keep the :class:`VectorStoreBase` contract
non-blocking.  That is safe because zvec releases the GIL inside its
native calls and one collection handle serves several threads.  The one
exception is a scan: an upsert can trigger a native optimize, and while
it runs the engine refuses to create a payload iterator, so writes make
:meth:`ZvecStore.list_documents` fail transiently.  That scan retries
instead of surfacing the refusal.
"""
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import threading
from typing import TYPE_CHECKING, Any, Literal

from ._vector_store import (
    DocumentSummary,
    VectorRecord,
    VectorSearchResult,
    VectorStoreBase,
)
from .._document import Chunk

if TYPE_CHECKING:
    from zvec import Collection

logger = logging.getLogger(__name__)


# One directory per collection, so the name becomes a path segment and
# is restricted to characters that are safe on every filesystem.  The
# pattern also rules out ``.`` / ``..`` traversal, since a name must
# start with a letter or a digit.
_COLLECTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")

# The metrics the SDK knows about.  ``metric_type`` is matched against
# this set after upper-casing, so ``"cosine"`` behaves like ``"COSINE"``
# instead of reaching ``getattr(zvec.MetricType, ...)`` and failing with
# an ``AttributeError``.
_METRIC_TYPES = frozenset({"COSINE", "IP", "L2"})


class _MetricMismatch(ValueError):
    """The collection on disk was built with another metric.

    Raised when a collection is opened under a ``metric_type`` other
    than the one it was created with.  It subclasses ``ValueError`` so
    callers guarding the documented contract keep working, and it is a
    distinct type so :meth:`ZvecStore.has_collection` can report the
    collection as existing — otherwise a misconfigured knowledge base
    could not even be deleted.
    """


class ZvecStore(VectorStoreBase):
    """Vector store backend backed by the embedded `zvec`_ engine.

    Each knowledge base maps to one zvec collection, stored as its own
    directory inside ``path``.  Every record stores the owning
    ``document_id``, the serialized :class:`~agentscope.rag.Chunk`, and
    a set of SHA-256 filter keys covering the document id and each
    ``chunk.metadata`` pair, which is what :meth:`search`,
    :meth:`delete`, :meth:`list_documents`, and :meth:`list_chunks`
    match ``metadata_filter`` against.

    The native engine locks a collection directory per handle rather
    than per process, so a second store pointing at the same ``path``
    raises ``Can't lock read-write collection`` on its first collection
    operation, even inside one process.  Keep one live store per path
    per process.

    .. _zvec: https://github.com/alibaba/zvec

    .. note:: The ``zvec`` package is required. Install it with
        ``pip install agentscope[vdb-zvec]``.

    .. code-block:: python

        store = ZvecStore(path="./rag_demo_zvec")

        async with store:
            await store.create_collection("kb-1", dimensions=768)

    """

    _DOCUMENT_ID_FIELD = "document_id"
    """Scalar column holding the source document id of a record."""

    _SOURCE_FIELD = "source"
    """Scalar column holding the source filename of a chunk."""

    _CHUNK_FIELD = "chunk_json"
    """Scalar column holding the serialized :class:`Chunk`."""

    _FILTER_KEY_FIELD = "filter_key"
    """Inverted-indexed array column holding the matchable digests."""

    _VECTOR_FIELD = "embedding"
    """Name of the dense vector column."""

    _DOCUMENT_KEY_PREFIX = "d:"
    """Namespace of a filter key derived from a document id."""

    _METADATA_KEY_PREFIX = "m:"
    """Namespace of a filter key derived from a metadata pair."""

    _TRANSIENT_SCAN_MARKER = "create_iterator is not allowed"
    """Fragment of the SDK error raised while a write repairs the index.

    The native engine rejects a new payload iterator while an
    upsert-triggered optimize runs, and "retry later" is the only
    signal it gives: the message is the marker, there is no error
    code to test against.
    """

    _SCAN_RETRY_ATTEMPTS = 5
    """Attempts :meth:`list_documents` makes before giving up."""

    _SCAN_RETRY_DELAY = 0.02
    """Base delay between scan attempts, doubled on each retry.

    Five attempts sleep 20 / 40 / 80 / 160 ms, so a repair window that
    lasts a fraction of a second is ridden out, while a collection that
    stays locked still fails fast.
    """

    def __init__(
        self,
        path: str = "./agentscope_zvec",
        metric_type: Literal["COSINE", "IP", "L2"] = "COSINE",
        hnsw_m: int | None = None,
        hnsw_ef_construction: int | None = None,
        batch_size: int = 256,
        option_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the zvec vector store.

        Args:
            path (`str`, defaults to ``"./agentscope_zvec"``):
                Root directory holding one subdirectory per collection.
                Created on the first :meth:`create_collection` call.
            metric_type (`Literal["COSINE", "IP", "L2"]`, defaults to \
             ``"COSINE"``):
                Similarity metric of the vector index, matched
                case-insensitively and stored upper-cased.  Scores are
                normalized to higher-is-better before being returned by
                :meth:`search`; see that method for the mapping.  Any
                operation that opens a collection built with a different
                metric raises ``ValueError`` instead of applying the
                wrong formula to its scores.
            hnsw_m (`int | None`, optional):
                Number of bi-directional links per HNSW node.  ``None``
                keeps the zvec default.
            hnsw_ef_construction (`int | None`, optional):
                Candidate list size used while building the HNSW graph.
                ``None`` keeps the zvec default.
            batch_size (`int`, defaults to ``256``):
                Positive number of records written per native call, and
                number of ids fetched per native call by
                :meth:`list_chunks`.
            option_kwargs (`dict[str, Any] | None`, optional):
                Extra keyword arguments forwarded to
                ``zvec.CollectionOption`` when a collection is opened,
                e.g. ``{"enable_mmap": False}``.

        Raises:
            ValueError: If ``metric_type`` is not one of ``COSINE``,
                ``IP``, or ``L2``, or if ``batch_size`` is not
                positive.
        """
        normalized_metric = str(metric_type).upper()
        if normalized_metric not in _METRIC_TYPES:
            raise ValueError(
                f"Unsupported metric_type {metric_type!r}: expected one of "
                f"{', '.join(sorted(_METRIC_TYPES))}.",
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._path = path
        self._metric_type = normalized_metric
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._batch_size = batch_size
        self._option_kwargs = option_kwargs or {}
        self._collections: dict[str, "Collection"] = {}
        # Guards `_collections` and serializes opening / destroying
        # collection handles.  Re-entrant because `_create_collection`
        # re-enters through `_open_collection` while holding it.
        self._lock = threading.RLock()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the async context and close every open collection."""
        await asyncio.to_thread(self._close_collections)

    def _close_collections(self) -> None:
        """Close and forget all cached collection handles."""
        with self._lock:
            handles = list(self._collections.values())
            self._collections.clear()
        for handle in handles:
            handle.close()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def create_collection(
        self,
        name: str,
        dimensions: int,
    ) -> None:
        """Create a new zvec collection directory.

        No-op if the collection already exists at the same
        dimensionality.

        Args:
            name (`str`):
                The collection name. Typically, the knowledge base ID.
                Must start with a letter or digit, contain only
                letters, digits, ``_``, ``.``, or ``-``, and be at most
                128 characters long, because it names a directory.
            dimensions (`int`):
                The fixed vector dimensionality for this collection.
                All vectors inserted later must have this many elements.

        Raises:
            ValueError: If ``name`` is not a valid collection name, or
                if the collection already exists with a different
                dimensionality or metric.
        """
        collection_path = self._collection_path(name)
        await asyncio.to_thread(
            self._create_collection,
            name,
            collection_path,
            dimensions,
        )

    def _create_collection(
        self,
        name: str,
        collection_path: str,
        dimensions: int,
    ) -> None:
        """Create (or adopt) one collection directory.

        Args:
            name (`str`):
                The collection name.
            collection_path (`str`):
                Directory backing the collection.
            dimensions (`int`):
                The vector dimensionality the caller expects.

        Raises:
            ValueError: If an existing collection has a different
                dimensionality or metric.
        """
        import zvec

        with self._lock:
            handle = self._collections.get(name)
            if handle is None and os.path.isdir(collection_path):
                # Opens, verifies the metric, and only then caches, so a
                # rejected collection never becomes reachable through
                # the cache on a later call.
                handle = self._open_collection(name, collection_path)

            if handle is not None:
                existing = self._dimension_of(handle)
                if existing != dimensions:
                    raise ValueError(
                        f"Collection {name!r} already exists with "
                        f"dimension {existing}, but {dimensions} was "
                        "requested.",
                    )
                return

            os.makedirs(os.path.dirname(collection_path), exist_ok=True)
            self._collections[name] = zvec.create_and_open(
                path=collection_path,
                schema=self._build_schema(name, dimensions),
                option=self._collection_option(),
            )

    async def delete_collection(self, name: str) -> None:
        """Delete a collection and all its data.

        Releases the cached handle first, then destroys the directory.
        A missing collection is a no-op, so deletion is idempotent.

        Args:
            name (`str`):
                The collection name to delete.
        """
        collection_path = self._collection_path(name)
        await asyncio.to_thread(
            self._destroy_collection,
            name,
            collection_path,
        )

    def _destroy_collection(self, name: str, collection_path: str) -> None:
        """Destroy one collection directory and drop its handle.

        Args:
            name (`str`):
                The collection name to delete.
            collection_path (`str`):
                Directory backing the collection.
        """
        import zvec

        with self._lock:
            handle = self._collections.pop(name, None)
            if handle is None:
                if not os.path.isdir(collection_path):
                    return
                handle = zvec.open(
                    collection_path,
                    self._collection_option(),
                )
            handle.destroy()

    async def has_collection(self, name: str) -> bool:
        """Check whether a collection exists.

        A directory that exists but cannot be opened — corrupt data, or
        a writer in another process holding the lock — raises instead of
        reporting ``False``, so the failure is not mistaken for an
        absent collection.

        A collection built with a different ``metric_type`` reports
        ``True``: it is there and can be dropped, while every data
        operation on it keeps failing loudly on first use.

        Args:
            name (`str`):
                The collection name to check.

        Returns:
            `bool`: ``True`` if the collection exists.
        """
        if name in self._collections:
            return True
        collection_path = self._collection_path(name)
        if not await asyncio.to_thread(os.path.isdir, collection_path):
            return False
        try:
            await self._get_collection(name)
        except _MetricMismatch as exc:
            logger.warning("%s", exc)
        return True

    # ------------------------------------------------------------------
    # Data operations
    # ------------------------------------------------------------------

    async def insert(
        self,
        collection: str,
        records: list[VectorRecord],
    ) -> None:
        """Insert records into a collection.

        Each record stores the :attr:`VectorRecord.document_id`, the
        source filename, the serialized :class:`Chunk`, and the filter
        keys used by :meth:`search` and :meth:`delete`.

        Record ids are derived deterministically from
        ``(document_id, chunk_index)`` and written with ``upsert``, so
        re-indexing the same document after a mid-pipeline crash
        replaces the previous records instead of duplicating them.

        Args:
            collection (`str`):
                The target collection name.
            records (`list[VectorRecord]`):
                The records to insert, each carrying a :class:`Chunk`
                and its embedding vector.

        Raises:
            ValueError: If a vector does not match the collection
                dimensionality, or if the collection does not exist.
            RuntimeError: If zvec reports a failed write.

        Records are written batch by batch, so the call is not atomic:
        when a later batch fails, the earlier ones stay written.
        Retrying the whole call is safe, because the record ids are
        deterministic and the rewrite lands on the same rows.
        """
        if not records:
            return

        handle = await self._get_collection(collection)
        for start in range(0, len(records), self._batch_size):
            batch = records[start : start + self._batch_size]
            await asyncio.to_thread(self._upsert_batch, handle, batch)

    def _upsert_batch(
        self,
        handle: "Collection",
        batch: list[VectorRecord],
    ) -> None:
        """Write one batch of records and check every status.

        Args:
            handle (`Collection`):
                The open zvec collection.
            batch (`list[VectorRecord]`):
                The records to write.

        Raises:
            RuntimeError: If zvec reports a failed write.
        """
        import zvec

        docs = [
            zvec.Doc(
                id=self._record_id(
                    record.document_id,
                    record.chunk.chunk_index,
                ),
                vectors={self._VECTOR_FIELD: list(record.vector)},
                fields={
                    self._DOCUMENT_ID_FIELD: record.document_id,
                    self._SOURCE_FIELD: record.chunk.source,
                    self._CHUNK_FIELD: record.chunk.model_dump_json(),
                    self._FILTER_KEY_FIELD: self._filter_keys(
                        document_id=record.document_id,
                        metadata=record.chunk.metadata,
                    ),
                },
            )
            for record in batch
        ]
        for status in handle.upsert(docs):
            if not status.ok():
                raise RuntimeError(
                    f"zvec failed to write a record: {status.message()}",
                )

    async def delete(
        self,
        collection: str,
        document_id: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> None:
        """Delete all records belonging to one source document.

        Matches the document filter key written by :meth:`insert`, plus
        every key of ``metadata_filter`` when given.

        Args:
            collection (`str`):
                The target collection name.
            document_id (`str`):
                The source document ID whose records should be removed.
            metadata_filter (`dict[str, Any] | None`, optional):
                Restrict deletion to records matching every metadata
                pair in addition to the document ID. ``None`` disables
                filtering.
        """
        expression = self._build_expression(
            self._filter_keys(document_id, metadata_filter),
        )
        handle = await self._get_collection(collection)
        await asyncio.to_thread(handle.delete_by_filter, expression)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Find the most similar records to a query vector.

        The filter is pushed into the native query as a pre-filter over
        the inverted ``filter_key`` column, so ``top_k`` is honored
        within the matching records only.

        Args:
            collection (`str`):
                The collection to search.
            query_vector (`list[float]`):
                The query embedding vector.
            top_k (`int`, defaults to ``5``):
                Maximum number of results to return.
            metadata_filter (`dict[str, Any] | None`, optional):
                If provided, restrict the search to records whose
                ``chunk.metadata`` matches every ``key == value`` pair
                in this dict.  Values are matched exactly and are
                type-sensitive, so ``{"year": 2020}`` does not match a
                chunk stored with ``{"year": "2020"}`` or
                ``{"year": 2020.0}``.

        Returns:
            `list[VectorSearchResult]`:
                Results ordered by descending similarity score.  zvec
                reports a distance for ``COSINE`` (``1 - similarity``)
                and for ``L2`` (squared Euclidean distance) and a
                similarity for ``IP``, so ``COSINE`` scores are mapped
                back to cosine similarity in ``[-1, 1]``, ``L2`` scores
                are negated, and ``IP`` scores are returned unchanged.

        Raises:
            ValueError: If a metadata filter value is a non-finite
                float, if ``top_k`` is not positive, or if
                ``query_vector`` does not match the collection
                dimensionality.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        expression = self._build_expression(
            self._filter_keys(metadata=metadata_filter),
        )
        handle = await self._get_collection(collection)
        docs = await asyncio.to_thread(
            self._query,
            handle,
            query_vector,
            top_k,
            expression,
        )
        return [
            VectorSearchResult(
                score=self._normalize_score(doc.score),
                document_id=doc.field(self._DOCUMENT_ID_FIELD),
                chunk=Chunk.model_validate_json(
                    doc.field(self._CHUNK_FIELD),
                ),
            )
            for doc in docs
        ]

    def _query(
        self,
        handle: "Collection",
        query_vector: list[float],
        top_k: int,
        expression: str | None,
    ) -> list[Any]:
        """Run one native vector query.

        Args:
            handle (`Collection`):
                The open zvec collection.
            query_vector (`list[float]`):
                The query embedding vector.
            top_k (`int`):
                Maximum number of hits to return.
            expression (`str | None`):
                Native filter expression, or ``None`` for no filtering.

        Returns:
            `list[Any]`:
                The matched ``zvec.Doc`` objects, best match first.
        """
        import zvec

        return handle.query(
            queries=zvec.Query(
                field_name=self._VECTOR_FIELD,
                vector=list(query_vector),
            ),
            topk=top_k,
            filter=expression,
            output_fields=[self._DOCUMENT_ID_FIELD, self._CHUNK_FIELD],
        )

    # ------------------------------------------------------------------
    # Document listing
    # ------------------------------------------------------------------

    async def list_documents(
        self,
        collection: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[DocumentSummary]:
        """List all distinct source documents indexed in a collection.

        Streams the collection with a payload-only snapshot iterator
        (vectors excluded) and aggregates by ``document_id``.  The first
        matching chunk encountered for each document supplies the source
        filename and the document-level ``metadata``.

        An upsert can trigger a native optimize, and the engine refuses
        to create the scan iterator while one runs.  That window is
        short and the SDK asks the caller to retry, so the scan is
        retried a few times before the error is raised.

        Args:
            collection (`str`):
                The target collection name.
            metadata_filter (`dict[str, Any] | None`, optional):
                If provided, restrict aggregation to records whose
                ``chunk.metadata`` matches every ``key == value`` pair
                in this dict.

        Returns:
            `list[DocumentSummary]`:
                One summary per distinct ``document_id``.

        Raises:
            RuntimeError: If the scan keeps failing, for instance
                because the collection is corrupt.
        """
        keys = set(self._filter_keys(metadata=metadata_filter))
        handle = await self._get_collection(collection)
        attempt = 0
        while True:
            try:
                return await asyncio.to_thread(
                    self._scan_documents,
                    handle,
                    keys,
                )
            except RuntimeError as exc:
                attempt += 1
                if (
                    attempt >= self._SCAN_RETRY_ATTEMPTS
                    or self._TRANSIENT_SCAN_MARKER not in str(exc)
                ):
                    raise
                await asyncio.sleep(
                    self._SCAN_RETRY_DELAY * 2 ** (attempt - 1),
                )

    def _scan_documents(
        self,
        handle: "Collection",
        keys: set[str],
    ) -> list[DocumentSummary]:
        """Aggregate one collection into per-document summaries.

        ``iter_docs`` cannot filter natively, so the filter keys are
        matched in Python against the same digests the native query path
        uses — both routes therefore agree on what a filter means.

        Args:
            handle (`Collection`):
                The open zvec collection.
            keys (`set[str]`):
                Metadata filter keys every record must carry; empty to
                keep all records.

        Returns:
            `list[DocumentSummary]`:
                One summary per distinct ``document_id``.
        """
        output_fields = [
            self._DOCUMENT_ID_FIELD,
            self._SOURCE_FIELD,
            self._CHUNK_FIELD,
        ]
        if keys:
            output_fields.append(self._FILTER_KEY_FIELD)

        # Records are keyed by ``(document_id, chunk_index)``, so
        # ``iter_docs`` yields every record exactly once and the
        # aggregation below is the only bookkeeping needed: memory
        # stays O(documents), not O(chunks).
        summaries: dict[str, DocumentSummary] = {}
        with handle.iter_docs(
            output_fields=output_fields,
            include_vector=False,
        ) as docs:
            for doc in docs:
                if keys and not keys.issubset(
                    doc.field(self._FILTER_KEY_FIELD) or [],
                ):
                    continue

                document_id = doc.field(self._DOCUMENT_ID_FIELD)
                summary = summaries.get(document_id)
                if summary is None:
                    payload = json.loads(doc.field(self._CHUNK_FIELD))
                    summaries[document_id] = DocumentSummary(
                        document_id=document_id,
                        source=doc.field(self._SOURCE_FIELD) or "",
                        chunk_count=1,
                        metadata=dict(payload.get("metadata", {})),
                    )
                else:
                    summary.chunk_count += 1

        return list(summaries.values())

    # ------------------------------------------------------------------
    # Chunk listing
    # ------------------------------------------------------------------

    async def list_chunks(
        self,
        collection: str,
        document_id: str,
        *,
        offset: int = 0,
        limit: int = 30,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """List one document's chunks ordered by ``chunk_index``.

        zvec has no ORDER BY and no filter-only scan, so instead of
        sorting a whole document in memory the page is read by primary
        key: record ids are deterministic in
        ``(document_id, chunk_index)``, which makes
        ``[offset, offset + limit)`` a batch of point lookups.  Missing
        indices are simply absent from the result.

        Args:
            collection (`str`):
                The target collection name.
            document_id (`str`):
                The source document whose chunks should be listed.
            offset (`int`, defaults to ``0``):
                Number of leading chunks to skip.
            limit (`int`, defaults to ``30``):
                Maximum number of chunks to return.
            metadata_filter (`dict[str, Any] | None`, optional):
                Extra ``chunk.metadata`` equality constraints.

        Returns:
            `list[Chunk]`:
                At most ``limit`` chunks, ``chunk_index`` ascending.

        .. note:: The page is the ``chunk_index`` window
            ``[offset, offset + limit)``; ``metadata_filter`` only
            drops records inside that window, so a filtered page can
            hold fewer than ``limit`` chunks even when later indices
            would match.  This matches the ``VectorStoreBase``
            contract, where ``offset`` is a chunk index rather than a
            count of matches.
        """
        if limit <= 0:
            return []

        keys = set(self._filter_keys(metadata=metadata_filter))
        handle = await self._get_collection(collection)
        return await asyncio.to_thread(
            self._fetch_chunk_page,
            handle,
            document_id,
            offset,
            limit,
            keys,
        )

    def _fetch_chunk_page(
        self,
        handle: "Collection",
        document_id: str,
        offset: int,
        limit: int,
        keys: set[str],
    ) -> list[Chunk]:
        """Fetch one ``chunk_index`` window of a document.

        Args:
            handle (`Collection`):
                The open zvec collection.
            document_id (`str`):
                The source document whose chunks should be listed.
            offset (`int`):
                First ``chunk_index`` of the window.
            limit (`int`):
                Width of the window.
            keys (`set[str]`):
                Metadata filter keys a chunk must carry; empty to keep
                every chunk of the window.

        Returns:
            `list[Chunk]`:
                The chunks found in the window, ``chunk_index``
                ascending.
        """
        output_fields = [self._CHUNK_FIELD]
        if keys:
            output_fields.append(self._FILTER_KEY_FIELD)

        chunks: list[Chunk] = []
        for start in range(offset, offset + limit, self._batch_size):
            indexes = range(
                start,
                min(start + self._batch_size, offset + limit),
            )
            ids = [self._record_id(document_id, index) for index in indexes]
            fetched = handle.fetch(
                ids,
                output_fields=output_fields,
                include_vector=False,
            )
            for record_id, index in zip(ids, indexes):
                doc = fetched.get(record_id)
                if doc is None:
                    continue
                # Match the digests written with the record, exactly
                # like `_scan_documents`: recomputing them from the
                # decoded chunk would compare a pre-serialization value
                # with a post-serialization one for types such as
                # `datetime`, and silently drop records `search` finds.
                if keys and not keys.issubset(
                    doc.field(self._FILTER_KEY_FIELD) or [],
                ):
                    continue
                chunks.append(
                    Chunk.model_validate_json(
                        doc.field(self._CHUNK_FIELD),
                    ),
                )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_collection(self, name: str) -> "Collection":
        """Return the cached handle of a collection, opening it if needed.

        Args:
            name (`str`):
                The collection name.

        Returns:
            `Collection`:
                The open zvec collection handle.

        Raises:
            _MetricMismatch: If the collection was built with another
                metric than the configured ``metric_type``.
        """
        handle = self._collections.get(name)
        if handle is not None:
            return handle
        collection_path = self._collection_path(name)
        return await asyncio.to_thread(
            self._open_collection,
            name,
            collection_path,
        )

    def _open_collection(
        self,
        name: str,
        collection_path: str,
    ) -> "Collection":
        """Open one collection exactly once, even under concurrent calls.

        This is the only place a collection handle enters the cache, so
        the metric check here covers every operation — the application
        calls :meth:`create_collection` only when it creates a knowledge
        base, and looks an existing one up through :meth:`has_collection`
        first, which never reaches that method.

        Args:
            name (`str`):
                The collection name.
            collection_path (`str`):
                Directory backing the collection.

        Returns:
            `Collection`:
                The shared zvec collection handle.

        Raises:
            _MetricMismatch: If the collection was built with another
                metric than the configured ``metric_type``.
        """
        import zvec

        with self._lock:
            handle = self._collections.get(name)
            if handle is not None:
                return handle
            handle = zvec.open(collection_path, self._collection_option())
            try:
                self._require_metric(name, handle)
            except Exception:
                # A handle that is not cached is never closed anywhere
                # else, and it keeps the directory lock held, so every
                # failure has to release it — not just a metric
                # mismatch.
                handle.close()
                raise
            self._collections[name] = handle
            return handle

    def _collection_option(self) -> Any:
        """Build the option object used to open a collection.

        Returns:
            `Any`:
                A ``zvec.CollectionOption`` built from ``option_kwargs``.
        """
        import zvec

        return zvec.CollectionOption(**self._option_kwargs)

    def _build_schema(self, name: str, dimensions: int) -> Any:
        """Build the collection schema.

        Args:
            name (`str`):
                The collection name.
            dimensions (`int`):
                The vector dimensionality.

        Returns:
            `Any`:
                A ``zvec.CollectionSchema``.
        """
        import zvec

        index_kwargs: dict[str, Any] = {
            "metric_type": getattr(zvec.MetricType, self._metric_type),
        }
        if self._hnsw_m is not None:
            index_kwargs["m"] = self._hnsw_m
        if self._hnsw_ef_construction is not None:
            index_kwargs["ef_construction"] = self._hnsw_ef_construction

        return zvec.CollectionSchema(
            name=name,
            fields=[
                zvec.FieldSchema(
                    self._DOCUMENT_ID_FIELD,
                    zvec.DataType.STRING,
                    nullable=False,
                ),
                zvec.FieldSchema(
                    self._SOURCE_FIELD,
                    zvec.DataType.STRING,
                    nullable=False,
                ),
                zvec.FieldSchema(
                    self._CHUNK_FIELD,
                    zvec.DataType.STRING,
                    nullable=False,
                ),
                zvec.FieldSchema(
                    self._FILTER_KEY_FIELD,
                    zvec.DataType.ARRAY_STRING,
                    nullable=False,
                    index_param=zvec.InvertIndexParam(),
                ),
            ],
            vectors=[
                zvec.VectorSchema(
                    self._VECTOR_FIELD,
                    zvec.DataType.VECTOR_FP32,
                    dimension=dimensions,
                    index_param=zvec.HnswIndexParam(**index_kwargs),
                ),
            ],
        )

    @classmethod
    def _dimension_of(cls, handle: "Collection") -> int:
        """Read the dimensionality of an existing collection.

        Args:
            handle (`Collection`):
                The open zvec collection.

        Returns:
            `int`:
                The dimension of the dense vector column.
        """
        for vector in handle.schema.vectors:
            if vector.name == cls._VECTOR_FIELD:
                return int(vector.dimension)
        raise ValueError(
            f"Collection at {handle.path} has no "
            f"{cls._VECTOR_FIELD!r} vector column.",
        )

    def _require_metric(self, name: str, handle: "Collection") -> None:
        """Reject a collection built with a different metric.

        The metric lives in the index, so reopening a collection under
        another one leaves the data untouched but breaks
        :meth:`_normalize_score`, which trusts ``metric_type``: a
        ``COSINE`` store reading an ``L2`` collection would turn every
        distance into a similarity and silently reorder results.

        Called before a handle is cached, so the mismatch cannot be
        dodged through the cache on a later call.

        Args:
            name (`str`):
                The collection name, for the error message.
            handle (`Collection`):
                The open zvec collection.

        Raises:
            _MetricMismatch: If the stored metric differs from the
                configured ``metric_type``.
        """
        existing = self._metric_type_of(handle)
        if existing is None or existing == self._metric_type:
            return
        raise _MetricMismatch(
            f"Collection {name!r} already exists with metric "
            f"{existing}, but {self._metric_type} was requested.",
        )

    @classmethod
    def _metric_type_of(cls, handle: "Collection") -> str | None:
        """Read the metric an existing collection was built with.

        Returns ``None`` rather than guessing when the SDK exposes the
        metric in a shape this version does not recognize, so an
        unfamiliar build cannot turn into a false mismatch.

        Args:
            handle (`Collection`):
                The open zvec collection.

        Returns:
            `str | None`:
                The upper-cased metric name, or ``None`` when unknown.
        """
        for vector in handle.schema.vectors:
            if vector.name != cls._VECTOR_FIELD:
                continue
            param = getattr(vector, "index_param", None)
            value = getattr(param, "metric_type", None)
            if value is None and isinstance(param, dict):
                value = param.get("metric_type")
            if value is None:
                value = getattr(vector, "metric_type", None)
            label = getattr(value, "name", None)
            if isinstance(label, str):
                return label.upper()
            if isinstance(value, str):
                return value.upper()
            return None
        return None

    def _collection_path(self, name: str) -> str:
        """Validate a collection name and resolve its directory.

        Args:
            name (`str`):
                The collection name.

        Returns:
            `str`:
                Absolute path of the directory backing the collection.

        Raises:
            ValueError: If the name is not usable as a directory name.
        """
        if not isinstance(name, str) or not (
            _COLLECTION_NAME_PATTERN.fullmatch(name)
        ):
            raise ValueError(
                f"Invalid collection name {name!r}: it must start with a "
                "letter or digit, contain only letters, digits, '_', '.' "
                "or '-', and be at most 128 characters long.",
            )

        root = os.path.abspath(self._path)
        collection_path = os.path.abspath(os.path.join(root, name))
        if os.path.dirname(collection_path) != root:
            # Unreachable through the pattern above; kept so a future
            # relaxation of the name rules cannot escape the root.
            raise ValueError(
                f"Collection name {name!r} escapes the store directory.",
            )
        return collection_path

    @classmethod
    def _filter_keys(
        cls,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Build the filter keys a record or a filter is matched by.

        Args:
            document_id (`str | None`, optional):
                Document id to include, or ``None`` to omit it.
            metadata (`dict[str, Any] | None`, optional):
                Metadata pairs to include.

        Returns:
            `list[str]`:
                The namespaced digests, in a stable order.

        Raises:
            ValueError: If a metadata value is a non-finite float or
                an unordered container such as ``set``.
        """
        keys = []
        if document_id is not None:
            keys.append(cls._document_key(document_id))
        for key, value in (metadata or {}).items():
            keys.append(cls._metadata_key(key, value))
        return keys

    @classmethod
    def _document_key(cls, document_id: str) -> str:
        """Digest identifying one source document.

        Args:
            document_id (`str`):
                The document id.

        Returns:
            `str`:
                The namespaced digest.
        """
        return cls._DOCUMENT_KEY_PREFIX + cls._digest(document_id)

    @classmethod
    def _metadata_key(cls, key: str, value: Any) -> str:
        """Digest identifying one ``key == value`` metadata pair.

        The pair is JSON-encoded before hashing so the match is exact
        and type-sensitive, and so nested values compare independently
        of their key order.  This is also the only place a digest is
        computed: readers compare the stored ``filter_key`` values, so
        the encoding never has to agree with the JSON form a reader
        sees.

        Args:
            key (`str`):
                The metadata key.
            value (`Any`):
                The metadata value.

        Returns:
            `str`:
                The namespaced digest.

        Raises:
            ValueError: If the value is not canonically encodable; see
                :meth:`_canonical_value`.
        """
        encoded = json.dumps(
            [key, cls._canonical_value(value)],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return cls._METADATA_KEY_PREFIX + cls._digest(encoded)

    @classmethod
    def _canonical_value(cls, value: Any) -> Any:
        """Reduce a metadata value to a process-stable JSON shape.

        ``json.dumps(..., default=str)`` alone is not enough: the
        ``str()`` of a ``set`` depends on hash randomization, so the
        digest written by one process would not match the one computed
        by the next — a filter would silently stop matching.  Unordered
        containers are therefore rejected, and the finite-float check
        recurses into nested values instead of looking at the top level
        only.

        Args:
            value (`Any`):
                The metadata value to normalize.

        Returns:
            `Any`:
                A value made of JSON-native types only.

        Raises:
            ValueError: If the value is a non-finite float or an
                unordered container.
        """
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    "Metadata float values must be finite.",
                )
            return value
        if isinstance(value, dict):
            return {
                str(key): cls._canonical_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._canonical_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            raise ValueError(
                "Metadata values must be JSON-serializable: convert "
                f"{type(value).__name__} to a list first, because its "
                "iteration order is not stable across processes.",
            )
        return value

    @staticmethod
    def _digest(value: str) -> str:
        """Hash one value into a filter-safe token.

        Args:
            value (`str`):
                The value to hash.

        Returns:
            `str`:
                The lowercase hex digest.
        """
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _build_expression(cls, keys: list[str]) -> str | None:
        """Render filter keys as a native filter expression.

        Keys are hex digests with a fixed prefix, so the expression
        needs no escaping and no user-supplied text ever reaches the
        filter parser.

        Args:
            keys (`list[str]`):
                The filter keys that must all match.

        Returns:
            `str | None`:
                The expression, or ``None`` when there is nothing to
                filter on.
        """
        if not keys:
            return None
        quoted = ", ".join(f"'{key}'" for key in keys)
        return f"{cls._FILTER_KEY_FIELD} contain_all ({quoted})"

    @classmethod
    def _record_id(cls, document_id: str, chunk_index: int) -> str:
        """Deterministic record id for ``(document_id, chunk_index)``.

        A SHA-256 over the ``\0``-joined pair: ``\0`` cannot occur in
        either field, so no two pairs collide.  Stable ids make indexing
        retries idempotent (via ``upsert``) and let :meth:`list_chunks`
        read a page by primary key.

        Args:
            document_id (`str`):
                The source document id.
            chunk_index (`int`):
                The index of the chunk within the document.

        Returns:
            `str`:
                The 64-character hex digest.
        """
        raw = f"{document_id}\0{chunk_index}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _normalize_score(self, score: float | None) -> float:
        """Convert a native score into a higher-is-better one.

        Args:
            score (`float | None`):
                The score reported by zvec, or ``None`` when absent.

        Returns:
            `float`:
                The normalized score.
        """
        if score is None:
            return 0.0
        if self._metric_type == "COSINE":
            return 1.0 - score
        if self._metric_type == "L2":
            return -score
        return score
