import json
import math

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterable

import chromadb

from src.rag.chunk_builder import RAGChunk
from src.rag.embedding_model import (
    LocalEmbeddingModel
)


@dataclass(slots=True)
class VectorSearchResult:
    """
    Represents one retrieved RAG chunk.
    """

    rank: int
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    distance: float | None
    similarity: float | None

    @property
    def chunk_type(self) -> str:

        return str(
            self.metadata.get(
                "chunk_type",
                ""
            )
        )

    def to_dict(self) -> dict[str, Any]:

        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": dict(
                self.metadata
            ),
            "distance": self.distance,
            "similarity": self.similarity
        }


class TariffVectorStore:
    """
    Persistent Chroma vector database for electricity tariff
    knowledge chunks.

    The store uses externally generated Sentence Transformer
    embeddings rather than Chroma's default embedding model.
    """

    DEFAULT_PERSIST_DIRECTORY = (
        "output/chroma"
    )

    DEFAULT_COLLECTION_NAME = (
        "electricity_tariff_intelligence"
    )

    def __init__(
        self,
        embedding_model: (
            LocalEmbeddingModel | None
        ) = None,
        persist_directory: str = (
            DEFAULT_PERSIST_DIRECTORY
        ),
        collection_name: str = (
            DEFAULT_COLLECTION_NAME
        ),
        batch_size: int = 64
    ) -> None:

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be greater "
                "than zero."
            )

        cleaned_collection_name = (
            collection_name.strip()
        )

        if not cleaned_collection_name:

            raise ValueError(
                "collection_name cannot be empty."
            )

        self.embedding_model = (
            embedding_model
            or LocalEmbeddingModel()
        )

        self.persist_directory = str(
            Path(
                persist_directory
            )
        )

        self.collection_name = (
            cleaned_collection_name
        )

        self.batch_size = batch_size

        Path(
            self.persist_directory
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = (
            chromadb.PersistentClient(
                path=self.persist_directory
            )
        )

        self.collection = (
            self._get_or_create_collection()
        )

    @property
    def count(self) -> int:
        """
        Returns the number of indexed chunks.
        """

        return int(
            self.collection.count()
        )

    def upsert_chunks(
        self,
        chunks: Iterable[RAGChunk],
        rebuild: bool = False
    ) -> int:
        """
        Embeds and stores RAG chunks.

        Args:
            chunks:
                Chunks to store.

            rebuild:
                When True, deletes and recreates the collection
                before indexing.

        Returns:
            Number of chunks supplied to this operation.
        """

        chunk_list = list(
            chunks
        )

        if not chunk_list:

            raise ValueError(
                "At least one RAG chunk is required."
            )

        self._validate_chunks(
            chunk_list
        )

        if rebuild:

            self.rebuild_collection()

        for start_index in range(
            0,
            len(chunk_list),
            self.batch_size
        ):

            batch = chunk_list[
                start_index:
                start_index
                + self.batch_size
            ]

            documents = [
                chunk.content
                for chunk in batch
            ]

            embeddings = (
                self.embedding_model
                .embed_documents(
                    documents
                )
            )

            ids = [
                chunk.chunk_id
                for chunk in batch
            ]

            metadatas = [
                self._sanitize_metadata(
                    chunk.metadata
                )
                for chunk in batch
            ]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )

        return len(
            chunk_list
        )

    def search(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None
    ) -> list[VectorSearchResult]:
        """
        Performs semantic search using the local embedding model.

        Example filters:

            {"chunk_type": "RATE"}

            {"chunk_type": "COMPARISON"}

            {
                "$and": [
                    {"chunk_type": "RATE"},
                    {
                        "source_file":
                        "Oncor_May_1_2023.pdf"
                    }
                ]
            }
        """

        cleaned_query = " ".join(
            query.split()
        )

        if not cleaned_query:

            raise ValueError(
                "The search query cannot be empty."
            )

        if n_results <= 0:

            raise ValueError(
                "n_results must be greater "
                "than zero."
            )

        collection_count = self.count

        if collection_count == 0:

            return []

        query_embedding = (
            self.embedding_model
            .embed_query(
                cleaned_query
            )
        )

        query_arguments: dict[
            str,
            Any
        ] = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": min(
                n_results,
                collection_count
            ),
            "include": [
                "documents",
                "metadatas",
                "distances"
            ]
        }

        if where is not None:

            query_arguments[
                "where"
            ] = where

        query_result = (
            self.collection.query(
                **query_arguments
            )
        )

        ids = self._first_result_list(
            query_result.get(
                "ids"
            )
        )

        documents = (
            self._first_result_list(
                query_result.get(
                    "documents"
                )
            )
        )

        metadatas = (
            self._first_result_list(
                query_result.get(
                    "metadatas"
                )
            )
        )

        distances = (
            self._first_result_list(
                query_result.get(
                    "distances"
                )
            )
        )

        search_results = []

        for index, chunk_id in enumerate(
            ids
        ):

            content = ""

            if index < len(
                documents
            ):

                content = str(
                    documents[index]
                    or ""
                )

            metadata: dict[
                str,
                Any
            ] = {}

            if index < len(
                metadatas
            ):

                raw_metadata = (
                    metadatas[index]
                )

                if isinstance(
                    raw_metadata,
                    dict
                ):

                    metadata = dict(
                        raw_metadata
                    )

            distance = None

            if index < len(
                distances
            ):

                raw_distance = (
                    distances[index]
                )

                if raw_distance is not None:

                    distance = float(
                        raw_distance
                    )

            similarity = (
                self._distance_to_similarity(
                    distance
                )
            )

            search_results.append(
                VectorSearchResult(
                    rank=index + 1,
                    chunk_id=str(
                        chunk_id
                    ),
                    content=content,
                    metadata=metadata,
                    distance=distance,
                    similarity=similarity
                )
            )

        return search_results

    def get_chunk(
        self,
        chunk_id: str
    ) -> VectorSearchResult | None:
        """
        Retrieves one stored chunk by its ID.
        """

        cleaned_chunk_id = (
            chunk_id.strip()
        )

        if not cleaned_chunk_id:

            raise ValueError(
                "chunk_id cannot be empty."
            )

        result = self.collection.get(
            ids=[
                cleaned_chunk_id
            ],
            include=[
                "documents",
                "metadatas"
            ]
        )

        ids = result.get(
            "ids"
        ) or []

        if not ids:

            return None

        documents = result.get(
            "documents"
        ) or []

        metadatas = result.get(
            "metadatas"
        ) or []

        content = ""

        if documents:

            content = str(
                documents[0]
                or ""
            )

        metadata = {}

        if (
            metadatas
            and isinstance(
                metadatas[0],
                dict
            )
        ):

            metadata = dict(
                metadatas[0]
            )

        return VectorSearchResult(
            rank=1,
            chunk_id=str(
                ids[0]
            ),
            content=content,
            metadata=metadata,
            distance=None,
            similarity=None
        )

    def rebuild_collection(
        self
    ) -> None:
        """
        Deletes only this collection and recreates it.

        Other Chroma collections inside the same persistence
        directory are not affected.
        """

        existing_names = {
            self._collection_name(
                collection
            )
            for collection
            in self.client.list_collections()
        }

        if (
            self.collection_name
            in existing_names
        ):

            self.client.delete_collection(
                name=self.collection_name
            )

        self.collection = (
            self._get_or_create_collection()
        )

    def _get_or_create_collection(
        self
    ):
        """
        Creates a cosine-distance collection.

        The fallback supports older Chroma versions that used
        metadata-based HNSW configuration.
        """

        collection_metadata = {
            "description": (
                "Electricity tariff rate, "
                "comparison and section chunks"
            ),
            "embedding_model": (
                self.embedding_model.model_name
            )
        }

        try:

            return (
                self.client
                .get_or_create_collection(
                    name=self.collection_name,
                    metadata=(
                        collection_metadata
                    ),
                    embedding_function=None,
                    configuration={
                        "hnsw": {
                            "space": "cosine"
                        }
                    }
                )
            )

        except TypeError:

            legacy_metadata = dict(
                collection_metadata
            )

            legacy_metadata[
                "hnsw:space"
            ] = "cosine"

            return (
                self.client
                .get_or_create_collection(
                    name=self.collection_name,
                    metadata=legacy_metadata,
                    embedding_function=None
                )
            )

    def _validate_chunks(
        self,
        chunks: list[RAGChunk]
    ) -> None:

        chunk_ids = []

        for index, chunk in enumerate(
            chunks
        ):

            if not isinstance(
                chunk,
                RAGChunk
            ):

                raise TypeError(
                    "Chunk at index "
                    f"{index} is not a RAGChunk."
                )

            if not chunk.chunk_id.strip():

                raise ValueError(
                    "Chunk at index "
                    f"{index} has no chunk ID."
                )

            if not chunk.content.strip():

                raise ValueError(
                    "Chunk at index "
                    f"{index} has empty content."
                )

            chunk_ids.append(
                chunk.chunk_id
            )

        if (
            len(chunk_ids)
            != len(set(chunk_ids))
        ):

            raise ValueError(
                "Duplicate chunk IDs were supplied."
            )

    def _sanitize_metadata(
        self,
        metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Converts metadata to Chroma-compatible scalar values.
        """

        sanitized = {}

        for raw_key, raw_value in (
            metadata.items()
        ):

            key = str(
                raw_key
            ).strip()

            if not key:
                continue

            sanitized[
                key
            ] = self._sanitize_metadata_value(
                raw_value
            )

        return sanitized

    def _sanitize_metadata_value(
        self,
        value: Any
    ) -> str | int | float | bool:

        if value is None:
            return ""

        if isinstance(
            value,
            bool
        ):

            return value

        if isinstance(
            value,
            int
        ):

            return value

        if isinstance(
            value,
            float
        ):

            if math.isfinite(
                value
            ):

                return value

            return str(
                value
            )

        if isinstance(
            value,
            str
        ):

            return value

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
                dict
            )
        ):

            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str
            )

        return str(
            value
        )

    def _distance_to_similarity(
        self,
        distance: float | None
    ) -> float | None:
        """
        Chroma cosine distance is 1 - cosine similarity.
        """

        if distance is None:
            return None

        similarity = (
            1.0
            - distance
        )

        return max(
            -1.0,
            min(
                1.0,
                similarity
            )
        )

    def _first_result_list(
        self,
        value: Any
    ) -> list[Any]:

        if not value:
            return []

        if not isinstance(
            value,
            list
        ):

            return []

        first_value = value[0]

        if not isinstance(
            first_value,
            list
        ):

            return []

        return first_value

    def _collection_name(
        self,
        collection: Any
    ) -> str:

        name = getattr(
            collection,
            "name",
            collection
        )

        return str(
            name
        )