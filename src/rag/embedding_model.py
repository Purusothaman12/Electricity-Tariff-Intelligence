from collections.abc import Sequence
from typing import Any

import numpy as np
from sentence_transformers import (
    SentenceTransformer
)


class LocalEmbeddingModel:
    """
    Creates local semantic embeddings for tariff RAG chunks.

    Documents and user questions are encoded separately because
    tariff retrieval is an asymmetric semantic-search task:

        query:
            What was the Residential Customer Charge in 2023?

        document:
            Tariff rate record. Schedule: Residential Service...
    """

    DEFAULT_MODEL_NAME = (
        "sentence-transformers/"
        "all-MiniLM-L6-v2"
    )

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False
    ) -> None:

        if batch_size <= 0:

            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = (
            normalize_embeddings
        )
        self.show_progress_bar = (
            show_progress_bar
        )

        self._model: (
            SentenceTransformer | None
        ) = None

    @property
    def model(self) -> SentenceTransformer:
        """
        Loads the embedding model only when first required.
        """

        if self._model is None:

            model_arguments: dict[
                str,
                Any
            ] = {}

            if self.device:

                model_arguments[
                    "device"
                ] = self.device

            self._model = (
                SentenceTransformer(
                    self.model_name,
                    **model_arguments
                )
            )

        return self._model

    @property
    def embedding_dimension(self) -> int:
        """
        Returns the number of dimensions in each embedding.
        """

        dimension = (
            self.model
            .get_sentence_embedding_dimension()
        )

        if dimension is None:

            raise RuntimeError(
                "The embedding model did not report "
                "its output dimension."
            )

        return int(
            dimension
        )

    def embed_documents(
        self,
        documents: Sequence[str]
    ) -> list[list[float]]:
        """
        Embeds tariff chunks for storage in the vector database.
        """

        cleaned_documents = (
            self._validate_texts(
                texts=documents,
                text_type="document"
            )
        )

        embeddings = self._encode_documents(
            cleaned_documents
        )

        return self._to_python_vectors(
            embeddings
        )

    def embed_query(
        self,
        query: str
    ) -> list[float]:
        """
        Embeds one user question for semantic retrieval.
        """

        cleaned_query = self._validate_query(
            query
        )

        embedding = self._encode_query(
            cleaned_query
        )

        embedding_array = np.asarray(
            embedding,
            dtype=np.float32
        )

        if embedding_array.ndim == 2:

            if embedding_array.shape[0] != 1:

                raise RuntimeError(
                    "Query embedding returned more "
                    "than one vector."
                )

            embedding_array = (
                embedding_array[0]
            )

        if embedding_array.ndim != 1:

            raise RuntimeError(
                "Query embedding must be "
                "one-dimensional."
            )

        return embedding_array.tolist()

    def similarity(
        self,
        left_vector: Sequence[float],
        right_vector: Sequence[float]
    ) -> float:
        """
        Calculates cosine similarity between two embeddings.
        """

        left = np.asarray(
            left_vector,
            dtype=np.float32
        )

        right = np.asarray(
            right_vector,
            dtype=np.float32
        )

        if left.ndim != 1 or right.ndim != 1:

            raise ValueError(
                "Similarity inputs must each be "
                "one-dimensional vectors."
            )

        if left.shape != right.shape:

            raise ValueError(
                "Similarity vectors must have "
                "matching dimensions."
            )

        left_norm = float(
            np.linalg.norm(
                left
            )
        )

        right_norm = float(
            np.linalg.norm(
                right
            )
        )

        if (
            left_norm == 0.0
            or right_norm == 0.0
        ):

            raise ValueError(
                "Cannot calculate similarity for "
                "a zero-length embedding vector."
            )

        similarity_value = float(
            np.dot(
                left,
                right
            )
            / (
                left_norm
                * right_norm
            )
        )

        return similarity_value

    def _encode_documents(
        self,
        documents: list[str]
    ) -> np.ndarray:
        """
        Uses encode_document when supported by the installed
        Sentence Transformers version. Falls back to encode for
        backward compatibility.
        """

        encoding_arguments = {
            "batch_size": self.batch_size,
            "show_progress_bar": (
                self.show_progress_bar
            ),
            "convert_to_numpy": True,
            "normalize_embeddings": (
                self.normalize_embeddings
            )
        }

        encode_document = getattr(
            self.model,
            "encode_document",
            None
        )

        if callable(
            encode_document
        ):

            embeddings = encode_document(
                documents,
                **encoding_arguments
            )

        else:

            embeddings = self.model.encode(
                documents,
                **encoding_arguments
            )

        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    def _encode_query(
        self,
        query: str
    ) -> np.ndarray:
        """
        Uses encode_query when supported. Falls back to encode
        for backward compatibility.
        """

        encoding_arguments = {
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": (
                self.normalize_embeddings
            )
        }

        encode_query = getattr(
            self.model,
            "encode_query",
            None
        )

        if callable(
            encode_query
        ):

            embedding = encode_query(
                query,
                **encoding_arguments
            )

        else:

            embedding = self.model.encode(
                query,
                **encoding_arguments
            )

        return np.asarray(
            embedding,
            dtype=np.float32
        )

    def _to_python_vectors(
        self,
        embeddings: np.ndarray
    ) -> list[list[float]]:

        embedding_array = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embedding_array.ndim != 2:

            raise RuntimeError(
                "Document embeddings must form "
                "a two-dimensional matrix."
            )

        expected_dimension = (
            self.embedding_dimension
        )

        if (
            embedding_array.shape[1]
            != expected_dimension
        ):

            raise RuntimeError(
                "Unexpected embedding dimension. "
                f"Expected {expected_dimension}, "
                f"received "
                f"{embedding_array.shape[1]}."
            )

        return embedding_array.tolist()

    def _validate_texts(
        self,
        texts: Sequence[str],
        text_type: str
    ) -> list[str]:

        if isinstance(
            texts,
            str
        ):

            raise TypeError(
                f"{text_type.capitalize()} texts "
                "must be supplied as a sequence, "
                "not as one string."
            )

        cleaned_texts = []

        for index, text in enumerate(
            texts
        ):

            if not isinstance(
                text,
                str
            ):

                raise TypeError(
                    f"{text_type.capitalize()} at "
                    f"index {index} must be a string."
                )

            cleaned_text = (
                " ".join(
                    text.split()
                )
            )

            if not cleaned_text:

                raise ValueError(
                    f"{text_type.capitalize()} at "
                    f"index {index} is empty."
                )

            cleaned_texts.append(
                cleaned_text
            )

        if not cleaned_texts:

            raise ValueError(
                f"At least one {text_type} "
                "is required."
            )

        return cleaned_texts

    def _validate_query(
        self,
        query: str
    ) -> str:

        if not isinstance(
            query,
            str
        ):

            raise TypeError(
                "The query must be a string."
            )

        cleaned_query = " ".join(
            query.split()
        )

        if not cleaned_query:

            raise ValueError(
                "The query cannot be empty."
            )

        return cleaned_query