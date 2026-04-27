from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, TypeVar
from uuid import UUID

from vectorm.document import Document, ScoredDocument, TDocument
from vectorm.filter import ExpressionVisitor, FieldExpression, Filter, SortOrder
from vectorm.paginate import Page
from vectorm.types import FieldIndexParams, SparseTensor, Tensor


TBackend = TypeVar("TBackend", bound="VectorStoreBackend")


class VectorStoreBackend(ABC):
    backend: str
    expression_visitor: ExpressionVisitor

    async def __aenter__(self: TBackend) -> TBackend:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def initialize(self) -> None:
        return

    async def close(self) -> None:
        return

    @abstractmethod
    async def current_revision_id(self) -> str | None:
        """
        Get the current migration revision ID of the backend.

        :return: The current revision ID, or None if not applicable.
        :rtype: str | None
        """
        ...

    @abstractmethod
    async def set_revision_id(self, revision_id: str) -> None:
        """
        Set the current migration revision ID of the backend.

        :param revision_id: The revision ID to set.
        :type revision_id: str
        :return: None
        :rtype: None
        """
        ...

    @abstractmethod
    async def save_documents(
        self,
        collection_name: str,
        documents: Sequence[Document],
        **kwargs
    ) -> None:
        """
        Save documents in the vector store.

        :param collection_name: The name of the collection to save documents into.
        :type collection_name: str
        :param documents: A sequence of documents to save.
        :type documents: Sequence[Document]
        :return: None
        :rtype: None
        """
        ...

    @abstractmethod
    async def delete_documents(
        self,
        collection_name: str,
        document_ids: Sequence[UUID],
        **kwargs
    ) -> None:
        """
        Delete documents from the vector store.

        :param collection_name: The name of the collection to delete documents from.
        :type collection_name: str
        :param document_ids: A sequence of document IDs to delete.
        :type document_ids: Sequence[UUID]
        :return: None
        :rtype: None
        """
        ...

    @abstractmethod
    async def get_documents(
        self,
        collection_name: str,
        document_ids: Sequence[UUID],
        document_type: type[TDocument],
        **kwargs,
    ) -> Sequence[TDocument]:
        """
        Retrieve documents from the vector store by their IDs.

        :param collection_name: The name of the collection to retrieve documents from.
        :type collection_name: str
        :param document_ids: A sequence of document IDs to retrieve.
        :type document_ids: Sequence[UUID]
        :param document_type: The document type to deserialize the documents into.
        :type document_type: type[TDocument]
        :return: A sequence of retrieved documents.
        :rtype: Sequence[TDocument]
        """
        ...

    @abstractmethod
    async def query_documents(
        self,
        collection_name: str,
        query_tensor: Tensor | SparseTensor,
        tensor_field: FieldExpression,
        document_type: type[TDocument],
        filter: Filter | None = None,
        limit: int = 10,
        **kwargs,
    ) -> Sequence[ScoredDocument[TDocument]]:
        """
        Query the vector store for documents similar to the given query vector
        and optionally matching the filter.

        :param collection_name: The name of the collection to query.
        :type collection_name: str
        :param query_tensor: The query tensor to find similar documents.
        :type query_tensor: Tensor | SparseTensor
        :param tensor_field: The name of the tensor field to use for similarity search.
        :type tensor_field: FieldExpression
        :param document_type: The document type to deserialize the documents into.
        :type document_type: type[TDocument]
        :param filter: An optional filter to apply to the documents.
        :type filter: Filter | None
        :param limit: The maximum number of documents to return. Default is 10.
        :type limit: int
        :return: A sequence of scored documents matching the query and filter.
        :rtype: Sequence[ScoredDocument[TDocument]]
        """
        ...

    @abstractmethod
    async def find_documents(
        self,
        collection_name: str,
        document_type: type[TDocument],
        filter: Filter | None = None,
        limit: int = 10,
        cursor: str | None = None,
        sort: tuple[FieldExpression, SortOrder] | None = None,
        **kwargs,
    ) -> Page[TDocument]:
        """
        Find documents in the vector store matching the filter.

        :param collection_name: The name of the collection to find documents in.
        :type collection_name: str
        :param document_type: The document type to deserialize the documents into.
        :type document_type: type[TDocument]
        :param filter: An optional filter to apply to the documents.
        :type filter: Filter | None
        :param limit: The maximum number of documents to return. Default is 10.
        :type limit: int
        :param cursor: An optional cursor for pagination.
        :type cursor: str | None
        :param sort: An optional sort order as a tuple of FieldExpression and SortOrder.
        :type sort: tuple[FieldExpression, SortOrder] | None
        :return: A page of documents matching the filter.
        :rtype: Page[TDocument]
        """
        ...

    @abstractmethod
    async def count_documents(
        self,
        collection_name: str,
        filter: Filter | None = None,
        **kwargs,
    ) -> int:
        """
        Count the number of documents in the collection matching the filter.

        :param collection_name: The name of the collection to count documents in.
        :type collection_name: str
        :param filter: An optional filter to apply to the documents.
        :type filter: Filter | None
        :return: The count of matching documents.
        :rtype: int
        """
        ...

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        document_type: type[Document] | None = None,
        **kwargs,
    ) -> None:
        """
        Create a new collection in the vector store.

        :param collection_name: The name of the collection to create.
        :type collection_name: str
        :param document_type: The document type associated with the collection.
        If None, a collection with no specific schema is created.
        :type document_type: type[Document] | None
        :return: None
        :rtype: None
        """
        ...

    @abstractmethod
    async def delete_collection(
        self,
        collection_name: str,
        **kwargs,
    ) -> None:
        """
        Delete a collection from the vector store.

        :param collection_name: The name of the collection to delete.
        :type collection_name: str
        :return: None
        :rtype: None
        """
        ...

    @abstractmethod
    async def list_collections(self, **kwargs) -> list[str]:
        """
        List all collections in the vector store.

        :return: A list of collection names.
        :rtype: list[str]
        """
        ...

    @abstractmethod
    async def collection_exists(self, collection_name: str, **kwargs) -> bool:
        """
        Check if a collection exists in the vector store.

        :param collection_name: The name of the collection to check.
        :type collection_name: str
        :return: True if the collection exists, False otherwise.
        :rtype: bool
        """
        ...

    @abstractmethod
    async def create_index(
        self,
        collection_name: str,
        field_name: FieldExpression,
        index_params: FieldIndexParams,
        **kwargs,
    ) -> None:
        """
        Create an index on a field in the collection.

        :param collection_name: The name of the collection.
        :type collection_name: str
        :param field_name: The name of the field to index.
        :type field_name: FieldExpression
        :param index_params: Parameters for the index.
        :type index_params: FieldIndexParams
        :return: None
        :rtype: None
        """
        ...

    @abstractmethod
    async def delete_index(
        self,
        collection_name: str,
        field_name: FieldExpression,
        **kwargs,
    ) -> None:
        """
        Delete an index on a field in the collection.

        :param collection_name: The name of the collection.
        :type collection_name: str
        :param field_name: The name of the field whose index to delete.
        :type field_name: FieldExpression
        :return: None
        :rtype: None
        """
        ...

    @abstractmethod
    async def index_exists(
        self,
        collection_name: str,
        field_name: FieldExpression,
        **kwargs,
    ) -> bool:
        """
        Check if an index exists on a field in the collection.

        :param collection_name: The name of the collection.
        :type collection_name: str
        :param field_name: The name of the field to check for an index.
        :type field_name: FieldExpression
        :return: True if the index exists, False otherwise.
        :rtype: bool
        """
        ...
