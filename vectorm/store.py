from __future__ import annotations

from typing import Sequence
from uuid import UUID

from vectorm.backend.base import VectorStoreBackend
from vectorm.document import Document, ScoredDocument, TDocument
from vectorm.filter import FieldExpression, Filter, SortOrder
from vectorm.migrate import Migrator
from vectorm.paginate import Page
from vectorm.types import (
    FieldIndexParams,
    SparseTensor,
    SparseTensorLike,
    Tensor,
    TensorLike,
    is_sparse_tensor_like,
    is_tensor_like,
)


class VectorStore:
    def __init__(
        self,
        backend: VectorStoreBackend,
    ) -> None:
        self._backend = backend

    async def __aenter__(self) -> VectorStore:
        await self._backend.initialize()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._backend.close()

    @property
    def backend(self) -> VectorStoreBackend:
        """
        The underlying vector store backend.
        """
        return self._backend

    async def upgrade_migrations(
        self,
        migrator: Migrator,
        revision_id: str = "head",
    ) -> None:
        """
        Upgrade the vector store to the specified revision.

        :param migrator: The migrator instance containing the revision chain.
        :type migrator: Migrator
        :param revision_id: The target revision ID to upgrade to.
        :type revision_id: str
        :return: None
        :rtype: None
        """
        await migrator.upgrade(
            backend=self._backend,
            revision_id=revision_id,
        )

    async def downgrade_migrations(
        self,
        migrator: Migrator,
        revision_id: str = "tail",
    ) -> None:
        """
        Downgrade the vector store to the specified revision.

        :param migrator: The migrator instance containing the revision chain.
        :type migrator: Migrator
        :param revision_id: The target revision ID to downgrade to.
        :type revision_id: str
        :return: None
        :rtype: None
        """
        await migrator.downgrade(
            backend=self._backend,
            revision_id=revision_id,
        )

    async def save_documents(
        self,
        documents: Sequence[Document],
        *,
        collection_name: str | None = None,
        **kwargs,
    ) -> None:
        """
        Save documents in the vector store.

        :param documents: A sequence of documents to save.
        :type documents: Sequence[Document]
        :param collection_name: The name of the collection to save documents into.
        If None, the collection name is inferred from the document type.
        :type collection_name: str | None
        :return: None
        :rtype: None
        """
        if not documents:
            return

        # We want to group the documents into chunks by their collection name
        # and save them in batches to minimize the number of calls to the backend if
        # there was no collection_name provided.
        if not collection_name:
            collection_map = {
                document.collection_name(): [
                    doc
                    for doc in documents
                    if doc.collection_name() == document.collection_name()
                ]
                for document in documents
            }
        else:
            collection_map = {collection_name: list(documents)}

        for coll_name, docs in collection_map.items():
            await self._backend.save_documents(
                collection_name=coll_name,
                documents=docs,
                **kwargs,
            )

    async def delete_documents(
        self,
        document_ids: Sequence[UUID],
        *,
        collection_name: str,
        **kwargs,
    ) -> None:
        """
        Delete documents from the vector store.

        :param document_ids: A sequence of document IDs to delete.
        :type document_ids: Sequence[UUID]
        :param collection_name: The name of the collection to delete documents from.
        :type collection_name: str
        :return: None
        :rtype: None
        """
        if not document_ids:
            return

        await self._backend.delete_documents(
            collection_name=collection_name,
            document_ids=[
                doc_id if isinstance(doc_id, UUID) else UUID(doc_id)
                for doc_id in document_ids
            ],
            **kwargs,
        )

    async def get_documents(
        self,
        document_ids: Sequence[UUID],
        document_type: type[TDocument],
        *,
        collection_name: str | None = None,
        **kwargs,
    ) -> Sequence[TDocument]:
        """
        Retrieve documents from the vector store by their IDs.

        :param document_ids: A sequence of document IDs to retrieve.
        :type document_ids: Sequence[UUID]
        :param document_type: The type of documents to retrieve.
        :type document_type: type[TDocument]
        :param collection_name: The name of the collection to retrieve documents from.
        :type collection_name: str | None
        :return: A sequence of retrieved documents.
        :rtype: Sequence[TDocument]
        """
        if not document_ids:
            return []

        return await self._backend.get_documents(
            collection_name=collection_name or document_type.collection_name(),
            document_ids=document_ids,
            document_type=document_type,
            **kwargs,
        )

    async def query_documents(
        self,
        document_type: type[TDocument],
        query_tensor: TensorLike | SparseTensorLike,
        tensor_field: FieldExpression,
        filter: Filter | None = None,
        limit: int = 10,
        *,
        collection_name: str | None = None,
        **kwargs,
    ) -> Sequence[ScoredDocument[TDocument]]:
        """
        Query the vector store for documents similar to the given query vector
        and optionally matching the filter.

        :param document_type: The document type to deserialize the documents into.
        :type document_type: type[TDocument]
        :param query_tensor: The query tensor to find similar documents.
        :type query_tensor: Tensor | SparseTensor
        :param tensor_field: The name of the tensor field to use for similarity search.
        :type tensor_field: FieldExpression
        :param filter: An optional filter to apply to the documents.
        :type filter: Filter | None
        :param limit: The maximum number of documents to return. Default is 10.
        :type limit: int
        :param collection_name: The name of the collection to query.
        :type collection_name: str | None
        :return: A sequence of scored documents matching the query and filter.
        :rtype: Sequence[ScoredDocument[TDocument]]
        """
        if is_sparse_tensor_like(query_tensor):
            query_tensor = SparseTensor(query_tensor)
        elif is_tensor_like(query_tensor):
            query_tensor = Tensor(query_tensor)
        else:
            raise ValueError("query_tensor must be like a Tensor or SparseTensor")

        return await self._backend.query_documents(
            collection_name=collection_name or document_type.collection_name(),
            query_tensor=query_tensor,
            tensor_field=tensor_field,
            document_type=document_type,
            filter=filter,
            limit=limit,
            **kwargs,
        )

    async def find_documents(
        self,
        document_type: type[TDocument],
        filter: Filter | None = None,
        limit: int = 10,
        cursor: str | None = None,
        sort: tuple[FieldExpression, SortOrder] | None = None,
        *,
        collection_name: str | None = None,
        **kwargs,
    ) -> Page[TDocument]:
        """
        Find documents in the vector store matching the given filter.

        :param document_type: The type of documents to find.
        :type document_type: type[TDocument]
        :param filter: An optional filter to apply to the documents.
        :type filter: Filter | None
        :param limit: The maximum number of documents to return. Default is 10.
        :type limit: int
        :param cursor: An optional cursor for pagination.
        :type cursor: str | None
        :param sort: An optional sort order as a tuple of FieldExpression and SortOrder.
        :type sort: tuple[FieldExpression, SortOrder] | None
        :param collection_name: The name of the collection to find documents in.
        If None, the collection name is inferred from the document type.
        :type collection_name: str | None
        :return: A page of documents matching the filter.
        :rtype: Page[TDocument]
        """
        return await self._backend.find_documents(
            collection_name=collection_name or document_type.collection_name(),
            document_type=document_type,
            filter=filter,
            limit=limit,
            cursor=cursor,
            sort=sort,
            **kwargs,
        )

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
        return await self._backend.count_documents(
            collection_name=collection_name,
            filter=filter,
            **kwargs,
        )

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
        :param document_type: An optional document type to infer the schema from.
        :type document_type: type[Document] | None
        :return: None
        :rtype: None
        """
        await self._backend.create_collection(
            collection_name=collection_name,
            document_type=document_type,
            **kwargs,
        )

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
        await self._backend.delete_collection(
            collection_name=collection_name,
            **kwargs,
        )

    async def list_collections(self, **kwargs) -> list[str]:
        """
        List all collections in the vector store.

        :return: A list of collection names.
        :rtype: list[str]
        """
        return await self._backend.list_collections(**kwargs)

    async def collection_exists(self, collection_name: str, **kwargs) -> bool:
        """
        Check if a collection exists in the vector store.

        :param collection_name: The name of the collection to check.
        :type collection_name: str
        :return: True if the collection exists, False otherwise.
        :rtype: bool
        """
        return await self._backend.collection_exists(
            collection_name=collection_name,
            **kwargs,
        )

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
        :return: None
        :rtype: None
        """
        await self._backend.create_index(
            collection_name=collection_name,
            field_name=field_name,
            index_params=index_params,
            **kwargs,
        )

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
        :param field_name: The name of the field to delete the index from.
        :type field_name: FieldExpression
        :return: None
        :rtype: None
        """
        await self._backend.delete_index(
            collection_name=collection_name,
            field_name=field_name,
            **kwargs,
        )

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
        return await self._backend.index_exists(
            collection_name=collection_name,
            field_name=field_name,
            **kwargs,
        )
