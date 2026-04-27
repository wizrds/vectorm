from typing import Any, Awaitable, Callable, Sequence
from uuid import UUID

from qdrant_client import AsyncQdrantClient, models

from vectorm.backend.base import VectorStoreBackend
from vectorm.backend.qdrant.utils import (
    document_from_record,
    field_schema_from_field_index_params,
    normalize_tensor,
    point_from_document,
    scored_document_from_scored_point,
    sort_order_to_qdrant_direction,
    sparse_vector_params_from_sparse_tensor,
    vector_params_from_tensor,
)
from vectorm.backend.qdrant.visitor import QdrantExpressionVisitor
from vectorm.document import Document, ScoredDocument, TDocument
from vectorm.filter import FieldExpression, Filter, SortOrder
from vectorm.paginate import Page, decode_cursor, encode_cursor
from vectorm.types import FieldIndexParams, SparseTensor, Tensor


MIGRATIONS_COLLECTION = "_vectorm_migrations"


class QdrantVectorStoreBackend(VectorStoreBackend):
    backend = "qdrant"
    expression_visitor = QdrantExpressionVisitor()

    def __init__(
        self,
        url: str = ":memory:",
        *,
        prefer_grpc: bool = False,
        https: bool | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        force_disable_check_same_thread: bool = False,
        grpc_options: dict[str, Any] | None = None,
        auth_token_provider: Callable[[], str] | Callable[[], Awaitable[str]] | None = None,
        cloud_inference: bool = False,
        local_inference_batch_size: int | None = None,
        check_compatibility: bool = True,
        **kwargs,
    ) -> None:
        self.client = AsyncQdrantClient(
            url,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            timeout=timeout,
            force_disable_check_same_thread=force_disable_check_same_thread,
            grpc_options=grpc_options,
            auth_token_provider=auth_token_provider,
            cloud_inference=cloud_inference,
            local_inference_batch_size=local_inference_batch_size,
            check_compatibility=check_compatibility,
            **kwargs,
        )

    async def initialize(self) -> None:
        return

    async def close(self) -> None:
        await self.client.close()

    async def current_revision_id(self) -> str | None:
        if MIGRATIONS_COLLECTION not in [
            collection.name
            for collection in (await self.client.get_collections()).collections
        ]:
            return None

        records = await self.client.retrieve(
            collection_name=MIGRATIONS_COLLECTION,
            ids=[0],
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None

        return (records[-1].payload or {}).get("revision_id", None)

    async def set_revision_id(self, revision_id: str) -> None:
        if MIGRATIONS_COLLECTION not in [
            collection.name
            for collection in (await self.client.get_collections()).collections
        ]:
            await self.client.recreate_collection(
                collection_name=MIGRATIONS_COLLECTION,
                vectors_config=models.VectorParams(
                    size=1,
                    distance=models.Distance.COSINE,
                )
            )

        await self.client.upsert(
            collection_name=MIGRATIONS_COLLECTION,
            points=[
                models.PointStruct(
                    id=0,
                    vector=[0.0],
                    payload={
                        "revision_id": revision_id,
                    }
                )
            ]
        )

    async def save_documents(
        self,
        collection_name: str,
        documents: Sequence[Document],
        **kwargs,
    ) -> None:
        await self.client.upsert(
            collection_name=collection_name,
            points=[
                point_from_document(document)
                for document in documents
            ],
            **kwargs,
        )

    async def delete_documents(
        self,
        collection_name: str,
        document_ids: Sequence[UUID],
        **kwargs,
    ) -> None:
        await self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=[str(document_id) for document_id in document_ids]
            ),
            **kwargs,
        )

    async def get_documents(
        self,
        collection_name: str,
        document_ids: Sequence[UUID],
        document_type: type[TDocument],
        **kwargs,
    ) -> Sequence[TDocument]:
        return [
            document_from_record(record, document_type)
            for record in await self.client.retrieve(
                collection_name=collection_name,
                ids=[str(document_id) for document_id in document_ids],
                with_payload=True,
                with_vectors=True,
                **kwargs,
            )
        ]

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
        return [
            scored_document_from_scored_point(result, document_type)
            for result in await self.client.search(
                collection_name=collection_name,
                query_vector=(
                    models.NamedVector(
                        name=tensor_field.as_str("."),
                        vector=normalize_tensor(query_tensor),
                    )
                    if issubclass(type(query_tensor), Tensor)
                    else models.NamedSparseVector(
                        name=tensor_field.as_str("."),
                        vector=normalize_tensor(query_tensor),
                    )
                ),
                query_filter=filter.accept(self.expression_visitor) if filter else None,
                with_payload=True,
                with_vectors=True,
                limit=limit,
                **kwargs,
            )
        ]

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
        scroll_filter: models.Filter | None = (
            filter.accept(self.expression_visitor)
            if filter
            else None
        )
        count_filter = scroll_filter

        offset_id, sorted_value = None, None
        if cursor and len((decoded := decode_cursor(cursor.encode()))) == 2:
            offset_id, sorted_value = decoded
            offset_id = UUID(offset_id)

        sort_field, sort_order = sort if sort else (None, None)
        order_by = (
            models.OrderBy(
                key=sort_field.as_str("."),
                direction=sort_order_to_qdrant_direction(sort_order),
                start_from=sorted_value,
            )
            if sort_field and sort_order
            else None
        )

        if offset_id:
            if not scroll_filter:
                scroll_filter = models.Filter(must_not=[])

            must_not = (
                scroll_filter.must_not
                if isinstance(scroll_filter.must_not, list)
                else [scroll_filter.must_not]
                if scroll_filter.must_not
                else []
            )
            must_not.append(models.HasIdCondition(has_id=[str(offset_id)]))
            scroll_filter.must_not = must_not

        count = await self.client.count(
            collection_name=collection_name,
            count_filter=count_filter,
            **kwargs,
        )
        records, next_id = await self.client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=limit,
            order_by=order_by,
            offset=offset_id if not order_by else None,
            with_payload=True,
            with_vectors=True,
            **kwargs,
        )
        documents = [
            document_from_record(record, document_type)
            for record in records
        ]

        next_sorted_value = None
        if not next_id and len(documents) == limit:
            next_id = documents[-1].document_id
            next_sorted_value = (
                sort_field.get_value(documents[-1])
                if sort_field
                else None
            )

        return Page(
            items=documents,
            total=count.count,
            next_cursor=(
                encode_cursor(str(next_id), next_sorted_value)
                if next_id
                else None
            ),
        )

    async def count_documents(
        self,
        collection_name: str,
        filter: Filter | None = None,
        **kwargs,
    ) -> int:
        return (await self.client.count(
            collection_name=collection_name,
            count_filter=filter.accept(self.expression_visitor) if filter else None,
            **kwargs,
        )).count

    async def create_collection(
        self,
        collection_name: str,
        document_type: type[TDocument] | None = None,
        **kwargs,
    ) -> None:
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                name: vector_params_from_tensor(field_type, index_params)
                for name, (field_type, index_params) in (
                    (document_type.get_tensor_fields() if document_type else {})
                    .items()
                )
                if issubclass(field_type, Tensor)
            },
            sparse_vectors_config={
                name: sparse_vector_params_from_sparse_tensor(field_type, index_params)
                for name, (field_type, index_params) in (
                    (document_type.get_tensor_fields() if document_type else {})
                    .items()
                )
                if issubclass(field_type, SparseTensor)
            },
            **kwargs,
        )

    async def delete_collection(
        self,
        collection_name: str,
        **kwargs,
    ) -> None:
        await self.client.delete_collection(
            collection_name=collection_name,
            **kwargs,
        )

    async def list_collections(self, **kwargs) -> list[str]:
        return [
            collection.name
            for collection in (await self.client.get_collections(**kwargs)).collections
        ]

    async def collection_exists(
        self,
        collection_name: str,
        **kwargs,
    ) -> bool:
        return collection_name in [
            collection.name
            for collection in (await self.client.get_collections(**kwargs)).collections
        ]

    async def create_index(
        self,
        collection_name: str,
        field_name: FieldExpression,
        index_params: FieldIndexParams,
        **kwargs,
    ) -> None:
        await self.client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name.as_str("."),
            field_schema=field_schema_from_field_index_params(index_params),
            **kwargs,
        )

    async def delete_index(
        self,
        collection_name: str,
        field_name: FieldExpression,
        **kwargs,
    ) -> None:
        await self.client.delete_payload_index(
            collection_name=collection_name,
            field_name=field_name.as_str("."),
            **kwargs,
        )

    async def index_exists(
        self,
        collection_name: str,
        field_name: FieldExpression,
        **kwargs,
    ) -> bool:
        collection = await self.client.get_collection(
            collection_name=collection_name,
            **kwargs,
        )
        return any(
            index == field_name.as_str(".")
            for index in collection.payload_schema.keys() or []
        )
