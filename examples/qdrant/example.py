import asyncio
from typing import Annotated, Literal

import numpy as np
from rich import print

from vectorm import (
    Dim,
    DistanceMetric,
    Document,
    Field,
    FieldIndexParams,
    FieldIndexDataType,
    TensorIndexParams,
    SortOrder,
    SparseTensor,
    Tensor,
    VectorStore,
    Migrator,
    Revision,
    MigrateOp,
)
from vectorm.backend.qdrant import QdrantVectorStoreBackend


class MyDocument(Document):
    _collection_name_ = "my_documents"

    name: str
    age: int
    status: Literal["active", "inactive"]
    embeddings: Annotated[Tensor[Dim[4], np.float32], TensorIndexParams(metric=DistanceMetric.cosine)]
    sparse_embeddings: SparseTensor[Dim[4], np.float32]


class InitialRevision(Revision):
    revision_id = "initial"

    async def upgrade(self, op: MigrateOp) -> None:
        if not await op.collection_exists(MyDocument.collection_name()):
            await op.create_collection(
                collection_name=MyDocument.collection_name(),
                document_type=MyDocument,
            )

        if not await op.index_exists(MyDocument.collection_name(), Field.age):
            await op.create_index(
                collection_name=MyDocument.collection_name(),
                field_name=Field.age,
                index_params=FieldIndexParams(
                    data_type=FieldIndexDataType.integer,
                )
            )

        if not await op.index_exists(MyDocument.collection_name(), Field.name):
            await op.create_index(
                collection_name=MyDocument.collection_name(),
                field_name=Field.name,
                index_params=FieldIndexParams(
                    data_type=FieldIndexDataType.keyword,
                )
            )

        if not await op.index_exists(MyDocument.collection_name(), Field.status):
            await op.create_index(
                collection_name=MyDocument.collection_name(),
                field_name=Field.status,
                index_params=FieldIndexParams(
                    data_type=FieldIndexDataType.keyword,
                )
            )

    async def downgrade(self, op: MigrateOp) -> None:
        if await op.collection_exists(MyDocument.collection_name()):
            await op.delete_collection(MyDocument.collection_name())


migrator = Migrator([InitialRevision()])

async def main() -> None:
    async with VectorStore(QdrantVectorStoreBackend("http://localhost:6333")) as store:
        await store.upgrade_migrations(migrator)

        if await store.count_documents(MyDocument.collection_name()) == 0:
            await store.save_documents(
                [
                    MyDocument(
                        name=f"Document {i}",
                        age=20 + i,
                        status="active" if i % 2 == 0 else "inactive",
                        embeddings=np.random.rand(4).astype(np.float32),
                        sparse_embeddings={j: float(j) for j in range(i % 4)},
                    )
                    for i in range(10)
                ],
            )

        found_documents = await store.find_documents(
            document_type=MyDocument,
            filter=Field.status.eq("active"),
            limit=5,
            sort=(Field.age, SortOrder.asc)
        )
        print(found_documents)

        found_documents = await store.find_documents(
            document_type=MyDocument,
            filter=Field.status.eq("active"),
            limit=5,
            cursor=found_documents.next_cursor,
            sort=(Field.age, SortOrder.asc)
        )
        print(found_documents)

        retrieved_documents = await store.get_documents(
            document_ids=[doc._document_id_ for doc in found_documents.items][:2],
            document_type=MyDocument,
        )
        print(retrieved_documents)

        queried_documents = await store.query_documents(
            document_type=MyDocument,
            query_tensor=np.random.rand(4).astype(np.float32),
            tensor_field=Field.embeddings,
            filter=Field.age.ge(22) & Field.status.eq("active"),
            limit=5,
        )
        print(queried_documents)


if __name__ == "__main__":
    asyncio.run(main())
