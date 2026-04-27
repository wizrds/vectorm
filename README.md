# Vectorm

![Static Badge](https://img.shields.io/badge/python-03.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square)

**Vectorm** is a Python ORM-like abstraction layer for vector databases, enabling declarative modeling, querying, filtering, and schema migrations — inspired by popular ORMs but purpose-built for vector stores like Qdrant.

---

## Features

- 🚀 Simple `Document` modeling with fields and tensor annotations
- 🔍 Rich filtering, sorting, and cursor-based pagination
- 🎯 Supports dense and sparse vector search (depending on backend)
- 🔄 Declarative migrations system
- 🧩 Pluggable backend architecture (currently supports Qdrant)
- 🧪 Typed and fully async, suitable for large-scale pipelines

---

## Installation

```bash
pip install vectorm
```

## Usage


### Basic Example

The core concept in Vectorm is the `Document`, which represents a record in your vector store. You define your document schema by subclassing `Document` and adding fields with type annotations much like any other Pydantic model. All documents have a document ID field by default, which is a UUID. It is accessible via the `document_id` property.

There are special annotations used to define tensor fields and optionally their indexing parameters. The tensor annotations support both multi-dimensional arrays as well as sparse tensors. When defining a tensor field, you can specify the dimensions and data type which will enforce array validation and sometimes reshaping. You can also specify indexing parameters like the distance metric to use for similarity search. Additionally, by defining the shape and data type, Vectorm is able to generate the appropriate JSON schema.

Most backends support 1D vectors, but some (like Qdrant) support 2D matrices as well. Other dimensions may be supported in the future, but currently will be flattened to 1D (configurable in the indexing parameters).

The following example demonstrates defining a document schema, saving documents, and performing a vector similarity query with filtering. There are two tensor fields: `embeddings`, which is a 1D dense vector with a length of 4, and configured to use cosine distance for indexing; and `sparse_embeddings`, which is a sparse vector also of length 4.

```python
import asyncio

import asyncio
import numpy as np
from typing import Annotated, Literal

from vectorm import (
    Document, Field, FieldIndexParams, FieldIndexDataType,
    Tensor, SparseTensor, TensorIndexParams, DistanceMetric,
    VectorStore, Migrator, Revision, MigrateOp, SortOrder, Dim
)
from vectorm.backend.qdrant import QdrantVectorStoreBackend


# 1. Define your document schema
class MyDocument(Document):
    _collection_name_ = "my_documents"

    name: str
    age: int
    status: Literal["active", "inactive"]
    embeddings: Annotated[Tensor[Dim[4], np.float32], TensorIndexParams(metric=DistanceMetric.cosine)]
    sparse_embeddings: SparseTensor[Dim[4], np.float32]


async def main():
    backend = QdrantVectorStoreBackend(":memory:")

    async with VectorStore(backend) as store:
        # Save documents to the store
        await store.save_documents([
            MyDocument(
                name=f"Doc {i}",
                age=20 + i,
                status="active" if i % 2 == 0 else "inactive",
                embeddings=np.random.rand(4).astype(np.float32),
                sparse_embeddings={j: float(j) for j in range(i % 4)},
            )
            for i in range(10)
        ])

        # Query documents via vector similarity with an optional filter
        results = await store.query_documents(
            document_type=MyDocument,
            query_tensor=np.random.rand(4).astype(np.float32),
            tensor_field=Field.embeddings,
            filter=Field.age.ge(22) & Field.status.eq("active"),
            limit=5,
        )
        print("Query results:", results)

        # Retrieve documents based on IDs
        results = await store.get_documents(
            document_type=MyDocument,
            document_ids=[doc.document_id for doc in results]
        )
        print("Retrieved documents:", results)

if __name__ == "__main__":
    asyncio.run(main())

```

### Filtering & Sorting

Vectorm supports rich filtering based on non-tensor fields, as well as sorting and cursor-based pagination. There are two primary methods for searching for documents:

- `find_documents`: Retrieve documents based on filters, sorting, and pagination.
- `query_documents`: Perform vector similarity search with optional filters.

Filters are defined using the `Field` class, which provides methods for common operations like equality, inequality, greater than, less than, etc. The collection of expressions built using these methods end up as a Filter, and is automatically converted to the appropriate backend format, allowing swapping out backends without changing your code.

```python
documents = await store.find_documents(
    document_type=MyDocument,
    filter=Field.status.eq("active") & Field.age.ge(25),
    limit=10,
    sort=(Field.age, SortOrder.asc),
)

documents = await store.query_documents(
    document_type=MyDocument,
    query_tensor=np.random.rand(4).astype(np.float32),
    tensor_field=Field.embeddings,
    filter=Field.age.ge(22) & Field.status.eq("active"),
    limit=5,
)

paginated_docs = await store.find_documents(
    document_type=MyDocument,
    filter=Field.status.eq("active"),
    limit=5,
    cursor=previous_page.next_cursor,
    sort=(Field.age, SortOrder.asc)
)
```

### Migrations

Vectorm includes a simple migrations system to manage schema changes over time. You define migrations as subclasses of `Revision`, specifying the operations to perform in the `upgrade` and `downgrade` methods. The `Migrator` class is used to apply these migrations to the vector store.

```python
class InitialRevision(Revision):
    revision_id = "initial"

    async def upgrade(self, op: MigrateOp) -> None:
        await op.create_collection(
            collection_name=MyDocument.collection_name(),
            document_type=MyDocument
        )

    async def downgrade(self, op: MigrateOp) -> None:
        await op.delete_collection(MyDocument.collection_name())


class AddStatusIndexRevision(Revision):
    revision_id = "add_status_index"
    previous_revision_id = "initial"

    async def upgrade(self, op: MigrateOp) -> None:
        await op.create_index(
            collection_name=MyDocument.collection_name(),
            field_name=Field.status,
            index_params=FieldIndexParams(data_type=FieldIndexDataType.keyword)
        )

    async def downgrade(self, op: MigrateOp) -> None:
        await op.delete_index(
            collection_name=MyDocument.collection_name(),
            field_name=Field.status
        )


migrator = Migrator([InitialRevision(), AddStatusIndexRevision()])

await store.upgrade_migrations(migrator)  # Upgrade to head
await store.upgrade_migrations(migrator, revision_id="add_status_index")
```

## License

This project is licensed under ISC License.

## Support & Feedback

If you encounter any issues or have feedback, please open an issue. We'd love to hear from you!

Made with ❤️ by Timothy Pogue