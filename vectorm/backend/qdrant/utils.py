from uuid import UUID

import numpy as np
from qdrant_client import models

from vectorm.document import Document, ScoredDocument, TDocument
from vectorm.filter import SortOrder
from vectorm.types import (
    DistanceMetric,
    FieldIndexDataType,
    FieldIndexParams,
    SparseTensor,
    Tensor,
    TensorIndexParams,
)


def metric_to_qdrant_distance(metric: DistanceMetric) -> models.Distance:
    match metric:
        case DistanceMetric.cosine:
            return models.Distance.COSINE
        case DistanceMetric.dot:
            return models.Distance.DOT
        case DistanceMetric.euclidean:
            return models.Distance.EUCLID
        case DistanceMetric.manhattan:
            return models.Distance.MANHATTAN
        case _:
            raise ValueError(f"Unsupported distance metric: {metric}")


def dtype_to_qdrant_datatype(dtype: np.dtype) -> models.Datatype:
    match dtype:
        case np.float32 | np.float64 | np.float128:
            return models.Datatype.FLOAT32
        case np.float16:
            return models.Datatype.FLOAT16
        case np.int8 | np.int16 | np.int32 | np.int64:
            return models.Datatype.UINT8
        case _:
            return models.Datatype.FLOAT32


def sort_order_to_qdrant_direction(order: SortOrder) -> models.Direction:
    match order:
        case SortOrder.asc:
            return models.Direction.ASC
        case SortOrder.desc:
            return models.Direction.DESC
        case _:
            raise ValueError(f"Unsupported sort order: {order}")


def vector_params_from_tensor(
    tensor_cls: type[Tensor],
    params: TensorIndexParams | None = None
) -> models.VectorParams:
    if (
        (params is not None and not params.flatten)
        and (not tensor_cls.__shape__ or len(tensor_cls.__shape__) in (1, 2))
    ):
        raise ValueError("Only 1D and 2D tensors are supported for vector fields")

    vector_size = (
        params.size_override
        if params and params.size_override
        else tensor_cls.__shape__[0]
        if tensor_cls.__shape__ and len(tensor_cls.__shape__) == 1
        else np.prod(tensor_cls.__shape__)
        if tensor_cls.__shape__
        else None
    )
    if vector_size is None:
        raise ValueError("Could not determine vector size from tensor shape")

    return models.VectorParams(
        size=vector_size,
        distance=(
            metric_to_qdrant_distance(params.metric)
            if params and params.metric
            else models.Distance.COSINE
        ),
        datatype=dtype_to_qdrant_datatype(tensor_cls.__dtype__),
        **(
            params.backend_params.get("qdrant", {}).get("vector_params", {})
            if params and params.backend_params
            else {}
        )
    )


def sparse_vector_params_from_sparse_tensor(
    tensor_cls: type[SparseTensor],
    params: TensorIndexParams | None = None,
) -> models.SparseVectorParams:
    return models.SparseVectorParams(
        index=models.SparseIndexParams(
            datatype=dtype_to_qdrant_datatype(tensor_cls.__dtype__),
            **(
                params.backend_params.get("qdrant", {}).get("sparse_index_params", {})
                if params and params.backend_params
                else {}
            )
        ),
        **(
            params.backend_params.get("qdrant", {}).get("sparse_vector_params", {})
            if params and params.backend_params
            else {}
        )
    )


def field_schema_from_field_index_params(
    params: FieldIndexParams
) -> models.PayloadFieldSchema:
    match params.data_type:
        case FieldIndexDataType.keyword:
            return models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
                **(
                    params.backend_params.get("qdrant", {}).get("field_schema_params", {})
                    if params and params.backend_params
                    else {}
                )
            )
        case FieldIndexDataType.text:
            return models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                **(
                    params.backend_params.get("qdrant", {}).get("field_schema_params", {})
                    if params and params.backend_params
                    else {}
                )
            )
        case FieldIndexDataType.integer:
            return models.IntegerIndexParams(
                type=models.IntegerIndexType.INTEGER,
                **(
                    params.backend_params.get("qdrant", {}).get("field_schema_params", {})
                    if params and params.backend_params
                    else {}
                )
            )
        case FieldIndexDataType.float:
            return models.FloatIndexParams(
                type=models.FloatIndexType.FLOAT,
                **(
                    params.backend_params.get("qdrant", {}).get("field_schema_params", {})
                    if params and params.backend_params
                    else {}
                )
            )
        case FieldIndexDataType.boolean:
            return models.BoolIndexParams(
                type=models.BoolIndexType.BOOL,
                **(
                    params.backend_params.get("qdrant", {}).get("field_schema_params", {})
                    if params and params.backend_params
                    else {}
                )
            )
        case FieldIndexDataType.datetime:
            return models.DatetimeIndexParams(
                type=models.DatetimeIndexType.DATETIME,
                **(
                    params.backend_params.get("qdrant", {}).get("field_schema_params", {})
                    if params and params.backend_params
                    else {}
                )
            )
        case FieldIndexDataType.uuid:
            return models.UuidIndexParams(
                type=models.UuidIndexType.UUID,
                **(
                    params.backend_params.get("qdrant", {}).get("field_schema_params", {})
                    if params and params.backend_params
                    else {}
                )
            )
        case _:
            raise ValueError(f"Unsupported field index type: {params.data_type}")


def normalize_tensor(
    tensor: Tensor | SparseTensor,
    params: TensorIndexParams | None = None
) -> list[float] | list[int] | models.SparseVector:
    if isinstance(tensor, Tensor):
        if params and (params.flatten and len(tensor.shape) > 1):
            return tensor.values.flatten().tolist()
        return tensor.values.tolist()
    elif isinstance(tensor, SparseTensor):
        return models.SparseVector(
            indices=tensor.indices.tolist(),
            values=tensor.values.tolist()
        )
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")


def point_from_document(document: Document) -> models.PointStruct:
    return models.PointStruct(
        id=str(document.document_id),
        vector={
            name: normalize_tensor(tensor, index_params)
            for name, (tensor_cls, index_params) in document.get_tensor_fields().items()
            if (
                (tensor := getattr(document, name, None)) is not None
                and (
                    issubclass(tensor_cls, Tensor)
                    or issubclass(tensor_cls, SparseTensor)
                )
                and (
                    len(tensor.shape) in (1, 2)
                    if issubclass(tensor_cls, Tensor)
                    else True
                )
            )
        },
        payload=document.to_dict(
            exclude_id=True,
            exclude_tensors=True,
            mode="python",
        )
    )


def document_from_point(point: models.PointStruct, document_type: type[TDocument]) -> TDocument:
    return document_type.from_dict({
        "_document_id_": (
            UUID(point.id)
            if isinstance(point.id, str)
            else UUID(int=point.id)
        ),
        **(point.payload or {}),
        **{
            name: tensor
            for name, tensor in (
                point.vector.items()
                if isinstance(point.vector, dict)
                else []
            )
            if tensor is not None
        }
    })



def document_from_record(record: models.Record, document_type: type[TDocument]) -> TDocument:
    return document_type.from_dict({
        "_document_id_": (
            UUID(record.id)
            if isinstance(record.id, str)
            else UUID(int=record.id)
        ),
        **(record.payload or {}),
        **{
            name: (
                tensor
                if not isinstance(tensor, models.SparseVector)
                else {
                    tensor.indices[i]: tensor.values[i]
                    for i in range(len(tensor.indices))
                }
            )
            for name, tensor in (
                record.vector.items()
                if isinstance(record.vector, dict)
                else []
            )
            if tensor is not None
        }
    })


def scored_document_from_scored_point(
    scored_point: models.ScoredPoint,
    document_type: type[TDocument]
) -> ScoredDocument[TDocument]:
    return ScoredDocument(
        document=(
            document_type.from_dict({
                "_document_id_": (
                    UUID(scored_point.id)
                    if isinstance(scored_point.id, str)
                    else UUID(int=scored_point.id)
                ),
                **(scored_point.payload or {}),
                **{
                    name: (
                        tensor
                        if not isinstance(tensor, models.SparseVector)
                        else {
                            tensor.indices[i]: tensor.values[i]
                            for i in range(len(tensor.indices))
                        }
                    )
                    for name, tensor in (
                        scored_point.vector.items()
                        if isinstance(scored_point.vector, dict)
                        else []
                    )
                    if tensor is not None
                }
            })
        ),
        score=scored_point.score,
    )
