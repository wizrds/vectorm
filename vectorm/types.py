from __future__ import annotations

from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeGuard,
    TypeVar,
    TypeVarTuple,
    get_args,
)

import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic.dataclasses import dataclass
from pydantic_core import CoreSchema, core_schema


if TYPE_CHECKING:
    from typing import Literal as Dim
else:
    Dim = Literal


T = TypeVar("T")
Shape = TypeVarTuple("Shape")
Length = TypeVar("Length", bound=int)
DType = TypeVar("DType")


def _unwrap_literal(value: Any) -> Any:
    if (
        (origin := getattr(value, "__origin__", None))
        and (origin is Literal)
    ):
        if (args := get_args(value)):
            return (
                args[0]
                if len(args) == 1
                else tuple(args)
            )
    return value


class DistanceMetric(str, Enum):
    cosine = "cosine"
    dot = "dot"
    euclidean = "euclidean"
    l2 = "l2"
    manhattan = "manhattan"


class FieldIndexDataType(str, Enum):
    integer = "integer"
    float = "float"
    boolean = "boolean"
    datetime = "datetime"
    uuid = "uuid"
    text = "text"
    keyword = "keyword"


# NOTE: We use a dataclass here to allow usage with Annotated,
# otherwise if it were a base model, pydantic would try to validate it
# with the input of its parent model
@dataclass
class TensorIndexParams:
    metric: DistanceMetric = DistanceMetric.cosine
    flatten: bool = True
    size_override: int | None = None
    backend_params: dict[str, dict[str, Any]] | None = None


@dataclass
class FieldIndexParams:
    data_type: FieldIndexDataType
    backend_params: dict[str, dict[str, Any]] | None = None


class TensorMeta(type):
    def __getitem__(
        cls,
        item: int | tuple[Any, ...]
    ) -> type[Tensor]:
        if not isinstance(item, tuple):
            item = (item, np.float32)

        *shape_parts, last = tuple(_unwrap_literal(dim) for dim in item)

        is_any_Shape = Any in shape_parts
        shape_parts = [dim for dim in shape_parts if dim is not Ellipsis]

        dtype: np.dtype
        shape: tuple[int, ...] | None

        if (
            isinstance(last, type)
            and (
                issubclass(last, (np.generic, np.number))
                or any(last is t for t in (int, float))
            )
        ):
            dtype = np.dtype(last)
            shape = None if is_any_Shape else tuple(shape_parts)
        elif (
            isinstance(last, np.dtype)
        ):
            dtype = last
            shape = None if is_any_Shape else tuple(shape_parts)
        else:
            dtype = np.dtype(np.float32)
            shape = None if is_any_Shape else tuple(item)

        if shape is not None and not all(isinstance(dim, int) and dim > 0 for dim in shape):
            raise TypeError("Shape dimensions must be positive integers")

        return cls._create_subclass(dtype, shape)

    def _create_subclass(
        cls,
        dtype: np.dtype,
        shape: tuple[int, ...] | None = None,
    ) -> type:
        class TensorShaped(Tensor):
            __shape__ = shape
            __dtype__ = dtype

        TensorShaped.__name__ = (
            f"Tensor[{', '.join(map(str, shape)) if shape is not None else '...'}, {dtype.name}]"
        )
        TensorShaped.__qualname__ = TensorShaped.__name__
        TensorShaped.__module__ = cls.__module__
        return TensorShaped


class Tensor(Generic[*Shape, DType], metaclass=TensorMeta):
    __shape__: tuple[int, ...] | None = None
    __dtype__: np.dtype = np.dtype(np.float32)

    def __init__(self, values: Any) -> None:
        self._values = self.parse_value(
            values,
            self.__dtype__,
            self.__shape__
        )

    def __repr__(self) -> str:
        return f"<Tensor shape={self.shape or '...'}, dtype={self.dtype}>"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        def build_schema(
            shape: tuple[int, ...],
            current_schema: CoreSchema
        ) -> CoreSchema:
            if not shape:
                return current_schema

            return build_schema(
                shape[1:],
                core_schema.list_schema(
                    current_schema,
                    min_length=shape[0],
                    max_length=shape[0],
                )
            )

        return core_schema.no_info_after_validator_function(
            lambda v: cls(v),
            schema=core_schema.union_schema([
                core_schema.is_instance_schema(cls),
                core_schema.is_instance_schema(np.ndarray),
                (
                    build_schema(
                        cls.__shape__,
                        (
                            core_schema.float_schema()
                            if np.issubdtype(cls.__dtype__, np.floating)
                            else core_schema.int_schema()
                            if np.issubdtype(cls.__dtype__, np.integer)
                            else core_schema.any_schema()
                        )
                    )
                    if cls.__shape__ is not None
                    else core_schema.list_schema(
                        (
                            core_schema.float_schema()
                            if np.issubdtype(cls.__dtype__, np.floating)
                            else core_schema.int_schema()
                            if np.issubdtype(cls.__dtype__, np.integer)
                            else core_schema.any_schema()
                        )
                    )
                )
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.to_list(),
                when_used="json"
            )
        )

    @staticmethod
    def parse_value(value: Any, dtype: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
        if isinstance(value, Tensor):
            array = value.values.astype(dtype)
        elif isinstance(value, np.ndarray):
            array = value.astype(dtype)
        elif isinstance(value, (list, tuple)):
            array = np.array(value, dtype=dtype)
        else:
            raise TypeError(f"Cannot convert type {type(value)} to numpy array")

        if shape is None or array.shape == shape:
            return array

        expected_size = np.prod(shape)
        actual_size = array.size

        if expected_size != actual_size:
            raise ValueError(
                f"Cannot reshape array of size {actual_size} "
                f"to shape {shape} (size {expected_size})"
            )

        try:
            return array.reshape(shape)
        except Exception as e:
            raise ValueError(
                f"Failed to reshape array to shape {shape}: {e}"
            ) from e

    @classmethod
    def validate(cls, value: Any) -> Tensor:
        return cls(value)

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def dtype(self) -> np.dtype:
        return self._values.dtype

    def to_list(self) -> list:
        return self._values.tolist()


class SparseTensorMeta(type):
    def __getitem__(
        cls,
        item: int | tuple[Any, ...],
    ) -> type[SparseTensor]:
        if not isinstance(item, tuple):
            item = (item, np.float32)

        if len(item) != 2:
            raise TypeError("SparseTensor must be parameterized with (length, dtype)")

        length_part, dtype_spec = item
        length_part = _unwrap_literal(length_part)

        if length_part is Any:
            length = None
        elif (isinstance(length_part, int) and length_part > 0):
            length = length_part
        else:
            raise TypeError("Length must be a positive integer or Any")

        if (
            isinstance(dtype_spec, type)
            and (
                issubclass(dtype_spec, (np.generic, np.number))
                or any(dtype_spec is t for t in (int, float))
            )
        ):
            dtype = np.dtype(dtype_spec)
        elif isinstance(dtype_spec, np.dtype):
            dtype = dtype_spec
        else:
            raise TypeError("DType must be a numpy dtype or a numeric type")

        return cls._create_subclass(dtype, length)

    def _create_subclass(
        cls,
        dtype: np.dtype,
        length: int | None = None,
    ) -> type:
        class SparseTensorShaped(SparseTensor):
            __length__ = length
            __dtype__ = dtype

        SparseTensorShaped.__name__ = f"SparseTensor[{length or '...'}, {dtype.name}]"
        SparseTensorShaped.__qualname__ = SparseTensorShaped.__name__
        SparseTensorShaped.__module__ = cls.__module__
        return SparseTensorShaped


class SparseTensor(Generic[Length, DType], metaclass=SparseTensorMeta):
    __length__: int | None = None
    __dtype__: np.dtype = np.dtype(np.float32)

    def __init__(self, values: Any) -> None:
        self._indices, self._values = self.parse_value(
            values,
            self.__dtype__,
            self.__length__,
        )

    def __repr__(self) -> str:
        return f"<SparseTensor length={self.length or '...'}, dtype={self.dtype}>"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            lambda v: cls(v),
            schema=core_schema.union_schema([
                core_schema.is_instance_schema(cls),
                core_schema.tuple_schema([
                    core_schema.is_instance_schema(np.ndarray),
                    core_schema.is_instance_schema(np.ndarray),
                ]),
                core_schema.dict_schema(
                    core_schema.int_schema(
                        ge=0,
                        lt=cls.__length__ if hasattr(cls, "__length__") else None
                    ),
                    (
                        core_schema.float_schema()
                        if np.issubdtype(cls.__dtype__, np.floating)
                        else core_schema.int_schema()
                        if np.issubdtype(cls.__dtype__, np.integer)
                        else core_schema.any_schema()
                    )
                ),
                core_schema.tuple_schema([
                    core_schema.list_schema(
                        core_schema.int_schema(
                            ge=0,
                            lt=cls.__length__ if hasattr(cls, "__length__") else None
                        ),
                    ),
                    core_schema.list_schema(
                        (
                            core_schema.float_schema()
                            if np.issubdtype(cls.__dtype__, np.floating)
                            else core_schema.int_schema()
                            if np.issubdtype(cls.__dtype__, np.integer)
                            else core_schema.any_schema()
                        )
                    ),
                ]),
                core_schema.list_schema(
                    core_schema.tuple_schema([
                        core_schema.int_schema(
                            ge=0,
                            lt=cls.__length__ if hasattr(cls, "__length__") else None
                        ),
                        (
                            core_schema.float_schema()
                            if np.issubdtype(cls.__dtype__, np.floating)
                            else core_schema.int_schema()
                            if np.issubdtype(cls.__dtype__, np.integer)
                            else core_schema.any_schema()
                        ),
                    ])
                )
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.to_dict(),
                when_used="json"
            )
        )

    @staticmethod
    def parse_value(
        value: Any,
        dtype: Any,
        length: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(value, SparseTensor):
            indices, values = value.indices, value.values.astype(dtype)
        elif isinstance(value, dict):
            indices = np.array(list(value.keys()), dtype=np.int64)
            values = np.array(list(value.values()), dtype=dtype)
        elif isinstance(value, tuple) and len(value) == 2:
            indices_raw, values_raw = value
            indices = np.array(indices_raw, dtype=np.int64)
            values = np.array(values_raw, dtype=dtype)
        elif isinstance(value, (list, tuple)):
            indices_raw = []
            values_raw = []

            for item in value:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                ):
                    idx, val = item
                    indices_raw.append(int(idx))
                    values_raw.append(val)
                else:
                    raise TypeError(
                        "Each item in the list/tuple must be a (index, value) pair"
                    )

            indices = np.array(indices_raw, dtype=np.int64)
            values = np.array(values_raw, dtype=dtype)
        else:
            raise TypeError(
                "Value must be a SparseVector, dict, (indices, values) tuple, "
                "or list/tuple of (index, value) pairs"
            )

        if length is None:
            length = int(indices.max()) + 1 if indices.size > 0 else 0

        if np.any(indices < 0) or np.any(indices >= length):
            raise ValueError(f"Indices must be in the range [0, {length})")
        if indices.shape != values.shape:
            raise ValueError("Indices and values must have the same shape")

        return indices, values

    @classmethod
    def validate(cls, value: Any) -> SparseTensor:
        return cls(value)

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def length(self) -> int:
        return (
            self.__length__
            if self.__length__ is not None
            else (
                int(self._indices.max()) + 1
                if self._indices.size > 0
                else 0
            )
        )

    @property
    def dtype(self) -> np.dtype:
        return self._values.dtype

    def to_tensor(self) -> Tensor:
        dense = np.zeros(self.length, dtype=self.dtype)
        dense[self._indices] = self._values
        return Tensor(dense)

    def to_coo(self) -> tuple[np.ndarray, np.ndarray]:
        return self._indices, self._values

    def to_dict(self) -> dict[str, Any]:
        return {
            "indices": self._indices.tolist(),
            "values": self._values.tolist(),
            "length": self.length,
        }

    def to_list(self) -> list[tuple[int, Any]]:
        return [
            (int(idx), val)
            for idx, val in zip(
                self._indices.tolist(),
                self._values.tolist(),
                strict=True,
            )
        ]


def is_tensor_like(value: Any) -> TypeGuard[TensorLike]:
    if isinstance(value, Tensor):
        return True

    if isinstance(value, np.ndarray):
        return value.ndim > 0 and np.issubdtype(value.dtype, np.number)

    if isinstance(value, (list, tuple)) and value:
        def is_numeric_nested(val: Any) -> bool:
            if isinstance(val, (int, float, np.number)):
                return True
            if isinstance(val, (list, tuple)) and val:
                return all(is_numeric_nested(x) for x in val)
            return False

        return all(is_numeric_nested(x) for x in value)

    return False


def is_sparse_tensor_like(value: Any) -> TypeGuard[SparseTensorLike]:
    if isinstance(value, SparseTensor):
        return True

    if isinstance(value, dict):
        return all(
            (
                isinstance(k, (int, np.integer))
                and k >= 0
                and isinstance(v, (float, int, np.floating, np.integer))
            )
            for k, v in value.items()
        )

    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], (list, tuple, np.ndarray))
        and isinstance(value[1], (list, tuple, np.ndarray))
    ):
        indices, vals = value
        return (
            all(isinstance(k, (int, np.integer)) and k >= 0 for k in indices)
            and all(isinstance(v, (float, int, np.floating, np.integer)) for v in vals)
            and len(indices) == len(vals)
        )

    if isinstance(value, (list, tuple)):
        return all(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], (int, np.integer)) and item[0] >= 0
            and isinstance(item[1], (int, float, np.integer, np.floating))
            for item in value
        )

    return False

TensorLike: TypeAlias = (
    Tensor
    | np.ndarray
    | list[Any]
    | tuple[Any, ...]
)
SparseTensorLike: TypeAlias = (
    SparseTensor
    | dict[int, Any]
    | tuple[
        np.ndarray | list[int] | tuple[int, ...],
        np.ndarray | list[Any] | tuple[Any, ...]
      ]
    | list[tuple[int, Any]] | tuple[tuple[int, Any], ...]
)

