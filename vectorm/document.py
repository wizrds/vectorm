from __future__ import annotations

from typing import Any, ClassVar, Generic, Literal, TypeVar, overload
from uuid import UUID, uuid4

from pydantic import BaseModel, PrivateAttr

from vectorm.types import SparseTensor, Tensor, TensorIndexParams


TDocument = TypeVar("TDocument", bound="Document")


class Document(BaseModel):
    _document_id_: UUID = PrivateAttr(default_factory=uuid4)
    _collection_name_: ClassVar[str | None] = None

    @property
    def document_id(self) -> UUID:
        return self._document_id_

    @classmethod
    def collection_name(cls) -> str:
        if cls._collection_name_ is not None:
            return cls._collection_name_
        return cls.__name__.lower()

    @classmethod
    def get_tensor_fields(cls) -> dict[
        str,
        tuple[
            type[Tensor] | type[SparseTensor],
            TensorIndexParams | None
        ]
    ]:
        return {
            name: (
                field.annotation,
                (
                    index_params
                    if (
                        field.metadata
                        and (index_params := next(
                            iter([m for m in field.metadata if isinstance(m, TensorIndexParams)]),
                            None
                        ))
                    )
                    else None
                ),
            )
            for name, field in cls.model_fields.items()
            if (
                field.annotation
                and (
                    isinstance(field.annotation, type)
                    and issubclass(field.annotation, (Tensor, SparseTensor))
                    or (
                        hasattr(field.annotation, "__origin__")
                        and isinstance(field.annotation.__origin__, type)
                        and issubclass(field.annotation.__origin__, (Tensor, SparseTensor))
                    )
                )
            )
        }

    @classmethod
    @overload
    def get_tensor_cls(
        cls,
        field_name: str,
        /,
        default: type[Tensor] | type[SparseTensor],
    ) -> type[Tensor] | type[SparseTensor]:
        ...

    @classmethod
    @overload
    def get_tensor_cls(
        cls,
        field_name: str,
        /,
        default: None = None,
    ) -> type[Tensor] | type[SparseTensor] | None:
        ...

    @classmethod
    def get_tensor_cls(
        cls,
        field_name: str,
        /,
        default: type[Tensor] | type[SparseTensor] | None = None,
    ) -> type[Tensor] | type[SparseTensor] | None:
        tensor_field = cls.get_tensor_fields().get(field_name)
        return tensor_field[0] if tensor_field else default

    def to_dict(
        self,
        *,
        exclude_id: bool = False,
        exclude_tensors: bool = True,
        mode: Literal["json", "python"] = "json",
        **kwargs
    ) -> dict[str, Any]:
        return {
            **self.model_dump(
                mode=mode,
                exclude={
                    *(self.get_tensor_fields().keys() if exclude_tensors else set()),
                    *kwargs.pop("exclude", set()),
                },
                **kwargs
            ),
            **({"_document_id_": str(self.document_id)} if not exclude_id else {}),
        }

    @classmethod
    def from_dict(cls: type[TDocument], data: dict[str, Any], **kwargs) -> TDocument:
        document_id = data.pop("_document_id_", uuid4())
        if not isinstance(document_id, (str, bytes, int, UUID)):
            raise ValueError("Invalid document ID format")

        document = cls.model_validate(data, **kwargs)
        document._document_id_ = (
            document_id
            if isinstance(document_id, UUID)
            else UUID(
                hex=(
                    document_id
                    if isinstance(document_id, str)
                    else None
                ),
                bytes=(
                    document_id
                    if isinstance(document_id, bytes)
                    else None
                ),
                int=(
                    document_id
                    if isinstance(document_id, int)
                    else None
                ),
            )
        )
        return document


class ScoredDocument(BaseModel, Generic[TDocument]):
    document: TDocument
    score: float

    def to_dict(
        self,
        *,
        exclude_id: bool = True,
        exclude_tensors: bool = False,
        mode: Literal["json", "python"] = "json",
        **kwargs
    ) -> dict[str, Any]:
        return {
            **self.model_dump(
                mode=mode,
                exclude={"document"},
            ),
            "document": self.document.to_dict(
                exclude_id=exclude_id,
                exclude_tensors=exclude_tensors,
                mode=mode,
                **kwargs
            ),
        }
