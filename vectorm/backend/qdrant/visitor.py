from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import BaseModel
from qdrant_client.http.models import (
    DatetimeRange,
    FieldCondition,
    IsNullCondition,
    MatchAny,
    MatchExcept,
    MatchValue,
    Range,
)
from qdrant_client.http.models import (
    Filter as QdrantFilter,
)

from vectorm.filter import (
    BinaryExpression,
    ExpressionVisitor,
    FieldExpression,
    LiteralExpression,
    Operator,
    UnaryExpression,
    VariadicExpression,
)


def coerce_literal(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    elif isinstance(value, Enum):
        return str(value.value)
    elif isinstance(value, Decimal):
        return float(value)
    elif isinstance(value, np.generic):
        return value.item()
    elif isinstance(value, np.ndarray):
        return [coerce_literal(v) for v in value.tolist()]
    elif isinstance(value, (list, tuple, set)):
        return [coerce_literal(v) for v in value]
    elif isinstance(value, BaseModel):
        return {k: coerce_literal(v) for k, v in value.model_dump().items()}
    elif is_dataclass(value) and not isinstance(value, type):
        return {k: coerce_literal(v) for k, v in asdict(value).items()}
    elif isinstance(value, dict):
        return {k: coerce_literal(v) for k, v in value.items()}
    return value


class QdrantExpressionVisitor(ExpressionVisitor):
    def visit_literal(self, literal: LiteralExpression) -> Any:
        return coerce_literal(literal.value)

    def visit_field(self, field: FieldExpression) -> Any:
        return ".".join(field.fields)

    def visit_unary(self, unary: UnaryExpression) -> Any:
        operand = unary.operand.accept(self)

        match unary.operator:
            case Operator.not_:
                return QdrantFilter(
                    must_not=[operand],
                )
            case Operator.exists:
                return QdrantFilter(
                    must=FieldCondition(
                        key=operand,
                        is_null=IsNullCondition(is_null=False)
                    )
                )
            case _:
                raise ValueError(f"Unsupported unary operator: {unary.operator}")

    def visit_binary(self, binary: BinaryExpression) -> Any:
        left = binary.left.accept(self)
        right = binary.right.accept(self)

        match binary.operator:
            case Operator.eq:
                return QdrantFilter(
                    must=FieldCondition(
                        key=left,
                        match=MatchValue(value=right)
                    )
                )
            case Operator.ne:
                return QdrantFilter(
                    must_not=[
                        FieldCondition(
                            key=left,
                            match=MatchValue(value=right)
                        )
                    ]
                )
            case Operator.lt:
                return QdrantFilter(
                    must=FieldCondition(
                        key=left,
                        range=(
                            DatetimeRange(lt=right)
                            if isinstance(right, datetime)
                            else Range(lt=right)
                        )
                    )
                )
            case Operator.le:
                return QdrantFilter(
                    must=FieldCondition(
                        key=left,
                        range=(
                            DatetimeRange(lte=right)
                            if isinstance(right, datetime)
                            else Range(lte=right)
                        )
                    )
                )
            case Operator.gt:
                return QdrantFilter(
                    must=FieldCondition(
                        key=left,
                        range=(
                            DatetimeRange(gt=right)
                            if isinstance(right, datetime)
                            else Range(gt=right)
                        )
                    )
                )
            case Operator.ge:
                return QdrantFilter(
                    must=FieldCondition(
                        key=left,
                        range=(
                            DatetimeRange(gte=right)
                            if isinstance(right, datetime)
                            else Range(gte=right)
                        )
                    )
                )
            case Operator.in_:
                return QdrantFilter(
                    must=FieldCondition(
                        key=left,
                        match=MatchAny(any=right)
                    )
                )
            case Operator.not_in:
                return QdrantFilter(
                    must_not=[
                        FieldCondition(
                            key=left,
                            match=MatchExcept(**{"except": right})
                        )
                    ]
                )
            case Operator.and_:
                return QdrantFilter(
                    must=[left, right],
                )
            case Operator.or_:
                return QdrantFilter(
                    should=[left, right],
                )

    def visit_variadic(self, variadic: VariadicExpression) -> Any:
        operands = [operand.accept(self) for operand in variadic.operands]

        match variadic.operator:
            case Operator.and_:
                return QdrantFilter(
                    must=operands,
                )
            case Operator.or_:
                return QdrantFilter(
                    should=operands,
                )
            case Operator.not_:
                return QdrantFilter(
                    must_not=operands,
                )
            case Operator.any_:
                return QdrantFilter(
                    should=operands,
                    min_should=1,
                )
            case Operator.all_:
                return QdrantFilter(
                    must=operands,
                )

        raise ValueError(f"Unsupported variadic operator: {variadic.operator}")
