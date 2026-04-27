from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import reduce
from re import Pattern
from re import compile as compile_pattern
from typing import Any, Iterable


class SortOrder(str, Enum):
    asc = "asc"
    desc = "desc"


class Operator(str, Enum):
    eq = "eq"
    ne = "ne"
    lt = "lt"
    le = "le"
    gt = "gt"
    ge = "ge"
    in_ = "in"
    not_in = "not_in"
    contains = "contains"
    matches = "matches"
    exists = "exists"
    and_ = "and"
    or_ = "or"
    not_ = "not"
    any_ = "any"
    all_ = "all"


class ExpressionVisitor(ABC):
    @abstractmethod
    def visit_literal(self, literal: LiteralExpression) -> Any:
        ...

    @abstractmethod
    def visit_field(self, field: FieldExpression) -> Any:
        ...

    @abstractmethod
    def visit_unary(self, unary: UnaryExpression) -> Any:
        ...

    @abstractmethod
    def visit_binary(self, binary: BinaryExpression) -> Any:
        ...

    @abstractmethod
    def visit_variadic(self, variadic: VariadicExpression) -> Any:
        ...

    def visit_filter(self, filter: Filter) -> Any | None:
        return filter.expression.accept(self) if filter.expression else None


class Expression(ABC):
    @abstractmethod
    def accept(self, visitor: ExpressionVisitor) -> Any:
        ...


class LiteralExpression(Expression):
    def __init__(self, value: Any) -> None:
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"<LiteralExpression {self}>"

    def accept(self, visitor: ExpressionVisitor) -> Any:
        return visitor.visit_literal(self)


class FieldExpression(Expression):
    def __init__(self, *fields: str) -> None:
        self.fields = list(fields)

    def __str__(self) -> str:
        return self.as_str()

    def __repr__(self) -> str:
        return f"<FieldExpression {self}>"

    def __getattr__(self, field: str) -> FieldExpression:
        if (
            (field.startswith("__")
            and field.endswith("__"))
            or field in self.__dict__
        ):
            return super().__getattribute__(field)

        return FieldExpression(*self.fields, field)

    def get_value(self, obj: Any) -> Any:
        for field in self.fields:
            obj = getattr(obj, field, None)
            if obj is None:
                return None
        return obj

    def as_str(self, delimiter: str = "::") -> str:
        return delimiter.join(self.fields)

    def eq(self, other: Any) -> Filter:
        return Filter(
            BinaryExpression(
                Operator.eq,
                self,
                other,
            )
        )

    def ne(self, other: Any) -> Filter:
        return Filter(
            BinaryExpression(
                Operator.ne,
                self,
                other,
            )
        )

    def lt(self, other: Any) -> Filter:
        return Filter(
            BinaryExpression(
                Operator.lt,
                self,
                other,
            )
        )

    def le(self, other: Any) -> Filter:
        return Filter(
            BinaryExpression(
                Operator.le,
                self,
                other,
            )
        )

    def gt(self, other: Any) -> Filter:
        return Filter(
            BinaryExpression(
                Operator.gt,
                self,
                other,
            )
        )

    def ge(self, other: Any) -> Filter:
        return Filter(
            BinaryExpression(
                Operator.ge,
                self,
                other,
            )
        )

    def exists(self) -> Filter:
        return Filter(
            UnaryExpression(
                Operator.exists,
                self,
            )
        )

    def matches(self, other: Pattern | str) -> Filter:
        if isinstance(other, str):
            other = compile_pattern(other)
        else:
            assert isinstance(other, Pattern)

        return Filter(
            BinaryExpression(
                Operator.matches,
                self,
                other,
            )
        )

    def is_any(self, other: Iterable[Any]) -> Filter:
        return reduce(
            lambda a, b: a.or_(b),
            [self.eq(value) for value in other],
        )

    def is_all(self, other: Iterable[Any]) -> Filter:
        return reduce(
            lambda a, b: a.and_(b),
            [self.eq(value) for value in other],
        )

    def one_of(self, other: Iterable[Any]) -> Filter:
        return Filter(
            BinaryExpression(
                Operator.in_,
                self,
                other,
            )
        )

    def none_of(self, other: Iterable[Any]) -> Filter:
        return Filter(
            BinaryExpression(
                Operator.not_in,
                self,
                other,
            )
        )

    def any_of(self, other: Iterable[Any]) -> Filter:
        return Filter(
            VariadicExpression(
                Operator.any_,
                [self] + [
                    other.expression
                    if isinstance(other, Filter) and other.expression
                    else other
                ]
            )
        )

    def accept(self, visitor: ExpressionVisitor) -> Any:
        return visitor.visit_field(self)


class UnaryExpression(Expression):
    def __init__(self, operator: Operator, operand: Any) -> None:
        self.operator = operator
        self.operand = (
            operand
            if isinstance(operand, Expression)
            else LiteralExpression(operand)
        )

    def __str__(self) -> str:
        return f"({self.operator} {self.operand})"

    def __repr__(self) -> str:
        return f"<UnaryExpression {self}>"

    def accept(self, visitor: ExpressionVisitor) -> Any:
        return visitor.visit_unary(self)


class BinaryExpression(Expression):
    def __init__(
        self,
        operator: Operator,
        left: Expression,
        right: Any,
    ) -> None:
        self.operator = operator
        self.left = left
        self.right = (
            right
            if isinstance(right, Expression)
            else LiteralExpression(right)
        )

    def __str__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"

    def __repr__(self) -> str:
        return f"<BinaryExpression {self}>"

    def accept(self, visitor: ExpressionVisitor) -> Any:
        return visitor.visit_binary(self)


class VariadicExpression(Expression):
    def __init__(self, operator: Operator, operands: list[Any]) -> None:
        self.operator = operator
        self.operands = [
            operand
            if isinstance(operand, Expression)
            else LiteralExpression(operand)
            for operand in operands
        ]

    def __str__(self) -> str:
        return f"({self.operator.value} {' '.join(map(str, self.operands))})"

    def __repr__(self) -> str:
        return f"<VariadicExpression {self}>"

    def accept(self, visitor: ExpressionVisitor) -> Any:
        return visitor.visit_variadic(self)


class Filter:
    def __init__(self, expression: Expression | None = None) -> None:
        self.expression = expression

    def __str__(self) -> str:
        return str(self.expression)

    def __repr__(self) -> str:
        return f"<Filter {self}>"

    def and_(self, other: Filter) -> Filter:
        if not self.expression or not other.expression:
            return self if self.expression else other

        return Filter(
            BinaryExpression(
                Operator.and_,
                self.expression,
                other.expression,
            )
        )

    def __and__(self, other: Filter) -> Filter:
        return self.and_(other)

    def or_(self, other: Filter) -> Filter:
        if not self.expression or not other.expression:
            return self if self.expression else other

        return Filter(
            BinaryExpression(
                Operator.or_,
                self.expression,
                other.expression,
            )
        )

    def __or__(self, other: Filter) -> Filter:
        return self.or_(other)

    def not_(self) -> Filter:
        if not self.expression:
            return self

        return Filter(
            UnaryExpression(
                Operator.not_,
                self.expression,
            )
        )

    def __invert__(self) -> Filter:
        return self.not_()

    def accept(self, visitor: ExpressionVisitor) -> Any | None:
        return visitor.visit_filter(self)


class FieldMeta(type):
    def __getattr__(cls, field: str) -> FieldExpression:
        return FieldExpression(field)

    def __getitem__(cls, field: str) -> FieldExpression:
        return FieldExpression(field)


class Field(metaclass=FieldMeta):
    @classmethod
    def from_str(cls, field: str, *, delimiter: str = "::") -> FieldExpression:
        return FieldExpression(*field.split(delimiter))
