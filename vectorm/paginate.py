from base64 import urlsafe_b64decode, urlsafe_b64encode
from datetime import datetime
from struct import pack, unpack
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel


T = TypeVar("T")


TYPE_IDENTIFIERS = {
    str: b'\x00',
    int: b'\x01',
    float: b'\x02',
    bool: b'\x03',
    type(None): b'\x04',
    datetime: b'\x05',
    UUID: b'\x06',
}

IDENTIFIER_TYPES = {
    v: k
    for k, v in TYPE_IDENTIFIERS.items()
}

ENCODE_FUNCTIONS = {
    str: lambda v: v.encode('utf-8'),
    int: lambda v: pack('>q', v),
    float: lambda v: pack('>d', v),
    bool: lambda v: pack('>?', v),
    type(None): lambda _: b'',
    datetime: lambda v: pack('>q', int(v.timestamp() * 1000)),
    UUID: lambda v: v.bytes,
}

DECODE_FUNCTIONS = {
    b'\x00': lambda b: b.decode('utf-8'),
    b'\x01': lambda b: unpack('>q', b)[0],
    b'\x02': lambda b: unpack('>d', b)[0],
    b'\x03': lambda b: unpack('>?', b)[0],
    b'\x04': lambda _: None,
    b'\x05': lambda b: datetime.fromtimestamp(unpack('>q', b)[0] / 1000),
    b'\x06': lambda b: UUID(bytes=b),
}

def encode_value(value: Any) -> bytes:
    """
    Encode a single value into a type-tagged and length-prefixed binary format.
    Format: [type_id][2-byte length][value_bytes]

    :param value: The value to encode
    :type value: Any
    :return: The encoded bytes
    :rtype: bytes
    :raises TypeError: If the value type is unsupported
    """
    value_type = type(value)

    type_id = TYPE_IDENTIFIERS.get(value_type)
    if type_id is None:
        raise TypeError(f"Unsupported type: {value_type.__name__}")

    encoded = ENCODE_FUNCTIONS[value_type](value)
    length = pack('>H', len(encoded))  # 2-byte unsigned short
    return type_id + length + encoded


def decode_value(buffer: bytes, offset: int) -> tuple[Any, int]:
    """
    Decode a single value from buffer starting at offset.
    Returns (value, new_offset)

    :param buffer: The byte buffer containing the encoded value
    :type buffer: bytes
    :param offset: The offset to start decoding from
    :type offset: int
    :return: A tuple of the decoded value and the new offset
    :rtype: tuple[Any, int]
    :raises ValueError: If the type identifier is unknown
    """
    type_id = buffer[offset:offset + 1]

    value_type = IDENTIFIER_TYPES.get(type_id)
    if value_type is None:
        raise ValueError(f"Unknown type identifier: {type_id!r}")

    length = unpack('>H', buffer[offset + 1:offset + 3])[0]
    decoded = DECODE_FUNCTIONS[type_id](
        buffer[offset + 3:offset + 3 + length]
    )
    return decoded, offset + 3 + length


def encode_cursor(*values: Any) -> bytes:
    """
    Encode a variable-length cursor with any number of values.
    Returns a URL-safe base64-encoded byte string (without padding).

    :param values: The values to encode
    :type values: Any
    :return: The encoded cursor
    :rtype: bytes
    :raises TypeError: If any value type is unsupported
    """
    encoded_parts = b''.join(encode_value(v) for v in values)
    return urlsafe_b64encode(encoded_parts).rstrip(b'=')


def decode_cursor(cursor: bytes) -> tuple[Any, ...]:
    """
    Decode a cursor encoded with `encode_cursor()`.
    Returns a tuple of decoded values.

    :param cursor: The encoded cursor
    :type cursor: bytes
    :return: The decoded values
    :rtype: tuple[Any, ...]
    :raises ValueError: If the cursor is malformed or contains unknown types
    """
    data = urlsafe_b64decode(
        cursor + b'=' * ((4 - len(cursor) % 4) % 4)
    )

    values = []
    offset = 0

    while offset < len(data):
        value, offset = decode_value(data, offset)
        values.append(value)

    return tuple(values)


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int | None = None
    next_cursor: str | None = None

