from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any


FIELD_TYPES = {
    0: "BYTE",
    1: "CHAR",
    2: "WORD",
    3: "SHORT",
    4: "DWORD",
    5: "INT",
    6: "DWORD64",
    7: "INT64",
    8: "FLOAT",
    9: "DOUBLE",
    10: "CExoString",
    11: "ResRef",
    12: "CExoLocString",
    13: "VOID",
    14: "Struct",
    15: "List",
    16: "Orientation",
    17: "Vector",
    18: "StrRef",
}

MAX_GFF_DEPTH = 128


@dataclass
class GffField:
    label: str
    type_name: str
    value: Any


@dataclass
class GffStruct:
    type_id: int
    fields: dict[str, Any]
    raw_fields: list[GffField]

    def get(self, label: str, default: Any = None) -> Any:
        return self.fields.get(label, default)


class GffReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        if len(data) < 56:
            raise ValueError("GFF file is too small")
        self.file_type = data[:4].decode("ascii", "replace").strip()
        self.version = data[4:8].decode("ascii", "replace").strip()
        if self.version != "V3.2":
            raise ValueError(f"Unsupported GFF version {self.version!r}")

        (
            self.struct_offset,
            self.struct_count,
            self.field_offset,
            self.field_count,
            self.label_offset,
            self.label_count,
            self.field_data_offset,
            self.field_data_count,
            self.field_indices_offset,
            self.field_indices_count,
            self.list_indices_offset,
            self.list_indices_count,
        ) = struct.unpack_from("<12I", data, 8)
        self._validate_sections()

    def read(self) -> GffStruct:
        if self.struct_count == 0:
            raise ValueError("GFF file has no root struct")
        return self._read_struct(0)

    def _validate_sections(self) -> None:
        self._require_range(self.struct_offset, self.struct_count * 12, "struct table")
        self._require_range(self.field_offset, self.field_count * 12, "field table")
        self._require_range(self.label_offset, self.label_count * 16, "label table")
        self._require_range(self.field_data_offset, self.field_data_count, "field data")
        self._require_range(self.field_indices_offset, self.field_indices_count, "field indices")
        self._require_range(self.list_indices_offset, self.list_indices_count, "list indices")

    def _read_struct(
        self,
        index: int,
        *,
        depth: int = 0,
        active: frozenset[int] = frozenset(),
    ) -> GffStruct:
        if depth > MAX_GFF_DEPTH:
            raise ValueError("GFF nesting is too deep")
        if not 0 <= index < self.struct_count:
            raise ValueError(f"GFF struct index out of range: {index}")
        if index in active:
            raise ValueError(f"GFF struct cycle detected at index {index}")

        at = self.struct_offset + index * 12
        type_id, data_or_offset, field_count = self._unpack_from("<III", at, "struct")
        field_indices = self._struct_field_indices(data_or_offset, field_count)
        child_active = active | {index}
        raw_fields = [
            self._read_field(field_index, depth=depth + 1, active=child_active)
            for field_index in field_indices
        ]

        fields: dict[str, Any] = {}
        for field in raw_fields:
            if field.label in fields:
                existing = fields[field.label]
                if isinstance(existing, list):
                    existing.append(field.value)
                else:
                    fields[field.label] = [existing, field.value]
            else:
                fields[field.label] = field.value
        return GffStruct(type_id=type_id, fields=fields, raw_fields=raw_fields)

    def _struct_field_indices(self, data_or_offset: int, field_count: int) -> list[int]:
        if field_count == 0:
            return []
        if field_count == 1:
            return [data_or_offset]
        start = self.field_indices_offset + data_or_offset
        self._require_range(start, field_count * 4, "struct field indices")
        return self._unpack_u32s(start, field_count, "struct field indices")

    def _read_field(
        self,
        index: int,
        *,
        depth: int,
        active: frozenset[int],
    ) -> GffField:
        if not 0 <= index < self.field_count:
            raise ValueError(f"GFF field index out of range: {index}")
        at = self.field_offset + index * 12
        field_type, label_index, data_or_offset = self._unpack_from("<III", at, "field")
        label = self._label(label_index)
        type_name = FIELD_TYPES.get(field_type, f"UNKNOWN_{field_type}")
        value = self._read_value(field_type, data_or_offset, depth=depth, active=active)
        return GffField(label=label, type_name=type_name, value=value)

    def _label(self, index: int) -> str:
        if not 0 <= index < self.label_count:
            raise ValueError(f"GFF label index out of range: {index}")
        at = self.label_offset + index * 16
        raw = self.data[at : at + 16]
        return raw.split(b"\x00", 1)[0].decode("ascii", "replace")

    def _read_value(
        self,
        field_type: int,
        data_or_offset: int,
        *,
        depth: int,
        active: frozenset[int],
    ) -> Any:
        if field_type == 0:
            return data_or_offset & 0xFF
        if field_type == 1:
            return struct.unpack("<b", bytes([data_or_offset & 0xFF]))[0]
        if field_type == 2:
            return data_or_offset & 0xFFFF
        if field_type == 3:
            return struct.unpack("<h", struct.pack("<H", data_or_offset & 0xFFFF))[0]
        if field_type == 4:
            return data_or_offset
        if field_type == 5:
            return struct.unpack("<i", struct.pack("<I", data_or_offset))[0]
        if field_type == 8:
            return struct.unpack("<f", struct.pack("<I", data_or_offset))[0]
        if field_type == 14:
            return self._read_struct(data_or_offset, depth=depth + 1, active=active)
        if field_type == 15:
            return self._read_list(data_or_offset, depth=depth + 1, active=active)

        at = self.field_data_offset + data_or_offset
        if field_type == 6:
            return self._unpack_from("<Q", at, "DWORD64 field")[0]
        if field_type == 7:
            return self._unpack_from("<q", at, "INT64 field")[0]
        if field_type == 9:
            return self._unpack_from("<d", at, "DOUBLE field")[0]
        if field_type == 10:
            size = self._unpack_from("<I", at, "CExoString size")[0]
            return _decode(self._slice(at + 4, size, "CExoString data"))
        if field_type == 11:
            self._require_range(at, 1, "ResRef size")
            size = self.data[at]
            return self._slice(at + 1, size, "ResRef data").decode("ascii", "ignore")
        if field_type == 12:
            return self._read_locstring(at)
        if field_type == 13:
            size = self._unpack_from("<I", at, "VOID size")[0]
            return self._slice(at + 4, size, "VOID data")
        if field_type == 16:
            return self._unpack_from("<4f", at, "Orientation field")
        if field_type == 17:
            return self._unpack_from("<3f", at, "Vector field")
        if field_type == 18:
            return self._unpack_from("<i", at, "StrRef field")[0]
        return data_or_offset

    def _read_list(
        self,
        data_or_offset: int,
        *,
        depth: int,
        active: frozenset[int],
    ) -> list[GffStruct]:
        at = self.list_indices_offset + data_or_offset
        count = self._unpack_from("<I", at, "list count")[0]
        self._require_range(at + 4, count * 4, "list indices")
        indices = self._unpack_u32s(at + 4, count, "list indices") if count else []
        return [self._read_struct(index, depth=depth + 1, active=active) for index in indices]

    def _read_locstring(self, at: int) -> dict[str, Any]:
        _total_size, string_ref, string_count = self._unpack_from("<IiI", at, "CExoLocString header")
        cursor = at + 12
        substrings: list[dict[str, Any]] = []
        for _ in range(string_count):
            string_id, size = self._unpack_from("<II", cursor, "CExoLocString substring header")
            cursor += 8
            text = _decode(self._slice(cursor, size, "CExoLocString substring data"))
            cursor += size
            substrings.append({"string_id": string_id, "text": text})
        return {"strref": string_ref, "substrings": substrings}

    def _unpack_from(self, fmt: str, offset: int, context: str) -> tuple:
        size = struct.calcsize(fmt)
        self._require_range(offset, size, context)
        return struct.unpack_from(fmt, self.data, offset)

    def _unpack_u32s(self, offset: int, count: int, context: str) -> list[int]:
        self._require_range(offset, count * 4, context)
        return [struct.unpack_from("<I", self.data, offset + i * 4)[0] for i in range(count)]

    def _slice(self, offset: int, size: int, context: str) -> bytes:
        self._require_range(offset, size, context)
        return self.data[offset : offset + size]

    def _require_range(self, offset: int, size: int, context: str) -> None:
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError(f"{context} points outside GFF data")


def read_gff(data: bytes) -> GffStruct:
    return GffReader(data).read()


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace").strip()
