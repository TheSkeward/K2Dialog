from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class TlkString:
    strref: int
    text: str
    sound_resref: str = ""


class TlkTable:
    def __init__(self, strings: list[TlkString]) -> None:
        self._strings = strings

    def get(self, strref: int) -> str:
        if 0 <= strref < len(self._strings):
            return self._strings[strref].text
        return ""

    def __len__(self) -> int:
        return len(self._strings)

    @classmethod
    def read(cls, path: Path) -> "TlkTable":
        data = path.read_bytes()
        if len(data) < 20 or data[:8] != b"TLK V3.0":
            raise ValueError(f"{path} is not a TLK V3.0 file")

        _language_id, count, text_offset = struct.unpack_from("<III", data, 8)
        entries_offset = 20
        entry_size = 40
        table_size = count * entry_size
        if entries_offset + table_size > len(data):
            raise ValueError(f"{path} has a truncated TLK string table")
        if text_offset > len(data):
            raise ValueError(f"{path} has an invalid TLK text offset")

        strings: list[TlkString] = []

        for strref in range(count):
            entry_at = entries_offset + strref * entry_size
            (
                _flags,
                sound_raw,
                _volume_variance,
                _pitch_variance,
                offset,
                size,
                _sound_length,
            ) = struct.unpack_from("<I16sIIIIf", data, entry_at)
            start = text_offset + offset
            end = start + size
            if start < text_offset or end < start or end > len(data):
                raise ValueError(f"{path} has an invalid TLK string range at StrRef {strref}")
            raw_text = data[start:end]
            text = _decode_text(raw_text)
            sound_resref = sound_raw.split(b"\x00", 1)[0].decode("ascii", "ignore")
            strings.append(TlkString(strref=strref, text=text, sound_resref=sound_resref))

        return cls(strings)


def find_dialog_tlk(game_dir: Path) -> Path:
    candidates = [
        game_dir / "dialog.tlk",
        game_dir / "Dialog.tlk",
        game_dir / "DIALOG.TLK",
    ]
    for path in candidates:
        if path.is_file():
            return path

    matches = list(game_dir.glob("*.tlk"))
    for path in matches:
        if path.name.lower() == "dialog.tlk":
            return path
    raise FileNotFoundError(f"Could not find dialog.tlk under {game_dir}")


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding).replace("\r\n", "\n").strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace").replace("\r\n", "\n").strip()
