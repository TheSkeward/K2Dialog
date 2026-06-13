from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import struct


DLG_RESOURCE_TYPE = 2029
NCS_RESOURCE_TYPE = 2010
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DialogueResource:
    source_path: Path
    archive_name: str | None
    module_name: str | None
    dlg_name: str
    data: bytes


@dataclass(frozen=True)
class ScriptResource:
    source_path: Path
    archive_name: str | None
    module_name: str | None
    script_name: str
    data: bytes


def find_dialogue_resources(game_dir: Path) -> list[DialogueResource]:
    resources: list[DialogueResource] = []
    override = game_dir / "Override"
    if override.is_dir():
        for path in sorted(override.rglob("*.dlg")):
            resources.append(
                DialogueResource(
                    source_path=path,
                    archive_name=None,
                    module_name="Override",
                    dlg_name=path.name,
                    data=path.read_bytes(),
                )
            )

    modules = game_dir / "modules"
    if modules.is_dir():
        for path in sorted(modules.iterdir()):
            if path.suffix.lower() not in {".mod", ".erf", ".rim"}:
                continue
            try:
                resources.extend(read_archive_dialogues(path))
            except Exception as exc:
                LOGGER.warning("skipped %s: %s", path, exc)

    return _dedupe_dialogue_resources(resources)


def find_script_resources(game_dir: Path) -> list[ScriptResource]:
    resources: list[ScriptResource] = []
    override = game_dir / "Override"
    if override.is_dir():
        for path in sorted(override.rglob("*.ncs")):
            resources.append(
                ScriptResource(
                    source_path=path,
                    archive_name=None,
                    module_name="Override",
                    script_name=path.name,
                    data=path.read_bytes(),
                )
            )

    modules = game_dir / "modules"
    if modules.is_dir():
        for path in sorted(modules.iterdir()):
            if path.suffix.lower() not in {".mod", ".erf", ".rim"}:
                continue
            try:
                resources.extend(read_archive_scripts(path))
            except Exception as exc:
                LOGGER.warning("skipped %s scripts: %s", path, exc)

    return _dedupe_script_resources(resources)


def read_archive_dialogues(path: Path) -> list[DialogueResource]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("archive is too small")
    kind = data[:4]
    if kind in {b"ERF ", b"MOD ", b"SAV "}:
        return _read_erf(path, data)
    if kind == b"RIM ":
        return _read_rim(path, data)
    raise ValueError("unknown archive type")


def read_archive_scripts(path: Path) -> list[ScriptResource]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("archive is too small")
    kind = data[:4]
    if kind in {b"ERF ", b"MOD ", b"SAV "}:
        return _read_erf_scripts(path, data)
    if kind == b"RIM ":
        return _read_rim_scripts(path, data)
    raise ValueError("unknown archive type")


def _read_erf(path: Path, data: bytes) -> list[DialogueResource]:
    if len(data) < 160 or data[4:8] != b"V1.0":
        raise ValueError("unsupported ERF")

    entry_count = _unpack_from("<I", data, 16, "ERF entry count")[0]
    key_offset = _unpack_from("<I", data, 24, "ERF key table offset")[0]
    resource_offset = _unpack_from("<I", data, 28, "ERF resource table offset")[0]
    _require_range(data, key_offset, entry_count * 24, "ERF key table")
    _require_range(data, resource_offset, entry_count * 8, "ERF resource table")

    keys = []
    for i in range(entry_count):
        at = key_offset + i * 24
        resref_raw, _res_id, res_type, _unused = _unpack_from("<16sIHH", data, at, "ERF key")
        resref = resref_raw.split(b"\x00", 1)[0].decode("ascii", "ignore")
        keys.append((resref, res_type))

    resources: list[DialogueResource] = []
    for i, (resref, res_type) in enumerate(keys):
        at = resource_offset + i * 8
        offset, size = _unpack_from("<II", data, at, "ERF resource entry")
        if res_type != DLG_RESOURCE_TYPE:
            continue
        _require_range(data, offset, size, f"ERF resource {resref}")
        dlg_name = f"{resref}.dlg"
        resources.append(_dialogue_resource(path, dlg_name, data[offset : offset + size]))
    return resources


def _read_erf_scripts(path: Path, data: bytes) -> list[ScriptResource]:
    if len(data) < 160 or data[4:8] != b"V1.0":
        raise ValueError("unsupported ERF")

    entry_count = _unpack_from("<I", data, 16, "ERF entry count")[0]
    key_offset = _unpack_from("<I", data, 24, "ERF key table offset")[0]
    resource_offset = _unpack_from("<I", data, 28, "ERF resource table offset")[0]
    _require_range(data, key_offset, entry_count * 24, "ERF key table")
    _require_range(data, resource_offset, entry_count * 8, "ERF resource table")

    keys = []
    for i in range(entry_count):
        at = key_offset + i * 24
        resref_raw, _res_id, res_type, _unused = _unpack_from("<16sIHH", data, at, "ERF key")
        resref = resref_raw.split(b"\x00", 1)[0].decode("ascii", "ignore")
        keys.append((resref, res_type))

    resources: list[ScriptResource] = []
    for i, (resref, res_type) in enumerate(keys):
        at = resource_offset + i * 8
        offset, size = _unpack_from("<II", data, at, "ERF resource entry")
        if res_type != NCS_RESOURCE_TYPE:
            continue
        _require_range(data, offset, size, f"ERF resource {resref}")
        resources.append(_script_resource(path, f"{resref}.ncs", data[offset : offset + size]))
    return resources


def _read_rim(path: Path, data: bytes) -> list[DialogueResource]:
    if len(data) < 12 or data[4:8] != b"V1.0":
        raise ValueError("unsupported RIM")

    entry_count, table_offset = _unpack_from("<II", data, 8, "RIM header")
    _require_range(data, table_offset, entry_count * 32, "RIM resource table")

    resources: list[DialogueResource] = []
    for i in range(entry_count):
        at = table_offset + i * 32
        resref_raw = data[at : at + 16]
        resref = resref_raw.split(b"\x00", 1)[0].decode("ascii", "ignore")
        res_type, _res_id, offset, size = _unpack_from("<IIII", data, at + 16, "RIM resource entry")
        if res_type != DLG_RESOURCE_TYPE:
            continue
        _require_range(data, offset, size, f"RIM resource {resref}")
        resources.append(_dialogue_resource(path, f"{resref}.dlg", data[offset : offset + size]))
    return resources


def _read_rim_scripts(path: Path, data: bytes) -> list[ScriptResource]:
    if len(data) < 12 or data[4:8] != b"V1.0":
        raise ValueError("unsupported RIM")

    entry_count, table_offset = _unpack_from("<II", data, 8, "RIM header")
    _require_range(data, table_offset, entry_count * 32, "RIM resource table")

    resources: list[ScriptResource] = []
    for i in range(entry_count):
        at = table_offset + i * 32
        resref_raw = data[at : at + 16]
        resref = resref_raw.split(b"\x00", 1)[0].decode("ascii", "ignore")
        res_type, _res_id, offset, size = _unpack_from("<IIII", data, at + 16, "RIM resource entry")
        if res_type != NCS_RESOURCE_TYPE:
            continue
        _require_range(data, offset, size, f"RIM resource {resref}")
        resources.append(_script_resource(path, f"{resref}.ncs", data[offset : offset + size]))
    return resources


def _dialogue_resource(path: Path, dlg_name: str, data: bytes) -> DialogueResource:
    return DialogueResource(
        source_path=path,
        archive_name=path.name,
        module_name=_module_name(path),
        dlg_name=dlg_name,
        data=data,
    )


def _script_resource(path: Path, script_name: str, data: bytes) -> ScriptResource:
    return ScriptResource(
        source_path=path,
        archive_name=path.name,
        module_name=_module_name(path),
        script_name=script_name,
        data=data,
    )


def _dedupe_dialogue_resources(resources: list[DialogueResource]) -> list[DialogueResource]:
    best: dict[tuple[str, str], DialogueResource] = {}
    order: list[tuple[str, str]] = []
    for resource in resources:
        key = ((resource.module_name or "").lower(), resource.dlg_name.lower())
        existing = best.get(key)
        if existing is None:
            best[key] = resource
            order.append(key)
            continue
        if _resource_priority(resource) > _resource_priority(existing):
            best[key] = resource
    return [best[key] for key in order]


def _dedupe_script_resources(resources: list[ScriptResource]) -> list[ScriptResource]:
    best: dict[tuple[str, str], ScriptResource] = {}
    order: list[tuple[str, str]] = []
    for resource in resources:
        key = ((resource.module_name or "").lower(), resource.script_name.lower())
        existing = best.get(key)
        if existing is None:
            best[key] = resource
            order.append(key)
            continue
        if _resource_priority(resource) > _resource_priority(existing):
            best[key] = resource
    return [best[key] for key in order]


def _resource_priority(resource: DialogueResource | ScriptResource) -> int:
    if resource.archive_name is None:
        return 40
    name = resource.archive_name.lower()
    if name.endswith(".mod"):
        return 30
    if name.endswith("_dlg.erf"):
        return 20
    if name.endswith(".erf"):
        return 15
    if name.endswith(".rim"):
        return 10
    return 0


def _module_name(path: Path) -> str:
    stem = path.stem
    lowered = stem.lower()
    for suffix in ("_dlg", "_s"):
        if lowered.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if len(stem) == 6 and stem[:3].isdigit():
        return stem.upper()
    return stem


def _unpack_from(fmt: str, data: bytes, offset: int, context: str) -> tuple:
    size = struct.calcsize(fmt)
    _require_range(data, offset, size, context)
    return struct.unpack_from(fmt, data, offset)


def _require_range(data: bytes, offset: int, size: int, context: str) -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError(f"{context} points outside archive")
