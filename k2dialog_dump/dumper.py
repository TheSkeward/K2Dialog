from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil
import struct

from .archives import (
    DialogueResource,
    NameResource,
    ScriptResource,
    find_dialogue_resources,
    find_name_resources,
    find_script_resources,
)
from .gff import GffStruct, read_gff
from .tlk import TlkTable, find_dialog_tlk


@dataclass
class DumpOptions:
    game_dir: Path
    out_dir: Path
    single_file: bool = True
    by_module: bool = True
    by_dlg: bool = True
    show_unresolved_checks: bool = False


@dataclass
class DumpedDialogue:
    resource: DialogueResource
    markdown: str


@dataclass
class ParsedDialogue:
    resource: DialogueResource
    root: GffStruct


@dataclass
class StateEffectIndex:
    effects_by_state: dict[tuple[str, int], list[str]]
    dynamic_global_sets: dict[str, list[str]]
    constant_global_sets: dict[tuple[str, int], list[tuple[str, int]]]
    contextual_script_effects: dict[tuple[str, str, int], list[str]] | None = None
    cross_module_script_effects: dict[tuple[str, int], list[str]] | None = None
    current_module: str | None = None
    route_effect_cache: dict[tuple[int, int, bool, tuple[int, ...]], list[str]] | None = None
    common_reply_effect_cache: dict[tuple[int, int, tuple[int, ...]], list[str]] | None = None


@dataclass
class SpeakerNameIndex:
    names_by_module_key: dict[tuple[str, str], list[str]]
    names_by_key: dict[str, list[str]]


@dataclass(frozen=True)
class NcsInstruction:
    opcode: int
    qualifier: int
    args: tuple[object, ...]
    offset: int = 0
    end_offset: int = 0


@dataclass
class NcsStateAnalysis:
    global_reads: set[str]
    state_conditions: set[tuple[str, int]]
    dynamic_global_sets: set[str]
    constant_global_sets: dict[int, list[tuple[str, int]]]


@dataclass
class ReplyLineParts:
    text: str
    annotations: list[str]


LOGGER = logging.getLogger(__name__)
OUTPUT_MARKER = ".k2dialog_dump_output"
EFFECT_ANNOTATION_RE = re.compile(r"(.+? Influence|Light Side|Dark Side) ([+-]\d+)")
ALIGNMENT_SCRIPT_EFFECTS = {
    "a_darksml": ("Dark Side", 1),
    "a-darksml": ("Dark Side", 1),
    "a_darkmed": ("Dark Side", 2),
    "a_darkhigh": ("Dark Side", 3),
    "a_lightsml": ("Light Side", 1),
    "a_lightmed": ("Light Side", 2),
    "a_lighthigh": ("Light Side", 3),
}


def dump_game(options: DumpOptions) -> list[DumpedDialogue]:
    game_dir = options.game_dir.resolve()
    out_dir = options.out_dir.resolve()
    _validate_game_dir(game_dir)
    _validate_output_dir(out_dir, game_dir)
    tlk = TlkTable.read(find_dialog_tlk(game_dir))
    resources = find_dialogue_resources(game_dir)

    parsed: list[ParsedDialogue] = []
    for resource in resources:
        try:
            parsed.append(ParsedDialogue(resource=resource, root=read_gff(resource.data)))
        except Exception as exc:
            LOGGER.warning("failed to parse %s::%s: %s", resource.source_path, resource.dlg_name, exc)

    try:
        state_effects = _build_state_effect_index(parsed, find_script_resources(game_dir), tlk)
    except Exception as exc:
        LOGGER.warning("failed to inspect compiled scripts for deferred effects: %s", exc)
        state_effects = StateEffectIndex({}, {}, {})

    try:
        speaker_names = _build_speaker_name_index(find_name_resources(game_dir), tlk)
    except Exception as exc:
        LOGGER.warning("failed to inspect creature/placeable names: %s", exc)
        speaker_names = SpeakerNameIndex({}, {})

    dumped: list[DumpedDialogue] = []
    for item in parsed:
        try:
            markdown = render_dialogue(
                item.resource,
                item.root,
                tlk,
                state_effects=state_effects,
                speaker_names=speaker_names,
                show_unresolved_checks=options.show_unresolved_checks,
            )
            if not markdown.strip():
                continue
            dumped.append(DumpedDialogue(resource=item.resource, markdown=markdown))
        except Exception as exc:
            LOGGER.warning("failed to render %s::%s: %s", item.resource.source_path, item.resource.dlg_name, exc)

    _prepare_output_dir(out_dir)
    if options.single_file:
        _write_all_dialogue(out_dir / "all_dialogue.md", dumped)
    if options.by_module:
        _write_by_module(out_dir / "by_module", dumped)
    if options.by_dlg:
        _write_by_dlg(out_dir / "by_dlg", dumped)
    return dumped


def render_dialogue(
    resource: DialogueResource,
    root: GffStruct,
    tlk: TlkTable,
    *,
    state_effects: StateEffectIndex | None = None,
    speaker_names: SpeakerNameIndex | None = None,
    show_unresolved_checks: bool = False,
) -> str:
    state_effects = state_effects or StateEffectIndex({}, {}, {})
    state_effects.current_module = (resource.module_name or "").lower()
    state_effects.route_effect_cache = {}
    state_effects.common_reply_effect_cache = {}
    entries = _as_list(root.get("EntryList"))
    replies = _as_list(root.get("ReplyList"))
    _resolve_entry_speaker_labels(entries, resource, speaker_names)
    speaker_hint = _conversation_speaker_hint(resource, root, speaker_names)
    if _is_low_value_dialogue(resource, entries, replies, tlk):
        return ""

    lines: list[str] = []
    title = f"{resource.module_name or 'Unknown'} / {resource.dlg_name}"
    lines.append(f"# {_md_escape(title)}")
    lines.append("")
    context = _conversation_context(resource)
    if context:
        lines.append(f"Context: {_md_escape(context)}")
    lines.append("")

    skip_entries: set[int] = set()
    seen_blocks: set[str] = set()
    rendered_any = False
    for index, entry in enumerate(entries):
        if index in skip_entries:
            continue
        if _is_empty_transition_entry(entry, replies, tlk):
            continue
        forced_path = _forced_reply_transcript_path_to_choices(
            index,
            entries,
            replies,
            tlk,
            speaker_hint,
            state_effects,
        )
        if forced_path:
            block = _render_forced_reply_transcript_path(
                forced_path,
                entries,
                replies,
                tlk,
                speaker_hint,
                state_effects,
                show_unresolved_checks=show_unresolved_checks,
            )
            if _block_seen(block, seen_blocks):
                skip_entries.update(forced_path[0])
                continue
            lines.extend(block)
            skip_entries.update(forced_path[0])
            lines.append("---")
            lines.append("")
            rendered_any = True
            continue
        forced_paths = _forced_reply_transcript_paths_to_choices(
            index,
            entries,
            replies,
            tlk,
            speaker_hint,
            state_effects,
        )
        if forced_paths:
            block = _render_forced_reply_transcript_paths(
                forced_paths,
                entries,
                replies,
                tlk,
                speaker_hint,
                state_effects,
                show_unresolved_checks=show_unresolved_checks,
            )
            prefix_indices, _prefix_turns, routed_paths = forced_paths
            if _block_seen(block, seen_blocks):
                skip_entries.update(prefix_indices)
                for _label, path in routed_paths:
                    skip_entries.update(path)
                continue
            lines.extend(block)
            skip_entries.update(prefix_indices)
            for _label, path in routed_paths:
                skip_entries.update(path)
            lines.append("---")
            lines.append("")
            rendered_any = True
            continue
        if _is_forced_terminal_entry(index, entries, replies, tlk, state_effects):
            continue
        chain = _linear_continue_chain(index, entries, replies, tlk, speaker_hint)
        if len(chain) >= 2:
            last_entry = entries[chain[-1]]
            last_has_choices = _entry_has_meaningful_replies(last_entry, entries, replies, tlk)
            last_reaches_choices = _auto_route_reaches_meaningful_replies(last_entry, entries, replies, tlk)
            if not last_has_choices and not last_reaches_choices:
                skip_entries.update(chain)
                continue
            if last_has_choices:
                block = _render_transcript_chain_with_choices(
                    chain,
                    entries,
                    replies,
                    tlk,
                    speaker_hint,
                    state_effects,
                    show_unresolved_checks=show_unresolved_checks,
                )
                if _block_seen(block, seen_blocks):
                    skip_entries.update(chain)
                    continue
                lines.extend(block)
                skip_entries.update(chain)
                lines.append("---")
                lines.append("")
                rendered_any = True
                continue
            transcript_chain = chain
            if transcript_chain:
                branch_paths = _auto_transcript_paths_from(
                    chain[-1],
                    "",
                    chain[:-1],
                    set(chain[:-1]),
                    entries,
                    replies,
                    tlk,
                )
                if branch_paths:
                    block = _render_transcript_paths_with_choices(
                        branch_paths,
                        entries,
                        replies,
                        tlk,
                        speaker_hint,
                        state_effects,
                        show_unresolved_checks=show_unresolved_checks,
                    )
                    if _block_seen(block, seen_blocks):
                        skip_entries.update(transcript_chain)
                        for _label, path in branch_paths:
                            skip_entries.update(path)
                        continue
                    lines.extend(block)
                    skip_entries.update(transcript_chain)
                    for _label, path in branch_paths:
                        skip_entries.update(path)
                else:
                    skip_entries.update(transcript_chain)
                    continue
                lines.append("---")
                lines.append("")
                rendered_any = True
                continue
        if _is_orphan_entry(entry, entries, replies, tlk):
            routed_paths = _auto_transcript_paths_to_choices(index, entries, replies, tlk)
            if not routed_paths:
                continue
            if len(routed_paths) == 1:
                block = _render_transcript_chain_with_choices(
                    routed_paths[0][1],
                    entries,
                    replies,
                    tlk,
                    speaker_hint,
                    state_effects,
                    show_unresolved_checks=show_unresolved_checks,
                )
            else:
                block = _render_transcript_paths_with_choices(
                    routed_paths,
                    entries,
                    replies,
                    tlk,
                    speaker_hint,
                    state_effects,
                    show_unresolved_checks=show_unresolved_checks,
                )
            if _block_seen(block, seen_blocks):
                skip_entries.add(index)
                continue
            lines.extend(block)
            for _label, path in routed_paths:
                skip_entries.update(path)
            lines.append("---")
            lines.append("")
            rendered_any = True
            continue
        if not _entry_has_meaningful_replies(entry, entries, replies, tlk):
            continue

        block = _render_entry(
            index,
            entry,
            entries,
            replies,
            tlk,
            speaker_hint,
            state_effects,
            show_unresolved_checks=show_unresolved_checks,
        )
        if _block_seen(block, seen_blocks):
            continue
        lines.extend(block)
        lines.append("---")
        lines.append("")
        rendered_any = True

    if not rendered_any:
        return ""
    return "\n".join(lines).rstrip() + "\n"


def _linear_continue_chain(
    start_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
) -> list[int]:
    if _is_terminal_label(speaker_hint):
        return [start_index]

    chain: list[int] = []
    seen: set[int] = set()
    current = start_index

    while 0 <= current < len(entries) and current not in seen:
        seen.add(current)
        next_index = _trivial_continue_next(entries[current], replies, tlk)
        if next_index is None:
            chain.append(current)
            break
        if next_index <= current:
            chain.append(current)
            break
        chain.append(current)
        current = next_index

    return chain


def _trivial_continue_next(entry: GffStruct, replies: list[GffStruct], tlk: TlkTable) -> int | None:
    next_entries = _auto_next_entries(entry, replies, tlk)
    if len(next_entries) != 1:
        return None
    return next_entries[0]


def _auto_next_entries(entry: GffStruct, replies: list[GffStruct], tlk: TlkTable) -> list[int]:
    links = _hidden_continue_links(entry, replies, tlk)
    if not links:
        return []

    next_entries: list[int] = []
    for link in links:
        reply = _linked_reply(link, replies)
        if reply is None:
            continue
        for next_link in _as_list(reply.get("EntriesList")):
            if _link_detail_lines(next_link):
                continue
            next_index = _index_from_link(next_link)
            if next_index is not None:
                next_entries.append(next_index)
    return next_entries


def _hidden_continue_links(entry: GffStruct, replies: list[GffStruct], tlk: TlkTable) -> list[GffStruct]:
    links = _as_list(entry.get("RepliesList"))
    if not links:
        return []
    if not all(_is_trivial_continue_reply(link, replies, tlk) for link in links):
        return []
    return links


def _linked_reply(link: GffStruct, replies: list[GffStruct]) -> GffStruct | None:
    reply_index = _index_from_link(link)
    if reply_index is None or not (0 <= reply_index < len(replies)):
        return None
    return replies[reply_index]


def _auto_route_reaches_meaningful_replies(
    entry: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> bool:
    queue = _auto_next_entries(entry, replies, tlk)
    seen: set[int] = set()
    while queue:
        entry_index = queue.pop(0)
        if entry_index in seen or not (0 <= entry_index < len(entries)):
            continue
        seen.add(entry_index)
        candidate = entries[entry_index]
        if _entry_has_meaningful_replies(candidate, entries, replies, tlk):
            return True
        queue.extend(_auto_next_entries(candidate, replies, tlk))
    return False


def _auto_transcript_paths_to_choices(
    start_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> list[tuple[str, list[int]]]:
    return _auto_transcript_paths_from(start_index, "", [], set(), entries, replies, tlk)


def _auto_transcript_paths_from(
    entry_index: int,
    label: str,
    path: list[int],
    seen: set[int],
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> list[tuple[str, list[int]]]:
    if entry_index in seen or not (0 <= entry_index < len(entries)):
        return []

    next_path = path + [entry_index]
    candidate = entries[entry_index]
    if _entry_has_meaningful_replies(candidate, entries, replies, tlk):
        return [(label, next_path)]

    results: list[tuple[str, list[int]]] = []
    for child_label, child_index in _auto_next_entry_labels(candidate, replies, tlk):
        results.extend(
            _auto_transcript_paths_from(
                child_index,
                _condition_label_join(label, child_label),
                next_path,
                seen | {entry_index},
                entries,
                replies,
                tlk,
            )
        )

    deduped: list[tuple[str, list[int]]] = []
    seen_paths: set[tuple[str, tuple[int, ...]]] = set()
    for branch_label, branch_path in results:
        key = (branch_label, tuple(branch_path))
        if key in seen_paths:
            continue
        seen_paths.add(key)
        deduped.append((branch_label, branch_path))
    return deduped


def _forced_reply_transcript_path_to_choices(
    start_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
    state_effects: StateEffectIndex,
) -> tuple[list[int], list[tuple[str, str]], int] | None:
    entry_indices: list[int] = []
    turns: list[tuple[str, str]] = []
    seen_entries: set[int] = set()
    forced_reply_count = 0
    current = start_index

    while 0 <= current < len(entries) and current not in seen_entries:
        seen_entries.add(current)
        entry_indices.append(current)
        entry = entries[current]
        _append_entry_turn(turns, current, entries, replies, tlk, speaker_hint)

        hidden_next = _trivial_continue_next(entry, replies, tlk)
        if hidden_next is not None:
            current = hidden_next
            continue

        reply_lines = _reply_lines(entry, entries, replies, tlk, state_effects)
        if len(reply_lines) > 1:
            if forced_reply_count:
                return entry_indices, turns, current
            return None
        if len(reply_lines) != 1:
            return None

        links = _as_list(entry.get("RepliesList"))
        if len(links) != 1:
            return None
        link = links[0]
        reply_index = _index_from_link(link)
        if reply_index is None or not (0 <= reply_index < len(replies)):
            return None
        reply = replies[reply_index]
        reply_text, _reply_notes = _split_designer_notes(_resolve_text(reply, tlk))
        if _reply_check_lines(reply, reply_text, entries, replies, tlk, state_effects):
            return None

        next_links = _as_list(reply.get("EntriesList"))
        if len(next_links) != 1:
            return None
        next_link = next_links[0]
        if _link_detail_lines(next_link):
            return None
        next_index = _index_from_link(next_link)
        if next_index is None:
            return None

        turn_text = _reply_line_text(link, entries, replies, tlk, state_effects)
        if not turn_text:
            return None
        turns.append(("Exile", turn_text))
        forced_reply_count += 1
        current = next_index

    return None


def _forced_reply_transcript_paths_to_choices(
    start_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
    state_effects: StateEffectIndex,
) -> tuple[list[int], list[tuple[str, str]], list[tuple[str, list[int]]]] | None:
    entry_indices: list[int] = []
    turns: list[tuple[str, str]] = []
    seen_entries: set[int] = set()
    forced_reply_count = 0
    current = start_index

    while 0 <= current < len(entries) and current not in seen_entries:
        seen_entries.add(current)
        entry_indices.append(current)
        entry = entries[current]
        _append_entry_turn(turns, current, entries, replies, tlk, speaker_hint)

        hidden_next = _trivial_continue_next(entry, replies, tlk)
        if hidden_next is not None:
            current = hidden_next
            continue

        reply_lines = _reply_lines(entry, entries, replies, tlk, state_effects)
        if len(reply_lines) != 1:
            return None

        links = _as_list(entry.get("RepliesList"))
        if len(links) != 1:
            return None
        link = links[0]
        reply_index = _index_from_link(link)
        if reply_index is None or not (0 <= reply_index < len(replies)):
            return None
        reply = replies[reply_index]
        reply_text, _reply_notes = _split_designer_notes(_resolve_text(reply, tlk))
        if _reply_check_lines(reply, reply_text, entries, replies, tlk, state_effects):
            return None

        next_links = _as_list(reply.get("EntriesList"))
        if len(next_links) != 1:
            return None
        next_link = next_links[0]
        if _link_detail_lines(next_link):
            return None
        next_index = _index_from_link(next_link)
        if next_index is None:
            return None

        turn_text = _reply_line_text(link, entries, replies, tlk, state_effects)
        if not turn_text:
            return None
        turns.append(("Exile", turn_text))
        forced_reply_count += 1

        routed_paths = _auto_transcript_paths_to_choices(next_index, entries, replies, tlk)
        if routed_paths and forced_reply_count:
            return entry_indices, turns, routed_paths
        current = next_index

    return None


def _append_entry_turn(
    turns: list[tuple[str, str]],
    entry_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
) -> None:
    entry = entries[entry_index]
    text, _entry_notes = _split_designer_notes(_resolve_text(entry, tlk))
    if text:
        turns.append(
            (
                _entry_speaker(entry, speaker_hint),
                text,
            )
        )


def _auto_next_entry_labels(entry: GffStruct, replies: list[GffStruct], tlk: TlkTable) -> list[tuple[str, int]]:
    links = _hidden_continue_links(entry, replies, tlk)
    if not links:
        return []

    sibling_entry_scripts = {_plain_text(link.get("Active")).lower() for link in links}

    targets: list[tuple[str, int]] = []
    for link in links:
        reply = _linked_reply(link, replies)
        if reply is None:
            continue
        next_links = _as_list(reply.get("EntriesList"))
        sibling_scripts = {_plain_text(next_link.get("Active")).lower() for next_link in next_links}
        link_label = _auto_link_condition_label(link, sibling_entry_scripts)
        for next_link in next_links:
            if _link_detail_lines(next_link):
                continue
            next_index = _index_from_link(next_link)
            if next_index is not None:
                next_label = _auto_link_condition_label(next_link, sibling_scripts)
                targets.append((_condition_label_join(link_label, next_label), next_index))
    return targets


def _auto_link_condition_label(link: GffStruct, sibling_scripts: set[str]) -> str:
    script = _plain_text(link.get("Active")).lower()
    label = _condition_label(script)
    if label or script:
        return label
    if "c_ismale" in sibling_scripts:
        return "female Exile"
    if "c_isfemale" in sibling_scripts:
        return "male Exile"
    if "c_con_attonpm" in sibling_scripts:
        return "Atton absent"
    return ""


def _combine_variant_labels(first: str, second: str) -> str:
    return _condition_label_join(first, second, separator=" / ")


def _condition_label_join(*values: str, separator: str = ", ") -> str:
    labels: list[str] = []
    for value in values:
        for label in re.split(r"\s*(?:,|/)\s*", value):
            if label and label not in labels:
                labels.append(label)
    return separator.join(labels)


def _condition_label(script: str) -> str:
    labels = {
        "c_isfemale": "female Exile",
        "c_ismale": "male Exile",
        "c_con_attonpm": "Atton present",
    }
    return labels.get(script.lower(), "")


def _render_transcript_chain(
    chain: list[int],
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
) -> list[str]:
    lines = [_entry_chain_heading(chain), ""]

    turns: list[tuple[str, list[str]]] = []
    for index in chain:
        speaker = _entry_speaker(entries[index], speaker_hint)
        text, _entry_notes = _split_designer_notes(_resolve_text(entries[index], tlk))
        if text:
            if turns and turns[-1][0] == speaker:
                turns[-1][1].append(text)
            else:
                turns.append((speaker, [text]))

    for speaker, parts in turns:
        text = _paragraph(" ".join(parts))
        if speaker:
            lines.append(f"**{_md_escape(speaker)}:** {text}")
        else:
            lines.append(text)
    return lines


def _render_transcript_chain_with_choices(
    chain: list[int],
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
    state_effects: StateEffectIndex,
    *,
    show_unresolved_checks: bool = False,
) -> list[str]:
    lines = _render_transcript_chain(chain, entries, replies, tlk, speaker_hint)
    reply_lines = _reply_lines(
        entries[chain[-1]],
        entries,
        replies,
        tlk,
        state_effects,
        show_unresolved_checks=show_unresolved_checks,
    )
    if reply_lines:
        lines.append("")
        lines.extend(reply_lines)
    return lines


def _render_forced_reply_transcript_path(
    path: tuple[list[int], list[tuple[str, str]], int],
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
    state_effects: StateEffectIndex,
    *,
    show_unresolved_checks: bool = False,
) -> list[str]:
    entry_indices, turns, final_entry_index = path
    lines = [_entry_chain_heading(entry_indices), ""]
    for speaker, parts in _merge_turns([(speaker, [text]) for speaker, text in turns]):
        text = _paragraph(" ".join(parts))
        if speaker:
            lines.append(f"**{_md_escape(speaker)}:** {text}")
        else:
            lines.append(text)

    reply_lines = _reply_lines(
        entries[final_entry_index],
        entries,
        replies,
        tlk,
        state_effects,
        show_unresolved_checks=show_unresolved_checks,
    )
    if reply_lines:
        lines.append("")
        lines.extend(reply_lines)
    return lines


def _render_forced_reply_transcript_paths(
    path: tuple[list[int], list[tuple[str, str]], list[tuple[str, list[int]]]],
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
    state_effects: StateEffectIndex,
    *,
    show_unresolved_checks: bool = False,
) -> list[str]:
    prefix_indices, prefix_turns, routed_paths = path
    combined_paths = [(label, prefix_indices + routed_path) for label, routed_path in routed_paths]
    lines = [_entry_chain_heading(_combined_path_indices(combined_paths)), ""]
    prefix = [(speaker, [text]) for speaker, text in prefix_turns]

    rendered_variants: list[tuple[str, list[str]]] = []
    seen_variants: dict[str, int] = {}
    for label, routed_path in routed_paths:
        variant_lines: list[str] = []
        turns = prefix + _chain_turns(routed_path, entries, replies, tlk, speaker_hint)
        for speaker, parts in _merge_turns(turns):
            text = _paragraph(" ".join(parts))
            if speaker:
                variant_lines.append(f"**{_md_escape(speaker)}:** {text}")
            else:
                variant_lines.append(text)

        reply_lines = _reply_lines(
            entries[routed_path[-1]],
            entries,
            replies,
            tlk,
            state_effects,
            show_unresolved_checks=show_unresolved_checks,
        )
        if reply_lines:
            variant_lines.append("")
            variant_lines.extend(reply_lines)

        fingerprint = _block_fingerprint(variant_lines)
        if fingerprint in seen_variants:
            existing_index = seen_variants[fingerprint]
            existing_label, existing_lines = rendered_variants[existing_index]
            rendered_variants[existing_index] = (_combine_variant_labels(existing_label, label), existing_lines)
            continue
        seen_variants[fingerprint] = len(rendered_variants)
        rendered_variants.append((label, variant_lines))

    show_variant_headings = len(rendered_variants) > 1 or any(label for label, _variant_lines in rendered_variants)
    for label, variant_lines in rendered_variants:
        if show_variant_headings:
            heading = f" ({label})" if label else ""
            lines.append(f"Variant{heading}:")
        lines.extend(variant_lines)
        lines.append("")
    return lines


def _render_transcript_paths_with_choices(
    paths: list[tuple[str, list[int]]],
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
    state_effects: StateEffectIndex,
    *,
    show_unresolved_checks: bool = False,
) -> list[str]:
    heading_indices = _combined_path_indices(paths)
    common_path = _common_path_prefix([path for _label, path in paths])
    common_turns = _chain_turns(common_path, entries, replies, tlk, speaker_hint)
    lines = [_entry_chain_heading(heading_indices), ""]

    rendered_variants: list[tuple[str, list[str]]] = []
    seen_variants: dict[str, int] = {}
    for label, path in paths:
        variant_path = path[len(common_path) :]
        variant_lines: list[str] = []
        for speaker, parts in _merge_turns(common_turns + _chain_turns(variant_path, entries, replies, tlk, speaker_hint)):
            text = _paragraph(" ".join(parts))
            if speaker:
                variant_lines.append(f"**{_md_escape(speaker)}:** {text}")
            else:
                variant_lines.append(text)

        reply_lines = _reply_lines(
            entries[path[-1]],
            entries,
            replies,
            tlk,
            state_effects,
            show_unresolved_checks=show_unresolved_checks,
        )
        if reply_lines:
            variant_lines.append("")
            variant_lines.extend(reply_lines)

        fingerprint = _block_fingerprint(variant_lines)
        if fingerprint in seen_variants:
            existing_index = seen_variants[fingerprint]
            existing_label, existing_lines = rendered_variants[existing_index]
            rendered_variants[existing_index] = (_combine_variant_labels(existing_label, label), existing_lines)
            continue
        seen_variants[fingerprint] = len(rendered_variants)
        rendered_variants.append((label, variant_lines))

    show_variant_headings = len(rendered_variants) > 1 or any(label for label, _variant_lines in rendered_variants)
    for label, variant_lines in rendered_variants:
        if show_variant_headings:
            heading = f" ({label})" if label else ""
            lines.append(f"Variant{heading}:")
        lines.extend(variant_lines)
        lines.append("")
    return lines


def _entry_chain_heading(indices: list[int]) -> str:
    if not indices:
        return "## Entries"
    if len(indices) == 1:
        return f"## Entry {indices[0]}"

    ordered_unique = list(dict.fromkeys(indices))
    sorted_unique = sorted(ordered_unique)
    is_contiguous = sorted_unique == list(range(sorted_unique[0], sorted_unique[-1] + 1))
    if is_contiguous and ordered_unique == sorted_unique:
        return f"## Entries {sorted_unique[0]}-{sorted_unique[-1]}"

    return f"## Entries {_compact_entry_path(ordered_unique)}"


def _combined_path_indices(paths: list[tuple[str, list[int]]]) -> list[int]:
    indices: list[int] = []
    for _label, path in paths:
        for index in path:
            if index not in indices:
                indices.append(index)
    return indices


def _common_path_prefix(paths: list[list[int]]) -> list[int]:
    if not paths:
        return []

    prefix = list(paths[0])
    for path in paths[1:]:
        shared_length = 0
        for left, right in zip(prefix, path):
            if left != right:
                break
            shared_length += 1
        prefix = prefix[:shared_length]
        if not prefix:
            break
    return prefix


def _compact_entry_path(indices: list[int]) -> str:
    parts: list[str] = []
    run_start = indices[0]
    previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        parts.append(_entry_path_part(run_start, previous))
        run_start = previous = index
    parts.append(_entry_path_part(run_start, previous))
    return " -> ".join(parts)


def _entry_path_part(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def _chain_turns(
    chain: list[int],
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
) -> list[tuple[str, list[str]]]:
    turns: list[tuple[str, list[str]]] = []
    for index in chain:
        speaker = _entry_speaker(entries[index], speaker_hint)
        text, _entry_notes = _split_designer_notes(_resolve_text(entries[index], tlk))
        if not text:
            continue
        turns.append((speaker, [text]))
    return _merge_turns(turns)


def _block_seen(block: list[str], seen_blocks: set[str]) -> bool:
    fingerprint = _block_fingerprint(block)
    if fingerprint in seen_blocks:
        return True
    seen_blocks.add(fingerprint)
    return False


def _block_fingerprint(block: list[str]) -> str:
    content_lines = [
        line
        for line in block
        if line.strip()
        and not line.startswith("## ")
        and not line.startswith("Variant")
    ]
    text = "\n".join(content_lines).lower()
    replacements = {
        "i'll": "i will",
        "you'll": "you will",
        "we'll": "we will",
        "they'll": "they will",
        "it's": "it is",
        "that's": "that is",
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    text = re.sub(r"[`*_]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _merge_turns(turns: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    merged: list[tuple[str, list[str]]] = []
    for speaker, parts in turns:
        if merged and merged[-1][0] == speaker:
            merged[-1][1].extend(parts)
        else:
            merged.append((speaker, list(parts)))
    return merged


def _render_entry(
    index: int,
    entry: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
    state_effects: StateEffectIndex,
    *,
    show_unresolved_checks: bool = False,
) -> list[str]:
    lines = [f"## Entry {index}", ""]
    speaker = _entry_speaker(entry, speaker_hint)
    text, notes = _split_designer_notes(_resolve_text(entry, tlk))
    if text:
        line_text = _display_text(text, speaker)
        if speaker:
            lines.extend(_speaker_text_lines(speaker, line_text))
        else:
            lines.append(line_text)

    reply_lines = _reply_lines(
        entry,
        entries,
        replies,
        tlk,
        state_effects,
        show_unresolved_checks=show_unresolved_checks,
    )
    if reply_lines:
        lines.append("")
        lines.extend(reply_lines)
    return lines


def _reply_lines(
    entry: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
    *,
    show_unresolved_checks: bool = False,
) -> list[str]:
    linked_replies = _as_list(entry.get("RepliesList"))
    if not linked_replies:
        return []
    linked_replies = [link for link in linked_replies if not _is_trivial_continue_reply(link, replies, tlk)]
    if not linked_replies:
        return []

    parts: list[ReplyLineParts] = []
    for link in linked_replies:
        line_parts = _reply_line_parts(
            link,
            entries,
            replies,
            tlk,
            state_effects,
            show_unresolved_checks=show_unresolved_checks,
        )
        if line_parts is not None:
            parts.append(line_parts)

    common_effects = _common_effect_annotations(parts) if len(parts) >= 2 else []
    return [f"- {_format_reply_line(part, common_effects)}" for part in parts]


def _reply_line_text(
    link: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
    *,
    show_unresolved_checks: bool = False,
) -> str:
    parts = _reply_line_parts(
        link,
        entries,
        replies,
        tlk,
        state_effects,
        show_unresolved_checks=show_unresolved_checks,
    )
    if parts is None:
        return ""
    return _format_reply_line(parts)


def _reply_line_parts(
    link: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
    *,
    show_unresolved_checks: bool = False,
    include_common_reply_effects: bool = True,
    include_outcome_annotations: bool = True,
    common_effect_seen: set[int] | None = None,
) -> ReplyLineParts | None:
    state_effects = state_effects or StateEffectIndex({}, {}, {})
    reply_index = _index_from_link(link)
    if reply_index is None:
        return None

    reply_text = ""
    reply: GffStruct | None = None
    if 0 <= reply_index < len(replies):
        reply = replies[reply_index]
        reply_text, _reply_notes = _split_designer_notes(_resolve_text(reply, tlk))
    force_tag = _force_persuade_choice_tag(link, reply)
    if not force_tag and _leading_tag(reply_text).lower() == "force persuade":
        force_tag = "Affect Mind"
    choice_text = _add_force_persuade_tag(reply_text, force_tag) if reply_text else "[continue]"
    annotations: list[str] = []
    annotations.extend(_link_detail_lines(link))
    visibility_lines = _visibility_check_lines(link, reply_text)
    prefix_tags = _visibility_prefix_tags(visibility_lines, choice_text)
    check_lines: list[str] = []
    if reply is not None:
        annotations.extend(_effect_lines(reply))
        if include_outcome_annotations:
            check_lines = _reply_check_lines(reply, reply_text, entries, replies, tlk, state_effects)
            prefix_tags.extend(_check_prefix_tags(check_lines, choice_text))
            annotations.extend(line for line in check_lines if not _check_prefix_tags_for_line(line, choice_text))
            if not check_lines:
                routed_check_lines = _routed_entry_check_lines(reply, entries, replies, tlk, state_effects)
                if routed_check_lines:
                    prefix_tags.extend(_check_prefix_tags(routed_check_lines, choice_text))
                    annotations.extend(
                        line for line in routed_check_lines if not _check_prefix_tags_for_line(line, choice_text)
                    )
                else:
                    annotations.extend(
                        _routed_entry_effect_lines(
                            reply,
                            entries,
                            replies,
                            tlk,
                            state_effects,
                            include_common_reply_effects=include_common_reply_effects,
                            common_effect_seen=common_effect_seen,
                        )
                    )
    elif show_unresolved_checks and _tag_without_check_line(reply_text, link, reply):
        annotations.append(_tag_without_check_line(reply_text, link, reply))
    if show_unresolved_checks and reply is not None and not check_lines and not visibility_lines:
        tag_line = _tag_without_check_line(reply_text, link, reply)
        if tag_line:
            annotations.append(tag_line)
    return ReplyLineParts(_choice_text(choice_text, prefix_tags), _merged_annotations(annotations))


def _format_reply_line(parts: ReplyLineParts, common_effects: list[str] | None = None) -> str:
    common_totals = _effect_totals(common_effects or [])
    annotations: list[str] = []
    for annotation in parts.annotations:
        parsed = _parse_effect_annotation(annotation)
        if parsed is None:
            annotations.append(annotation)
            continue

        label, amount = parsed
        common = common_totals.get(label)
        if common is None:
            annotations.append(annotation)
            continue

        remaining = amount - common
        if remaining:
            annotations.append(f"{label} {remaining:+d}")

    detail = f" [{'; '.join(annotations)}]" if annotations else ""
    return f"{parts.text}{detail}"


def _common_effect_annotations(parts: list[ReplyLineParts]) -> list[str]:
    if not parts:
        return []
    return _common_effect_annotations_from_lists([part.annotations for part in parts])


def _common_effect_annotations_from_lists(annotation_lists: list[list[str]]) -> list[str]:
    if not annotation_lists:
        return []

    effect_maps = [_effect_totals(annotations) for annotations in annotation_lists]
    if not effect_maps or any(not effect_map for effect_map in effect_maps):
        return []

    common_labels = set.intersection(*(set(effect_map) for effect_map in effect_maps))
    if not common_labels:
        return []

    common_totals: dict[str, int] = {}
    for label in common_labels:
        amounts = [effect_map[label] for effect_map in effect_maps]
        if all(amount > 0 for amount in amounts):
            common_totals[label] = min(amounts)
        elif all(amount < 0 for amount in amounts):
            common_totals[label] = -min(abs(amount) for amount in amounts)

    if not common_totals:
        return []

    ordered: list[str] = []
    for annotation in annotation_lists[0]:
        parsed = _parse_effect_annotation(annotation)
        if parsed is None:
            continue
        label, _amount = parsed
        if label in common_totals and label not in ordered:
            ordered.append(label)
    return [f"{label} {common_totals[label]:+d}" for label in ordered]


def _effect_totals(annotations: list[str]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for annotation in annotations:
        parsed = _parse_effect_annotation(annotation)
        if parsed is None:
            continue
        label, amount = parsed
        totals[label] = totals.get(label, 0) + amount
    return {label: amount for label, amount in totals.items() if amount}


def _is_effect_annotation(annotation: str) -> bool:
    return EFFECT_ANNOTATION_RE.fullmatch(annotation) is not None


def _parse_effect_annotation(annotation: str) -> tuple[str, int] | None:
    effect = EFFECT_ANNOTATION_RE.fullmatch(annotation)
    if not effect:
        return None
    return effect.group(1), int(effect.group(2))


def _merged_annotations(annotations: list[str]) -> list[str]:
    merged: list[str] = []
    effect_positions: dict[str, int] = {}
    effect_totals: dict[str, int] = {}

    for annotation in annotations:
        effect = EFFECT_ANNOTATION_RE.fullmatch(annotation)
        if effect:
            label = effect.group(1)
            amount = int(effect.group(2))
            if label not in effect_positions:
                effect_positions[label] = len(merged)
                merged.append("")
                effect_totals[label] = 0
            effect_totals[label] += amount
            continue

        if annotation not in merged:
            merged.append(annotation)

    for label, position in effect_positions.items():
        amount = effect_totals[label]
        if amount:
            merged[position] = f"{label} {amount:+d}"

    return [annotation for annotation in merged if annotation]


def _routed_entry_check_lines(
    reply: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
) -> list[str]:
    state_effects = state_effects or StateEffectIndex({}, {}, {})
    lines: list[str] = []
    for entry_link in _as_list(reply.get("EntriesList")):
        entry_index = _index_from_link(entry_link)
        if entry_index is None or not (0 <= entry_index < len(entries)):
            continue
        lines.extend(_automatic_route_check_lines(entry_index, entries, replies, tlk, set(), state_effects))
    return list(dict.fromkeys(lines))


def _automatic_route_check_lines(
    entry_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    seen: set[int],
    state_effects: StateEffectIndex,
) -> list[str]:
    if entry_index in seen or not (0 <= entry_index < len(entries)):
        return []

    seen.add(entry_index)
    entry = entries[entry_index]
    entry_effects = _effect_lines(entry) + _state_transition_effect_lines(entry, state_effects)
    lines: list[str] = []

    for link in _hidden_continue_links(entry, replies, tlk):
        reply = _linked_reply(link, replies)
        if reply is None:
            continue

        reply_text, _notes = _split_designer_notes(_resolve_text(reply, tlk))
        prefix = entry_effects + _effect_lines(reply) + _state_transition_effect_lines(reply, state_effects)
        check_lines = _reply_check_lines(reply, reply_text, entries, replies, tlk, state_effects)
        if check_lines:
            lines.extend(prefix + check_lines)
            continue

        for next_link in _as_list(reply.get("EntriesList")):
            if _link_detail_lines(next_link):
                continue
            next_index = _index_from_link(next_link)
            if next_index is None:
                continue
            child_lines = _automatic_route_check_lines(next_index, entries, replies, tlk, set(seen), state_effects)
            if child_lines:
                lines.extend(prefix + child_lines)

    return list(dict.fromkeys(lines))


def _routed_entry_effect_lines(
    reply: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
    *,
    include_common_reply_effects: bool = True,
    common_effect_seen: set[int] | None = None,
) -> list[str]:
    state_effects = state_effects or StateEffectIndex({}, {}, {})
    branch_effects: list[list[str]] = []
    for entry_link in _as_list(reply.get("EntriesList")):
        entry_index = _index_from_link(entry_link)
        if entry_index is None or not (0 <= entry_index < len(entries)):
            continue
        branch_effects.append(
            _automatic_route_effect_lines(
                entry_index,
                entries,
                replies,
                tlk,
                state_effects,
                include_common_reply_effects=include_common_reply_effects,
                common_effect_seen=common_effect_seen,
            )
        )
    if not branch_effects:
        return []
    if len(branch_effects) == 1:
        return branch_effects[0]
    return _common_annotations(branch_effects)


def _automatic_route_effect_lines(
    start_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
    *,
    include_common_reply_effects: bool = True,
    common_effect_seen: set[int] | None = None,
) -> list[str]:
    state_effects = state_effects or StateEffectIndex({}, {}, {})
    seen_key = tuple(sorted(common_effect_seen or set()))
    cache_key = (id(entries), start_index, include_common_reply_effects, seen_key)
    cache = state_effects.route_effect_cache
    if cache is not None and cache_key in cache:
        return list(cache[cache_key])

    result = _automatic_route_effect_path(
        start_index,
        entries,
        replies,
        tlk,
        set(),
        state_effects,
        [],
        include_common_reply_effects=include_common_reply_effects,
        common_effect_seen=common_effect_seen or set(),
    )
    if cache is not None:
        cache[cache_key] = list(result)
    return result


def _automatic_route_effect_path(
    entry_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    seen: set[int],
    state_effects: StateEffectIndex,
    path_effects: list[str],
    *,
    include_common_reply_effects: bool = True,
    common_effect_seen: set[int],
) -> list[str]:
    if entry_index in seen or not (0 <= entry_index < len(entries)):
        return []

    seen.add(entry_index)
    entry = entries[entry_index]
    entry_effects: list[str] = []
    entry_effects.extend(_effect_lines(entry))
    entry_effects.extend(_state_transition_effect_lines(entry, state_effects))
    entry_effects.extend(_cross_module_transition_effect_lines(entry, state_effects, path_effects + entry_effects))

    hidden_links = _hidden_continue_links(entry, replies, tlk)
    if not hidden_links:
        effects = list(entry_effects)
        if include_common_reply_effects and entry_index not in common_effect_seen:
            effects.extend(
                _common_reply_effects(
                    entry,
                    entries,
                    replies,
                    tlk,
                    state_effects,
                    common_effect_seen | {entry_index},
                )
            )
        return _merged_annotations(effects)

    branches: list[list[str]] = []
    for link in hidden_links:
        reply = _linked_reply(link, replies)
        if reply is None:
            continue
        reply_effects = list(entry_effects)
        reply_effects.extend(_effect_lines(reply))
        reply_effects.extend(_state_transition_effect_lines(reply, state_effects))
        reply_effects.extend(
            _cross_module_transition_effect_lines(reply, state_effects, path_effects + reply_effects)
        )

        next_links = [
            next_link
            for next_link in _as_list(reply.get("EntriesList"))
            if not _link_detail_lines(next_link)
        ]
        if not next_links:
            branches.append(_merged_annotations(reply_effects))
            continue

        for next_link in next_links:
            if _link_detail_lines(next_link):
                continue
            next_index = _index_from_link(next_link)
            if next_index is not None:
                child_effects = _automatic_route_effect_path(
                    next_index,
                    entries,
                    replies,
                    tlk,
                    set(seen),
                    state_effects,
                    path_effects + reply_effects,
                    include_common_reply_effects=include_common_reply_effects,
                    common_effect_seen=common_effect_seen,
                )
                branches.append(_merged_annotations(reply_effects + child_effects))

    if not branches:
        return _merged_annotations(entry_effects)
    if len(branches) == 1:
        return branches[0]
    return _common_annotations(branches)


def _common_annotations(branches: list[list[str]]) -> list[str]:
    if not branches:
        return []
    common_effects = _common_effect_annotations_from_lists(branches)
    non_effect_sets = [
        {annotation for annotation in branch if not _is_effect_annotation(annotation)}
        for branch in branches
    ]
    common_non_effects: set[str] = set()
    if non_effect_sets and all(non_effect_sets):
        common_non_effects = set.intersection(*non_effect_sets)

    common: list[str] = []
    for annotation in branches[0]:
        if _is_effect_annotation(annotation):
            parsed = _parse_effect_annotation(annotation)
            if parsed is None:
                continue
            label, _amount = parsed
            replacement = next((effect for effect in common_effects if effect.startswith(f"{label} ")), None)
            if replacement and replacement not in common:
                common.append(replacement)
        elif annotation in common_non_effects:
            common.append(annotation)
    return common


def _common_reply_effects(
    entry: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex,
    common_effect_seen: set[int] | None = None,
) -> list[str]:
    common_effect_seen = common_effect_seen or set()
    seen_key = tuple(sorted(common_effect_seen))
    cache_key = (id(entries), id(entry), seen_key)
    cache = state_effects.common_reply_effect_cache
    if cache is not None and cache_key in cache:
        return list(cache[cache_key])

    linked_replies = _as_list(entry.get("RepliesList"))
    if len(linked_replies) < 2:
        return []

    parts: list[ReplyLineParts] = []
    visible_links: list[GffStruct] = []
    for link in linked_replies:
        if _is_trivial_continue_reply(link, replies, tlk):
            continue
        reply = _linked_reply(link, replies)
        if reply is None:
            continue
        annotations = _effect_lines(reply)
        annotations.extend(
            _routed_entry_effect_lines(
                reply,
                entries,
                replies,
                tlk,
                state_effects,
                include_common_reply_effects=False,
                common_effect_seen=common_effect_seen,
            )
        )
        parts.append(ReplyLineParts("", _merged_annotations(annotations)))
        visible_links.append(link)

    if len(parts) < 2:
        return []

    common = _common_effect_annotations(parts)
    next_indices = [
        _single_unconditional_reply_next_entry(link, replies)
        for link in visible_links
    ]
    same_next = next_indices and all(index is not None and index == next_indices[0] for index in next_indices)
    if same_next and next_indices[0] not in common_effect_seen:
        nested = _common_reply_effects(
            entries[next_indices[0]],
            entries,
            replies,
            tlk,
            state_effects,
            common_effect_seen | {next_indices[0]},
        )
        common = _merged_annotations(common + nested)
    if cache is not None:
        cache[cache_key] = list(common)
    return common


def _single_unconditional_reply_next_entry(link: GffStruct, replies: list[GffStruct]) -> int | None:
    reply = _linked_reply(link, replies)
    if reply is None:
        return None
    next_links = _as_list(reply.get("EntriesList"))
    if len(next_links) != 1:
        return None
    next_link = next_links[0]
    if _link_detail_lines(next_link) or _visibility_check_lines(next_link, ""):
        return None
    return _index_from_link(next_link)


def _resolve_text(node: GffStruct, tlk: TlkTable) -> str:
    value = node.get("Text")
    if isinstance(value, dict):
        strref = value.get("strref", -1)
        if isinstance(strref, int) and strref >= 0:
            resolved = tlk.get(strref)
            if resolved:
                return resolved
        substrings = value.get("substrings") or []
        for substring in substrings:
            text = substring.get("text", "")
            if text:
                return text
    if isinstance(value, int):
        return tlk.get(value)
    if isinstance(value, str):
        return value
    return ""


def _link_detail_lines(link: GffStruct) -> list[str]:
    details = []
    for label, value in _script_fields(link):
        suffix = "b" if label.lower().endswith("2") else ""
        if _skill_check_from_script(value, link, param_suffix=suffix) is not None:
            continue
        normalized = label.lower()
        if "active" in normalized or "condition" in normalized:
            continue
        kind = "action"
        details.append(f"{kind}: `{value}`")
    if _plain_text(link.get("Comment")):
        details.append(f"comment: {_plain_text(link.get('Comment'))}")
    if _plain_text(link.get("LinkComment")):
        details.append(f"comment: {_plain_text(link.get('LinkComment'))}")
    return details


def _reply_check_lines(
    reply: GffStruct,
    reply_text: str,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
) -> list[str]:
    state_effects = state_effects or StateEffectIndex({}, {}, {})
    checks: list[tuple[dict[str, object], int]] = []
    fallback_entries: list[int] = []
    for link in _as_list(reply.get("EntriesList")):
        entry_index = _index_from_link(link)
        has_condition = False
        for field, suffix in (("Active", ""), ("Active2", "b")):
            active = _plain_text(link.get(field))
            if active:
                has_condition = True
            check = _skill_check_details(active, link, reply_text, param_suffix=suffix)
            if check is not None and entry_index is not None:
                checks.append((check, entry_index))
        if entry_index is not None and not has_condition:
            fallback_entries.append(entry_index)

    lines: list[str] = []
    if not checks:
        return lines

    gt_checks = [item for item in checks if item[0]["op"] == "gt"]
    lt_checks = [item for item in checks if item[0]["op"] == "lt"]
    attr_gt_checks = [item for item in checks if item[0]["op"] == "attr_gt"]
    other_checks = [item for item in checks if item[0]["op"] not in {"gt", "lt", "attr_gt"}]

    success_checks = gt_checks + attr_gt_checks
    if len(success_checks) > 1:
        grouped_success_checks: list[tuple[dict[str, object], list[int]]] = []
        grouped_by_condition: dict[str, int] = {}
        for check, entry_index in sorted(success_checks, key=lambda item: int(item[0]["dc"]), reverse=True):
            condition = _outcome_check_condition(check, reply_text)
            if condition in grouped_by_condition:
                grouped_success_checks[grouped_by_condition[condition]][1].append(entry_index)
            else:
                grouped_by_condition[condition] = len(grouped_success_checks)
                grouped_success_checks.append((check, [entry_index]))
        for check, entry_indices in grouped_success_checks:
            condition = _outcome_check_condition(check, reply_text)
            lines.append(f"{condition}: {_common_entry_outcome(entry_indices, entries, replies, tlk, state_effects)}")
        if fallback_entries:
            lines.append(f"otherwise: {_entry_outcomes(fallback_entries, entries, replies, tlk, state_effects)}")
        for check, entry_index in lt_checks:
            lines.append(f"DC {check['dc']}")
            lines.append(f"failure: {_entry_outcome(entry_index, entries, replies, tlk, state_effects)}")
        for check, entry_index in other_checks:
            lines.append(_skill_check_label(check))
            lines.append(f"success: {_entry_outcome(entry_index, entries, replies, tlk, state_effects)}")
        return lines

    for check, entry_index in gt_checks:
        lines.append(_outcome_check_condition(check, reply_text))
        lines.append(f"success: {_entry_outcome(entry_index, entries, replies, tlk, state_effects)}")
        if fallback_entries:
            lines.append(f"failure: {_entry_outcomes(fallback_entries, entries, replies, tlk, state_effects)}")

    for check, entry_index in lt_checks:
        lines.append(_outcome_check_condition(check, reply_text))
        if fallback_entries:
            lines.append(f"success: {_entry_outcomes(fallback_entries, entries, replies, tlk, state_effects)}")
        lines.append(f"failure: {_entry_outcome(entry_index, entries, replies, tlk, state_effects)}")

    for check, entry_index in other_checks:
        lines.append(_skill_check_label(check))
        lines.append(f"success: {_entry_outcome(entry_index, entries, replies, tlk, state_effects)}")
        if fallback_entries:
            lines.append(f"failure: {_entry_outcomes(fallback_entries, entries, replies, tlk, state_effects)}")
    return lines


def _outcome_check_condition(check: dict[str, object], reply_text: str) -> str:
    skill = str(check["skill"])
    dc = check.get("dc", "")
    prefix = "below DC" if check.get("op") == "lt" else "DC"
    if _choice_tag_matches_check(reply_text, skill):
        return f"{prefix} {dc}"
    return f"{skill} {prefix} {dc}"


def _visibility_check_lines(link: GffStruct, reply_text: str) -> list[str]:
    lines: list[str] = []
    for field in ("Active", "Active2"):
        active = _plain_text(link.get(field))
        if _force_persuade_level(active):
            continue
        suffix = "b" if field == "Active2" else ""
        check = _skill_check_from_script(active, link, reply_text, param_suffix=suffix)
        if check is not None:
            lines.append(check)
    return lines


def _check_prefix_tags(check_lines: list[str], choice_text: str) -> list[str]:
    tags: list[str] = []
    for line in check_lines:
        for tag in _check_prefix_tags_for_line(line, choice_text):
            if tag and tag not in tags:
                tags.append(tag)
    return tags


def _visibility_prefix_tags(check_lines: list[str], choice_text: str) -> list[str]:
    tags: list[str] = []
    for line in check_lines:
        tag = _visibility_prefix_tag_for_line(line, choice_text)
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def _visibility_prefix_tag_for_line(check_line: str, choice_text: str) -> str:
    line = check_line.removeprefix("Requires ").strip()
    match = re.fullmatch(r"(?P<label>.+?) below (?P<dc>\d+)", line)
    if match and _choice_tag_matches_check(choice_text, match.group("label")):
        return f"below DC {match.group('dc')}"

    match = re.fullmatch(r"(?P<label>.+?) (?P<dc>\d+)(?: \+ (?P<item>.+))?", line)
    if match and _choice_tag_matches_check(choice_text, match.group("label")):
        return f"DC {match.group('dc')}"

    return ""


def _check_prefix_tags_for_line(check_line: str, choice_text: str) -> list[str]:
    line = check_line.removeprefix("Requires ").strip()
    if line.startswith(("success:", "failure:", "otherwise:")):
        return []
    if ":" in line:
        return []
    if re.fullmatch(r"(?:below )?DC \d+(?:-\d+)?", line):
        return [line]
    if line.startswith("DC "):
        return []

    match = re.fullmatch(r"(?P<label>.+?) below DC (?P<dc>[^;]+)(?P<extra>;.*)?", line)
    if match:
        extra = match.group("extra") or ""
        return _labeled_check_tags(match.group("label"), f"below DC {match.group('dc')}{extra}", choice_text)

    match = re.fullmatch(r"(?P<label>.+?) DC (?P<dc>[^;:]+)(?P<extra>;.*)?", line)
    if match:
        extra = match.group("extra") or ""
        return _labeled_check_tags(match.group("label"), f"DC {match.group('dc').strip()}{extra}", choice_text)

    match = re.fullmatch(r"(?P<label>.+?) below (?P<dc>\d+)", line)
    if match:
        return _labeled_check_tags(match.group("label"), f"below DC {match.group('dc')}", choice_text)

    match = re.fullmatch(r"(?P<label>.+?) (?P<dc>\d+)(?: \+ (?P<item>.+))?", line)
    if match:
        item = match.group("item")
        label = f"{match.group('label')} + {item}" if item else match.group("label")
        return _labeled_check_tags(label, f"DC {match.group('dc')}", choice_text)

    return []


def _labeled_check_tags(label: str, dc_tag: str, choice_text: str) -> list[str]:
    if _choice_tag_matches_check(choice_text, label):
        return [dc_tag]
    return [f"{label.strip()} {dc_tag}"]


def _choice_tag_matches_check(choice_text: str, check_label: str) -> bool:
    choice_tags = _leading_tags(choice_text)
    if not choice_tags:
        return False
    normalized_check = _normalize_check_label(check_label)
    return any(normalized_check in _normalize_check_label(tag) for tag in choice_tags)


def _leading_tags(text: str) -> list[str]:
    tags: list[str] = []
    rest = text.strip()
    while True:
        match = re.match(r"\[([^\]]+)\]\s*", rest)
        if not match:
            return tags
        tags.append(match.group(1).strip())
        rest = rest[match.end() :]


def _normalize_check_label(label: str) -> str:
    normalized = label.lower().replace("computer use", "computer")
    normalized = normalized.replace("persuade, lie", "persuade/lie")
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _build_state_effect_index(
    parsed: list[ParsedDialogue],
    scripts: list[ScriptResource],
    tlk: TlkTable,
) -> StateEffectIndex:
    script_analysis = _script_state_analyses(scripts)
    empty_index = StateEffectIndex({}, {}, {})
    effects_by_state: dict[tuple[str, int], list[str]] = {}

    for item in parsed:
        entries = _as_list(item.root.get("EntryList"))
        replies = _as_list(item.root.get("ReplyList"))
        for start_link in _as_list(item.root.get("StartingList")):
            entry_index = _index_from_link(start_link)
            if entry_index is None or not (0 <= entry_index < len(entries)):
                continue

            effects = _automatic_route_effect_lines(
                entry_index,
                entries,
                replies,
                tlk,
                empty_index,
                include_common_reply_effects=False,
            )
            if not effects:
                continue

            direct_states = _condition_states_from_link(start_link, item.resource.module_name)
            if direct_states:
                for key in direct_states:
                    _extend_unique(effects_by_state.setdefault(key, []), effects)
                continue

            for field, suffix in (("Active", ""), ("Active2", "b")):
                script = _script_key(_plain_text(start_link.get(field)))
                state_value = _condition_param(start_link, 1, suffix)
                if not script or state_value is None:
                    continue

                analysis = script_analysis.get(script)
                if analysis is None:
                    continue
                for global_name in analysis.global_reads:
                    key = (_state_key(global_name), state_value)
                    _extend_unique(effects_by_state.setdefault(key, []), effects)

    dynamic_global_sets: dict[str, list[str]] = {}
    constant_global_sets: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for script, analysis in script_analysis.items():
        if analysis.dynamic_global_sets:
            dynamic_global_sets[script] = sorted(analysis.dynamic_global_sets)
        for param_value, pairs in analysis.constant_global_sets.items():
            constant_global_sets[(script, param_value)] = list(dict.fromkeys(pairs))

    state_effects = StateEffectIndex(
        effects_by_state=effects_by_state,
        dynamic_global_sets=dynamic_global_sets,
        constant_global_sets=constant_global_sets,
    )
    _propagate_script_state_effects(effects_by_state, script_analysis)
    _refresh_start_state_effects(parsed, tlk, state_effects, script_analysis)
    start_states = _start_condition_states(parsed, script_analysis)
    for script, global_name, value, effects in _forced_terminal_start_transitions(parsed, tlk, state_effects):
        for (candidate_script, _param_value), pairs in constant_global_sets.items():
            if candidate_script != script:
                continue
            for candidate_global, candidate_value in pairs:
                key = (_state_key(candidate_global), candidate_value)
                if _state_key(candidate_global) != _state_key(global_name):
                    continue
                if candidate_value == value or key not in start_states:
                    continue
                _extend_unique(effects_by_state.setdefault(key, []), effects)
    _propagate_script_state_effects(effects_by_state, script_analysis)
    _refresh_start_state_effects(parsed, tlk, state_effects, script_analysis)
    contextual_script_effects = _contextual_script_effects(scripts, effects_by_state)
    cross_module_script_effects: dict[tuple[str, int], list[str]] = {}
    for (_module, script, value), effects in contextual_script_effects.items():
        _extend_unique(cross_module_script_effects.setdefault((script, value), []), effects)

    return StateEffectIndex(
        effects_by_state=effects_by_state,
        dynamic_global_sets=dynamic_global_sets,
        constant_global_sets=constant_global_sets,
        contextual_script_effects=contextual_script_effects,
        cross_module_script_effects=cross_module_script_effects,
    )


def _start_condition_states(
    parsed: list[ParsedDialogue],
    script_analysis: dict[str, NcsStateAnalysis],
) -> set[tuple[str, int]]:
    states: set[tuple[str, int]] = set()
    for item in parsed:
        for start_link in _as_list(item.root.get("StartingList")):
            direct_states = _condition_states_from_link(start_link, item.resource.module_name)
            if direct_states:
                states.update(direct_states)
                continue

            for field, suffix in (("Active", ""), ("Active2", "b")):
                script = _script_key(_plain_text(start_link.get(field)))
                value = _condition_param(start_link, 1, suffix)
                if not script or value is None:
                    continue
                analysis = script_analysis.get(script)
                if analysis is None:
                    continue
                for global_name in analysis.global_reads:
                    states.add((_state_key(global_name), value))
    return states


def _refresh_start_state_effects(
    parsed: list[ParsedDialogue],
    tlk: TlkTable,
    state_effects: StateEffectIndex,
    script_analysis: dict[str, NcsStateAnalysis],
) -> None:
    for _pass in range(2):
        changed = False
        for item in parsed:
            entries = _as_list(item.root.get("EntryList"))
            replies = _as_list(item.root.get("ReplyList"))
            for start_link in _as_list(item.root.get("StartingList")):
                entry_index = _index_from_link(start_link)
                if entry_index is None or not (0 <= entry_index < len(entries)):
                    continue

                effects = _automatic_route_effect_lines(
                    entry_index,
                    entries,
                    replies,
                    tlk,
                    state_effects,
                    include_common_reply_effects=False,
                )
                if not effects:
                    continue

                states = _condition_states_from_link(start_link, item.resource.module_name)
                if not states:
                    states = []
                    for field, suffix in (("Active", ""), ("Active2", "b")):
                        script = _script_key(_plain_text(start_link.get(field)))
                        value = _condition_param(start_link, 1, suffix)
                        if not script or value is None:
                            continue
                        analysis = script_analysis.get(script)
                        if analysis is None:
                            continue
                        for state_name in analysis.global_reads:
                            states.append((_state_key(state_name), value))

                for state in states:
                    before = len(state_effects.effects_by_state.setdefault(state, []))
                    _extend_unique(state_effects.effects_by_state[state], effects)
                    if len(state_effects.effects_by_state[state]) != before:
                        changed = True
        if changed:
            _propagate_script_state_effects(state_effects.effects_by_state, script_analysis)
        else:
            break


def _forced_terminal_start_transitions(
    parsed: list[ParsedDialogue],
    tlk: TlkTable,
    state_effects: StateEffectIndex,
) -> list[tuple[str, str, int, list[str]]]:
    transitions: list[tuple[str, str, int, list[str]]] = []
    for item in parsed:
        entries = _as_list(item.root.get("EntryList"))
        replies = _as_list(item.root.get("ReplyList"))
        for start_link in _as_list(item.root.get("StartingList")):
            entry_index = _index_from_link(start_link)
            if entry_index is None:
                continue
            transitions.extend(
                _forced_terminal_route_transitions(entry_index, entries, replies, tlk, state_effects)
            )
    return transitions


def _forced_terminal_route_transitions(
    entry_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex,
) -> list[tuple[str, str, int, list[str]]]:
    if not _is_forced_terminal_entry(entry_index, entries, replies, tlk, state_effects):
        return []

    transitions: list[tuple[str, str, int, list[str]]] = []
    seen: set[int] = set()
    current = entry_index
    while 0 <= current < len(entries) and current not in seen:
        seen.add(current)
        entry = entries[current]
        transitions.extend(_state_transition_effect_details(entry, state_effects))

        links = _as_list(entry.get("RepliesList"))
        if not links:
            break
        link = links[0]
        reply = _linked_reply(link, replies)
        if reply is None:
            break
        transitions.extend(_state_transition_effect_details(reply, state_effects))

        next_links = _as_list(reply.get("EntriesList"))
        if not next_links:
            break
        next_index = _index_from_link(next_links[0])
        if next_index is None:
            break
        current = next_index

    return transitions


def _script_state_analyses(scripts: list[ScriptResource]) -> dict[str, NcsStateAnalysis]:
    analyses: dict[str, NcsStateAnalysis] = {}
    for resource in scripts:
        script = _script_key(resource.script_name)
        if not script:
            continue
        try:
            analysis = _analyze_ncs_state(resource.data, resource.module_name)
        except ValueError:
            continue
        if not analysis.global_reads and not analysis.dynamic_global_sets and not analysis.constant_global_sets:
            continue

        existing = analyses.setdefault(script, NcsStateAnalysis(set(), set(), set(), {}))
        existing.global_reads.update(analysis.global_reads)
        existing.state_conditions.update(analysis.state_conditions)
        existing.dynamic_global_sets.update(analysis.dynamic_global_sets)
        for param_value, pairs in analysis.constant_global_sets.items():
            _extend_unique(existing.constant_global_sets.setdefault(param_value, []), pairs)
    return analyses


def _analyze_ncs_state(data: bytes, module_name: str | None = None) -> NcsStateAnalysis:
    instructions = _read_ncs_instructions(data)
    analysis = NcsStateAnalysis(set(), set(), set(), {})
    branch_targets = _script_param_branch_targets(instructions)
    current_param: int | None = None

    for index, instruction in enumerate(instructions):
        if index in branch_targets:
            current_param = branch_targets[index]

        if _is_action(instruction, 580):
            global_name, _string_index = _previous_const_string(instructions, index)
            if global_name:
                state = _global_number_state(global_name)
                analysis.global_reads.add(state)
                value = _following_equality_value(instructions, index)
                if value is not None:
                    analysis.state_conditions.add((state, value))
            continue

        if _is_action(instruction, 578):
            global_name, _string_index = _previous_const_string(instructions, index)
            if global_name:
                state = _global_boolean_state(global_name)
                analysis.global_reads.add(state)
                value = _following_equality_value(instructions, index)
                if value is not None:
                    analysis.state_conditions.add((state, 1 if value else 0))
            continue

        if _is_action(instruction, 679):
            slot = _previous_const_int(instructions, index)
            if slot is not None:
                state = _local_boolean_state(slot, module_name)
                analysis.global_reads.add(state)
                value = _following_equality_value(instructions, index)
                if value is not None:
                    analysis.state_conditions.add((state, 1 if value else 0))
            continue

        if _is_action(instruction, 581):
            global_name, string_index = _previous_const_string(instructions, index)
            if not global_name or string_index is None:
                continue

            value_instruction = _previous_instruction(instructions, string_index)
            if value_instruction is None:
                continue

            state = _global_number_state(global_name)
            if _is_const_int(value_instruction):
                value = int(value_instruction.args[0])
                param_value = current_param if current_param is not None else 0
                analysis.constant_global_sets.setdefault(param_value, []).append((state, value))
            elif _is_stack_copy(value_instruction):
                analysis.dynamic_global_sets.add(state)
            continue

        if _is_action(instruction, 579):
            global_name, string_index = _previous_const_string(instructions, index)
            if not global_name or string_index is None:
                continue

            value_instruction = _previous_instruction(instructions, string_index)
            if value_instruction is None:
                continue

            state = _global_boolean_state(global_name)
            if _is_const_int(value_instruction):
                value = 1 if int(value_instruction.args[0]) else 0
                param_value = current_param if current_param is not None else 0
                analysis.constant_global_sets.setdefault(param_value, []).append((state, value))
            elif _is_stack_copy(value_instruction):
                analysis.dynamic_global_sets.add(state)
            continue

        if _is_action(instruction, 680):
            local_set = _previous_local_set_values(instructions, index)
            if local_set is None:
                continue
            slot, value = local_set
            param_value = current_param if current_param is not None else 0
            analysis.constant_global_sets.setdefault(param_value, []).append(
                (_local_boolean_state(slot, module_name), 1 if value else 0)
            )
            continue

        if _is_action(instruction, 682):
            local_set = _previous_local_set_values(instructions, index)
            if local_set is None:
                continue
            slot, value = local_set
            param_value = current_param if current_param is not None else 0
            analysis.constant_global_sets.setdefault(param_value, []).append(
                (_local_number_state(slot, module_name), value)
            )

    for param_value, pairs in list(analysis.constant_global_sets.items()):
        analysis.constant_global_sets[param_value] = list(dict.fromkeys(pairs))
    return analysis


def _read_ncs_instructions(data: bytes) -> list[NcsInstruction]:
    if len(data) < 13 or data[:4] != b"NCS " or data[4:8] != b"V1.0":
        raise ValueError("unsupported NCS header")

    declared_size = struct.unpack_from(">I", data, 9)[0]
    limit = min(declared_size, len(data))
    position = 13
    instructions: list[NcsInstruction] = []

    while position < limit:
        if position + 2 > limit:
            raise ValueError("truncated NCS instruction")

        start_position = position
        opcode = data[position]
        qualifier = data[position + 1]
        position += 2
        args: tuple[object, ...] = ()

        if opcode in {0x01, 0x03, 0x26, 0x27}:
            _require_ncs_range(data, position, 6, limit)
            args = (struct.unpack_from(">i", data, position)[0], struct.unpack_from(">H", data, position + 4)[0])
            position += 6
        elif opcode == 0x04:
            if qualifier == 0x03:
                _require_ncs_range(data, position, 4, limit)
                args = (struct.unpack_from(">i", data, position)[0],)
                position += 4
            elif qualifier == 0x04:
                _require_ncs_range(data, position, 4, limit)
                args = (struct.unpack_from(">f", data, position)[0],)
                position += 4
            elif qualifier == 0x05:
                _require_ncs_range(data, position, 2, limit)
                length = struct.unpack_from(">H", data, position)[0]
                position += 2
                _require_ncs_range(data, position, length, limit)
                args = (data[position : position + length].decode("ascii", "ignore"),)
                position += length
            else:
                _require_ncs_range(data, position, 4, limit)
                args = (struct.unpack_from(">i", data, position)[0],)
                position += 4
        elif opcode == 0x05:
            _require_ncs_range(data, position, 3, limit)
            args = (struct.unpack_from(">H", data, position)[0], data[position + 2])
            position += 3
        elif opcode in {0x1B, 0x1D, 0x1E, 0x1F, 0x25}:
            _require_ncs_range(data, position, 4, limit)
            args = (struct.unpack_from(">i", data, position)[0],)
            position += 4
        elif opcode == 0x21:
            _require_ncs_range(data, position, 6, limit)
            args = (
                struct.unpack_from(">H", data, position)[0],
                struct.unpack_from(">h", data, position + 2)[0],
                struct.unpack_from(">H", data, position + 4)[0],
            )
            position += 6
        elif opcode in {0x23, 0x24, 0x28, 0x29}:
            _require_ncs_range(data, position, 4, limit)
            args = (struct.unpack_from(">I", data, position)[0],)
            position += 4
        elif opcode == 0x2C:
            _require_ncs_range(data, position, 8, limit)
            args = (struct.unpack_from(">I", data, position)[0], struct.unpack_from(">I", data, position + 4)[0])
            position += 8
        elif opcode in {0x0B, 0x0C} and qualifier == 0x24:
            _require_ncs_range(data, position, 2, limit)
            args = (struct.unpack_from(">H", data, position)[0],)
            position += 2

        instructions.append(NcsInstruction(opcode, qualifier, args, start_position, position))

    return instructions


def _require_ncs_range(data: bytes, offset: int, size: int, limit: int) -> None:
    if offset < 0 or size < 0 or offset + size > limit or offset + size > len(data):
        raise ValueError("truncated NCS instruction")


def _script_param_branch_value(instructions: list[NcsInstruction], index: int) -> int | None:
    if index < 3:
        return None
    if not _is_jump_zero(instructions[index]):
        return None
    if not _is_stack_copy(instructions[index - 3]):
        return None
    if not _is_const_int(instructions[index - 2]):
        return None
    if not _is_int_equality(instructions[index - 1]):
        return None
    return int(instructions[index - 2].args[0])


def _script_param_branch_targets(instructions: list[NcsInstruction]) -> dict[int, int]:
    targets: dict[int, int] = {}
    offsets = {instruction.offset: index for index, instruction in enumerate(instructions)}
    for index, instruction in enumerate(instructions):
        if index < 3:
            continue
        if not (_is_jump_zero(instruction) or _is_jump_nonzero(instruction)):
            continue
        if not _is_stack_copy(instructions[index - 3]):
            continue
        if not _is_const_int(instructions[index - 2]):
            continue
        if not _is_int_equality(instructions[index - 1]):
            continue

        value = int(instructions[index - 2].args[0])
        if _is_jump_nonzero(instruction) and instruction.args:
            target = offsets.get(instruction.offset + int(instruction.args[0]))
            if target is not None:
                targets[target] = value
        elif index + 1 < len(instructions):
            targets[index + 1] = value
    return targets


def _following_equality_value(instructions: list[NcsInstruction], index: int) -> int | None:
    if index + 2 >= len(instructions):
        return None
    if not _is_const_int(instructions[index + 1]):
        return None
    if not _is_int_equality(instructions[index + 2]):
        return None
    return int(instructions[index + 1].args[0])


def _previous_const_string(instructions: list[NcsInstruction], index: int) -> tuple[str, int | None]:
    previous = _previous_instruction(instructions, index)
    if previous is None or previous.opcode != 0x04 or previous.qualifier != 0x05:
        return "", None
    return str(previous.args[0]), index - 1


def _previous_const_int(instructions: list[NcsInstruction], index: int) -> int | None:
    for previous in reversed(instructions[max(0, index - 6) : index]):
        if _is_const_int(previous):
            return int(previous.args[0])
    return None


def _previous_local_set_values(instructions: list[NcsInstruction], index: int) -> tuple[int, int] | None:
    values: list[int] = []
    for previous in reversed(instructions[max(0, index - 8) : index]):
        if _is_const_int(previous):
            values.append(int(previous.args[0]))
            if len(values) == 2:
                break
    if len(values) < 2:
        return None
    slot = values[0]
    value = values[1]
    return slot, value


def _previous_instruction(instructions: list[NcsInstruction], index: int) -> NcsInstruction | None:
    return instructions[index - 1] if index > 0 else None


def _is_action(instruction: NcsInstruction, action_id: int) -> bool:
    return instruction.opcode == 0x05 and bool(instruction.args) and int(instruction.args[0]) == action_id


def _is_const_int(instruction: NcsInstruction) -> bool:
    return instruction.opcode == 0x04 and instruction.qualifier == 0x03 and bool(instruction.args)


def _is_stack_copy(instruction: NcsInstruction) -> bool:
    return instruction.opcode in {0x03, 0x27}


def _is_int_equality(instruction: NcsInstruction) -> bool:
    return instruction.opcode == 0x0B and instruction.qualifier == 0x20


def _is_jump_zero(instruction: NcsInstruction) -> bool:
    return instruction.opcode == 0x1F


def _is_jump_nonzero(instruction: NcsInstruction) -> bool:
    return instruction.opcode == 0x25


def _state_transition_effect_lines(node: GffStruct, state_effects: StateEffectIndex) -> list[str]:
    effects: list[str] = []
    for _script, _global_name, _value, transition_effects in _state_transition_effect_details(node, state_effects):
        effects.extend(transition_effects)
    return _merged_annotations(effects)


def _cross_module_transition_effect_lines(
    node: GffStruct,
    state_effects: StateEffectIndex,
    route_effects: list[str],
) -> list[str]:
    if _has_alignment_effect(route_effects):
        return []

    effects: list[str] = []
    current_module = (state_effects.current_module or "").lower()
    for script, params in _action_script_calls(node):
        script_key = _script_key(script)
        if not script_key:
            continue

        state_value = params[0] if params else 0
        same_module_key = (current_module, script_key, state_value)
        if (state_effects.contextual_script_effects or {}).get(same_module_key):
            continue

        cross_effects = (state_effects.cross_module_script_effects or {}).get((script_key, state_value), [])
        _extend_unique(effects, cross_effects)
    return _merged_annotations(effects)


def _has_alignment_effect(annotations: list[str]) -> bool:
    return any(annotation.startswith(("Light Side ", "Dark Side ")) for annotation in annotations)


def _state_transition_effect_details(
    node: GffStruct,
    state_effects: StateEffectIndex,
) -> list[tuple[str, str, int, list[str]]]:
    details: list[tuple[str, str, int, list[str]]] = []
    for state_name, value in _direct_action_state_sets(node):
        effects = state_effects.effects_by_state.get((_state_key(state_name), value), [])
        if effects:
            details.append(("", state_name, value, _merged_annotations(effects)))

    for script, params in _action_script_calls(node):
        script_key = _script_key(script)
        if not script_key:
            continue
        state_value = params[0] if params else 0
        contextual_effects = (state_effects.contextual_script_effects or {}).get(
            ((state_effects.current_module or "").lower(), script_key, state_value),
            [],
        )
        if contextual_effects:
            details.append((script_key, f"context:{script_key}", state_value, _merged_annotations(contextual_effects)))

        for global_name, value in state_effects.constant_global_sets.get((script_key, state_value), []):
            effects = state_effects.effects_by_state.get((_state_key(global_name), value), [])
            if effects:
                details.append((script_key, global_name, value, _merged_annotations(effects)))

        for global_name in state_effects.dynamic_global_sets.get(script_key, []):
            effects = state_effects.effects_by_state.get((_state_key(global_name), state_value), [])
            if effects:
                details.append((script_key, global_name, state_value, _merged_annotations(effects)))

    return details


def _script_key(script: str) -> str:
    return Path(script).stem.lower()


def _state_key(global_name: str) -> str:
    return global_name.lower()


def _global_number_state(name: str) -> str:
    return f"global-number:{name.strip().lower()}"


def _global_boolean_state(name: str) -> str:
    return f"global-boolean:{name.strip().lower()}"


def _local_boolean_state(slot: int, module_name: str | None = None) -> str:
    return f"local-boolean:{(module_name or '').strip().lower()}:{slot}"


def _local_number_state(slot: int, module_name: str | None = None) -> str:
    return f"local-number:{(module_name or '').strip().lower()}:{slot}"


def _condition_states_from_link(link: GffStruct, module_name: str | None) -> list[tuple[str, int]]:
    states: list[tuple[str, int]] = []
    for field, suffix, str_field in (("Active", "", "ParamStrA"), ("Active2", "b", "ParamStrB")):
        script = _script_key(_plain_text(link.get(field)))
        if not script:
            continue
        value = _condition_param(link, 1, suffix)
        second_value = _condition_param(link, 2, suffix)
        string_param = _plain_text(link.get(str_field))
        state = _condition_state(script, value, second_value, string_param, module_name)
        if state is not None and state not in states:
            states.append(state)
    return states


def _condition_state(
    script: str,
    value: int | None,
    second_value: int | None,
    string_param: str,
    module_name: str | None,
) -> tuple[str, int] | None:
    if value is None:
        return None
    if script in {"c_global_eq", "c_global_gt", "c_global_lt"} and string_param:
        return (_global_number_state(string_param), value)
    if script in {"c_glob_bool_set", "c_global_bool_set"} and string_param:
        return (_global_boolean_state(string_param), 1)
    if script in {"c_glob_bool_notset", "c_global_bool_notset"} and string_param:
        return (_global_boolean_state(string_param), 0)
    if script == "c_local_set":
        return (_local_boolean_state(value, module_name), 1)
    if script == "c_local_notset":
        return (_local_boolean_state(value, module_name), 0)
    if script == "c_localn_eq":
        return (_local_number_state(value, module_name), second_value or 0)
    return None


def _direct_action_state_sets(node: GffStruct) -> list[tuple[str, int]]:
    states: list[tuple[str, int]] = []
    for script, suffix, str_field in (("Script", "", "ActionParamStrA"), ("Script2", "b", "ActionParamStrB")):
        script_name = _script_key(_plain_text(node.get(script)))
        if not script_name:
            continue
        value = _int_value(node.get(f"ActionParam1{suffix}"))
        second_value = _int_value(node.get(f"ActionParam2{suffix}"))
        string_param = _plain_text(node.get(str_field))
        state: tuple[str, int] | None = None
        if script_name == "a_global_set" and string_param and value is not None:
            state = (_global_number_state(string_param), value)
        elif script_name == "a_glob_bool_set" and string_param:
            state = (_global_boolean_state(string_param), 1 if value else 0)
        elif script_name == "a_local_set" and value is not None:
            state = (_local_boolean_state(value), 1)
        elif script_name == "a_local_reset" and value is not None:
            state = (_local_boolean_state(value), 0)
        elif script_name == "a_localn_set" and value is not None and second_value is not None:
            state = (_local_number_state(value), second_value)
        if state is not None and state not in states:
            states.append(state)
    return states


def _extend_unique(items: list, additions: list) -> None:
    for item in additions:
        if item not in items:
            items.append(item)


def _propagate_script_state_effects(
    effects_by_state: dict[tuple[str, int], list[str]],
    script_analysis: dict[str, NcsStateAnalysis],
) -> None:
    changed = True
    while changed:
        changed = False
        for script, analysis in script_analysis.items():
            if not analysis.state_conditions:
                continue
            if _skip_state_effect_propagation(script, analysis):
                continue
            downstream_effects: list[str] = []
            for state_name, value in analysis.constant_global_sets.get(0, []):
                _extend_unique(downstream_effects, effects_by_state.get((_state_key(state_name), value), []))
            if not downstream_effects:
                continue
            for state in analysis.state_conditions:
                before = len(effects_by_state.setdefault(state, []))
                _extend_unique(effects_by_state[state], downstream_effects)
                if len(effects_by_state[state]) != before:
                    changed = True


def _skip_state_effect_propagation(script: str, analysis: NcsStateAnalysis) -> bool:
    if re.fullmatch(r"k_.+_enter", script):
        return True

    downstream_sets = sum(len(pairs) for pairs in analysis.constant_global_sets.values())
    return len(analysis.state_conditions) > 6 or downstream_sets > 12


def _contextual_script_effects(
    scripts: list[ScriptResource],
    effects_by_state: dict[tuple[str, int], list[str]],
) -> dict[tuple[str, str, int], list[str]]:
    effects: dict[tuple[str, str, int], list[str]] = {}
    available_scripts = {((resource.module_name or "").lower(), _script_key(resource.script_name)) for resource in scripts}

    for resource in scripts:
        module = (resource.module_name or "").lower()
        script = _script_key(resource.script_name)
        match = re.fullmatch(r"k_(?P<stem>.+?)_(?:damage|damaged|death|dead)", script)
        if not match:
            continue

        action_script = f"a_{match.group('stem')}"
        if (module, action_script) not in available_scripts:
            continue

        try:
            analysis = _analyze_ncs_state(resource.data, resource.module_name)
        except ValueError:
            continue

        script_effects: list[str] = []
        for state_name, value in analysis.constant_global_sets.get(0, []):
            _extend_unique(script_effects, effects_by_state.get((_state_key(state_name), value), []))
        if script_effects:
            effects[(module, action_script, 1)] = _merged_annotations(script_effects)
    return effects


def _effect_lines(node: GffStruct) -> list[str]:
    effects: list[str] = []
    for script, params in _action_script_calls(node):
        normalized = script.lower()
        amount = params[0] if params else 0
        if normalized in ALIGNMENT_SCRIPT_EFFECTS:
            side, points = ALIGNMENT_SCRIPT_EFFECTS[normalized]
            effects.append(f"{side} +{points}")
        elif normalized == "a_givelight":
            effects.append(f"Light Side +{amount or 1}")
        elif normalized == "a_givedark":
            effects.append(f"Dark Side +{amount or 1}")
        elif normalized == "a_givecomp":
            effects.append(f"+{amount or 1} components")
        elif normalized in {"a_influence_inc", "a_influence_dec", "a_setinfluence"}:
            companion = _companion_label(params[0] if len(params) >= 1 else 0)
            influence_amount = params[1] if len(params) >= 2 else 1
            if normalized == "a_influence_dec":
                influence_amount = -abs(influence_amount or 1)
            elif normalized == "a_influence_inc":
                influence_amount = abs(influence_amount or 1)
            if influence_amount:
                effects.append(f"{companion} Influence {influence_amount:+d}")
    return effects


def _action_script_calls(node: GffStruct) -> list[tuple[str, list[int]]]:
    scripts: list[tuple[str, list[int]]] = []
    first = _plain_text(node.get("Script"))
    if first:
        scripts.append((first, _action_params(node, "", 5)))
    second = _plain_text(node.get("Script2"))
    if second:
        scripts.append((second, _action_params(node, "b", 5)))
    return scripts


def _action_params(node: GffStruct, suffix: str, count: int) -> list[int]:
    return [_int_value(node.get(f"ActionParam{index}{suffix}")) or 0 for index in range(1, count + 1)]


def _companion_label(companion_id: int) -> str:
    names = {
        0: "Atton",
        1: "Bao-Dur",
        2: "Mandalore",
        3: "G0-T0",
        4: "Handmaiden",
        5: "HK-47",
        6: "Kreia",
        7: "Mira",
        8: "T3-M4",
        9: "Visas",
        10: "Hanharr",
        11: "Disciple",
    }
    return names.get(companion_id, f"Companion {companion_id}")


def _entry_outcomes(
    indices: list[int],
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
) -> str:
    state_effects = state_effects or StateEffectIndex({}, {}, {})
    return ", ".join(_entry_outcome(index, entries, replies, tlk, state_effects) for index in indices)


def _common_entry_outcome(
    indices: list[int],
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
) -> str:
    state_effects = state_effects or StateEffectIndex({}, {}, {})
    outcomes = [_entry_outcome(index, entries, replies, tlk, state_effects) for index in indices]
    unique_outcomes = list(dict.fromkeys(outcome for outcome in outcomes if outcome))
    if len(unique_outcomes) == 1:
        return unique_outcomes[0]

    branch_effects = [
        _automatic_route_effect_lines(index, entries, replies, tlk, state_effects)
        for index in indices
        if 0 <= index < len(entries)
    ]
    common = _common_annotations(branch_effects)
    if common:
        return ", ".join(common)
    if unique_outcomes:
        return " / ".join(unique_outcomes)
    return _entry_outcomes(indices, entries, replies, tlk, state_effects)


def _entry_outcome(
    index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
) -> str:
    state_effects = state_effects or StateEffectIndex({}, {}, {})
    summary = ""
    if 0 <= index < len(entries):
        summary = _outcome_summary(index, entries, replies, tlk, state_effects)
    return summary or f"Entry {index}"


def _outcome_summary(
    index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex | None = None,
) -> str:
    effects = _automatic_route_effect_lines(index, entries, replies, tlk, state_effects)
    if effects:
        return ", ".join(effects)
    return ""


def _skill_check_from_script(
    script: str,
    link: GffStruct,
    reply_text: str = "",
    param_suffix: str = "",
) -> str | None:
    details = _skill_check_details(script, link, reply_text, param_suffix)
    if details is None:
        return None
    return _skill_check_label(details)


def _skill_check_details(
    script: str,
    link: GffStruct,
    reply_text: str = "",
    param_suffix: str = "",
) -> dict[str, object] | None:
    custom = _custom_skill_check_details(script, link, param_suffix)
    if custom is not None:
        return custom

    attribute = _attribute_check_details(script, link, param_suffix)
    if attribute is not None:
        return attribute

    match = re.fullmatch(r"c_sc_(?P<skill>[a-z]+)_(?P<op>gt|lt|bet)", script.lower())
    if not match:
        return None

    skill = _skill_label(match.group("skill"), reply_text)
    if skill is None:
        return None

    op = match.group("op")
    first = _condition_param(link, 1, param_suffix)
    second = _condition_param(link, 2, param_suffix)
    if first is None:
        return {"skill": skill, "op": op, "dc": ""}
    if op == "gt":
        return {"skill": skill, "op": op, "dc": first + 1}
    if op == "lt":
        return {"skill": skill, "op": op, "dc": first + 1}
    if op == "bet" and second is not None:
        return {"skill": skill, "op": op, "dc": f"{first + 1}-{second}"}
    return {"skill": skill, "op": op, "dc": ""}


def _attribute_check_details(script: str, link: GffStruct, param_suffix: str = "") -> dict[str, object] | None:
    match = re.fullmatch(r"c_ac_(?P<ability>str|dex|con|int|wis|cha)_(?P<op>gt|lt)", script.lower())
    if not match:
        return None

    value = _condition_param(link, 1, param_suffix)
    if value is None:
        return None
    abilities = {
        "str": "Strength",
        "dex": "Dexterity",
        "con": "Constitution",
        "int": "Intelligence",
        "wis": "Wisdom",
        "cha": "Charisma",
    }
    op = match.group("op")
    threshold = value + 1 if op == "gt" else value
    return {"skill": abilities[match.group("ability")], "op": f"attr_{op}", "dc": threshold}


def _custom_skill_check_details(script: str, link: GffStruct, param_suffix: str = "") -> dict[str, object] | None:
    normalized = script.lower()
    simple_checks = {
        "c_skilrep": "Repair",
        "c_skilcom": "Computer Use",
        "c_skilcomnorep": "Computer Use",
        "c_ic_skildemboo2": "Demolitions",
        "c_ic_skilcomboo2": "Computer Use",
    }
    if normalized in simple_checks:
        return {"skill": simple_checks[normalized], "op": "min", "dc": ""}

    credit_check = re.fullmatch(r"c_sc_(?P<skill>awa|rep)gtcredlt", normalized)
    if credit_check:
        first = _condition_param(link, 1, param_suffix)
        credits = _condition_param(link, 2, param_suffix)
        skill = _skill_label(credit_check.group("skill"), "")
        if first is not None and skill is not None:
            label = f"{skill} DC {first + 1}"
            if credits:
                label += f"; credits below {credits}"
            return {"skill": label, "op": "raw", "dc": ""}

    if normalized != "c_chk103statrec":
        return None

    case = _condition_param(link, 1, param_suffix)
    base = _condition_param(link, 2, param_suffix)
    if case is None or base is None:
        return None

    checks = {
        1: ("Intelligence", base + 1, False),
        2: ("Intelligence", base + 1, True),
        3: ("Computer Use", base + 3, False),
        4: ("Computer Use", base + 1, True),
        5: ("Persuade", base + 1, False),
        6: ("Persuade", base + 1, True),
    }
    match = checks.get(case)
    if match is None:
        return None
    skill, dc, sonic_sensor = match
    return {"skill": skill, "op": "min", "dc": dc, "item": "Sonic Sensor" if sonic_sensor else ""}


def _condition_param(link: GffStruct, index: int, suffix: str = "") -> int | None:
    return _int_value(link.get(f"Param{index}{suffix}"))


def _skill_check_label(details: dict[str, object]) -> str:
    skill = str(details["skill"])
    dc = details.get("dc")
    if dc == "":
        return skill
    item = details.get("item")
    suffix = f" + {item}" if item else ""
    if details.get("op") == "raw":
        return skill
    if details.get("op") == "min":
        return f"{skill} {dc}{suffix}"
    if details.get("op") == "attr_gt":
        return f"{skill} {dc}"
    if details.get("op") == "attr_lt":
        return f"{skill} below {dc}"
    if details.get("op") == "lt":
        return f"{skill} below DC {dc}{suffix}"
    return f"{skill} DC {dc}{suffix}"


def _skill_label(code: str, reply_text: str) -> str | None:
    names = {
        "per": "Persuade",
        "rep": "Repair",
        "com": "Computer Use",
        "dem": "Demolitions",
        "sec": "Security",
        "awa": "Awareness",
        "tre": "Treat Injury",
        "ste": "Stealth",
    }
    name = names.get(code)
    tag = _leading_tag(reply_text)
    if code == "per" and tag and any(word in tag.lower() for word in ("persuade", "intimidate", "lie")):
        return tag
    return name


def _force_persuade_choice_tag(link: GffStruct, reply: GffStruct | None) -> str:
    levels: list[str] = []
    for field in ("Active", "Active2"):
        level = _force_persuade_level(_plain_text(link.get(field)))
        if level:
            levels.append(level)
    if reply is not None:
        for entry_link in _as_list(reply.get("EntriesList")):
            for field in ("Active", "Active2"):
                level = _force_persuade_level(_plain_text(entry_link.get(field)))
                if level:
                    levels.append(level)
    if "Dominate Mind" in levels:
        return "Dominate Mind"
    if "Affect Mind" in levels:
        return "Affect Mind"
    return ""


def _force_persuade_level(script: str) -> str:
    normalized = script.lower()
    if normalized in {"c_affect_mind", "c_mind_trick"}:
        return "Affect Mind"
    if normalized == "c_domin_mind":
        return "Dominate Mind"
    return ""


def _add_force_persuade_tag(text: str, force_tag: str) -> str:
    if not force_tag:
        return text
    if re.match(r"\[Force Persuade\]\s*\[" + re.escape(force_tag) + r"\]", text, re.IGNORECASE):
        return text
    return re.sub(
        r"^\s*\[Force Persuade\]\s*",
        f"[Force Persuade] [{force_tag}] ",
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def _leading_tag(text: str) -> str:
    match = re.match(r"\[([^\]]+)\]", text.strip())
    return match.group(1).strip() if match else ""


def _tag_without_check_line(text: str, link: GffStruct | None = None, reply: GffStruct | None = None) -> str:
    tag = _leading_tag(text)
    if not tag:
        return ""
    if tag.lower() == "continue":
        return ""
    if tag.lower() == "force persuade":
        return ""
    if _has_resource_cost(text):
        return ""
    if _is_skill_choice_tag(tag) and _has_unresolved_check_script(link, reply):
        return f"{tag}; no DC found"
    return ""


def _has_unresolved_check_script(link: GffStruct | None, reply: GffStruct | None) -> bool:
    for script in _condition_scripts(link, reply):
        normalized = script.lower()
        if _known_non_check_condition(normalized):
            continue
        if "skil" in normalized or normalized.startswith(("c_sc_", "c_ac_")):
            return True
        if normalized.startswith(("c_chk", "k_con_")):
            return True
    return False


def _condition_scripts(link: GffStruct | None, reply: GffStruct | None) -> list[str]:
    scripts: list[str] = []
    for node in (link,):
        if node is None:
            continue
        for field in ("Active", "Active2"):
            value = _plain_text(node.get(field))
            if value:
                scripts.append(value)
    if reply is not None:
        for entry_link in _as_list(reply.get("EntriesList")):
            for field in ("Active", "Active2"):
                value = _plain_text(entry_link.get(field))
                if value:
                    scripts.append(value)
    return scripts


def _known_non_check_condition(script: str) -> bool:
    return (
        script.startswith(("c_local_", "c_global_", "c_hasitem", "c_quest_status", "c_influence_"))
        or script
        in {
            "c_chkcredits",
            "c_chkrevsex",
            "c_chksed",
            "c_chkturbocode",
            "c_hkdoors",
            "c_meddr",
            "c_npc_inprty",
            "c_skill_best",
            "c_skill_worst",
        }
    )


def _is_skill_choice_tag(tag: str) -> bool:
    skill_names = {
        "persuade",
        "intimidate",
        "awareness",
        "repair",
        "computer",
        "computer use",
        "demolitions",
        "security",
        "treat injury",
        "stealth",
        "intelligence",
    }
    parts = [part.strip().lower() for part in re.split(r"[/+]", tag)]
    return any(part in skill_names for part in parts)


def _has_resource_cost(text: str) -> bool:
    if "variable spike cost" in text.lower() or "variable repair part cost" in text.lower():
        return True
    if "base cost:" in text.lower():
        return True
    return (
        re.search(r"[\[(]\s*(?:<CUSTOM\d+>|\d+)\s+spikes?(?:\(s\))?\s*[\])]", text, re.IGNORECASE)
        is not None
        or re.search(r"[\[(]\s*(?:<CUSTOM\d+>|\d+)\s+(?:repair\s+)?parts?(?:\(s\))?\s*[\])]", text, re.IGNORECASE)
        is not None
    )


def _int_value(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _is_empty_transition_entry(entry: GffStruct, replies: list[GffStruct], tlk: TlkTable) -> bool:
    text, notes = _split_designer_notes(_resolve_text(entry, tlk))
    if text or notes:
        return False
    return _trivial_continue_next(entry, replies, tlk) is not None


def _is_noninteractive_examine_dialogue(entries: list[GffStruct], replies: list[GffStruct], tlk: TlkTable) -> bool:
    visible_entries = []
    for entry in entries:
        text, notes = _split_designer_notes(_resolve_text(entry, tlk))
        if text or notes:
            visible_entries.append(entry)
    if len(visible_entries) != 1:
        return False

    links = _as_list(visible_entries[0].get("RepliesList"))
    if not links:
        return True
    return len(links) == 1 and _is_trivial_end_continue(links[0], replies, tlk)


def _is_low_value_dialogue(
    resource: DialogueResource,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> bool:
    stem = Path(resource.dlg_name).stem.lower()
    if stem in {"kolto"}:
        return True
    return _is_noninteractive_examine_dialogue(entries, replies, tlk)


def _is_orphan_entry(
    entry: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> bool:
    return not _reply_lines(entry, entries, replies, tlk)


def _entry_has_meaningful_replies(
    entry: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> bool:
    return len(_reply_lines(entry, entries, replies, tlk)) >= 2


def _is_forced_terminal_entry(
    entry_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex,
) -> bool:
    return _forced_terminal_entry_end(entry_index, entries, replies, tlk, state_effects, set())


def _forced_terminal_entry_end(
    entry_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    state_effects: StateEffectIndex,
    seen: set[int],
) -> bool:
    if entry_index in seen or not (0 <= entry_index < len(entries)):
        return False

    seen.add(entry_index)
    entry = entries[entry_index]
    reply_lines = _reply_lines(entry, entries, replies, tlk, state_effects)
    if not reply_lines:
        return not _auto_route_reaches_meaningful_replies(entry, entries, replies, tlk)
    if len(reply_lines) != 1:
        return False

    links = _as_list(entry.get("RepliesList"))
    if len(links) != 1:
        return False
    link = links[0]
    if _link_detail_lines(link):
        return False

    reply = _linked_reply(link, replies)
    if reply is None:
        return True

    reply_text, _notes = _split_designer_notes(_resolve_text(reply, tlk))
    if _visibility_check_lines(link, reply_text):
        return False
    if _reply_check_lines(reply, reply_text, entries, replies, tlk, state_effects):
        return False

    next_links = _as_list(reply.get("EntriesList"))
    if not next_links:
        return True
    if len(next_links) != 1:
        return False
    next_link = next_links[0]
    if _link_detail_lines(next_link) or _visibility_check_lines(next_link, ""):
        return False

    next_index = _index_from_link(next_link)
    if next_index is None:
        return True
    return _forced_terminal_entry_end(next_index, entries, replies, tlk, state_effects, seen)


def _is_trivial_end_continue(link: GffStruct, replies: list[GffStruct], tlk: TlkTable) -> bool:
    if _link_detail_lines(link) or _visibility_check_lines(link, ""):
        return False
    reply_index = _index_from_link(link)
    if reply_index is None or not (0 <= reply_index < len(replies)):
        return False

    reply = replies[reply_index]
    reply_text, _notes = _split_designer_notes(_resolve_text(reply, tlk))
    if reply_text and reply_text.lower() != "[continue]":
        return False
    if _effect_lines(reply) or _reply_check_lines(reply, reply_text, [], replies, tlk):
        return False
    return not _as_list(reply.get("EntriesList"))


def _is_trivial_continue_reply(link: GffStruct, replies: list[GffStruct], tlk: TlkTable) -> bool:
    if _link_detail_lines(link):
        return False
    reply_index = _index_from_link(link)
    if reply_index is None or not (0 <= reply_index < len(replies)):
        return False

    reply = replies[reply_index]
    reply_text, _notes = _split_designer_notes(_resolve_text(reply, tlk))
    if reply_text and reply_text.lower() != "[continue]":
        return False
    return True


def _entry_speaker(entry: GffStruct, speaker_hint: str) -> str:
    speaker = _plain_text(entry.get("Speaker"))
    if speaker:
        return _pretty_label(speaker)
    return speaker_hint


def _resolve_entry_speaker_labels(
    entries: list[GffStruct],
    resource: DialogueResource,
    speaker_names: SpeakerNameIndex | None,
) -> None:
    if speaker_names is None:
        return
    for entry in entries:
        speaker = _plain_text(entry.get("Speaker"))
        resolved = _lookup_speaker_name(speaker, resource, speaker_names)
        if resolved:
            entry.fields["Speaker"] = resolved


def _build_speaker_name_index(resources: list[NameResource], tlk: TlkTable) -> SpeakerNameIndex:
    by_module: dict[tuple[str, str], list[str]] = {}
    by_key: dict[str, list[str]] = {}

    for resource in resources:
        try:
            root = read_gff(resource.data)
        except Exception as exc:
            LOGGER.warning("failed to parse %s::%s: %s", resource.source_path, resource.resource_name, exc)
            continue

        name = _resource_speaker_name(root, tlk)
        if not name:
            continue

        module = (resource.module_name or "").lower()
        keys = _speaker_lookup_keys(Path(resource.resource_name).stem)
        for field in ("TemplateResRef", "Tag", "Conversation", "Dialog", "Dialogue"):
            keys.extend(_speaker_lookup_keys(_plain_text(root.get(field))))

        for key in dict.fromkeys(keys):
            _add_speaker_name(by_key, key, name)
            if module:
                _add_speaker_name(by_module, (module, key), name)

    return SpeakerNameIndex(names_by_module_key=by_module, names_by_key=by_key)


def _resource_speaker_name(root: GffStruct, tlk: TlkTable) -> str:
    first = _resolve_locstring(root.get("FirstName"), tlk)
    last = _resolve_locstring(root.get("LastName"), tlk)
    combined = " ".join(part for part in (first, last) if part).strip()
    if combined:
        return _clean_resource_speaker_name(combined)

    for field in ("LocalizedName", "LocName"):
        name = _resolve_locstring(root.get(field), tlk)
        if name:
            return _clean_resource_speaker_name(name)
    return ""


def _resolve_locstring(value: object, tlk: TlkTable) -> str:
    if isinstance(value, dict):
        strref = value.get("strref", -1)
        if isinstance(strref, int) and strref >= 0:
            resolved = tlk.get(strref)
            if resolved:
                return resolved
        for substring in value.get("substrings") or []:
            text = substring.get("text", "")
            if text:
                return text
    if isinstance(value, int):
        return tlk.get(value) if value >= 0 else ""
    if isinstance(value, str):
        return value
    return ""


def _clean_resource_speaker_name(text: str) -> str:
    text, _notes = _split_designer_notes(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text.lower().startswith("bad strref"):
        return ""
    canonical = _canonical_speaker_name(text)
    if canonical:
        return canonical
    if "_" in text or text.islower() or re.fullmatch(r"[A-Za-z0-9_-]+", text):
        return _pretty_label(text)
    return text


def _canonical_speaker_name(text: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    known = {
        "3cfd": "3C-FD",
        "1b8d": "1B-8D",
        "b4d4": "B-4D4",
        "b5d8": "B-5D8",
        "c7e3": "C7-E3",
        "c9t9": "C-9T-9",
        "g0t0": "G0-T0",
        "hk47": "HK-47",
        "hk50": "HK-50",
        "it31": "IT-31",
        "p1dk": "P-1DK",
        "s4c8": "S4-C8",
        "t3m4": "T3-M4",
        "t1n1": "T1-N1",
        "tt32": "TT-32",
    }
    return known.get(compact, "")


def _speaker_lookup_keys(value: str) -> list[str]:
    value = value.strip()
    if not value:
        return []
    lowered = value.lower()
    cleaned = _clean_speaker_tag(value).lower()
    stripped = re.sub(r"^\d{3,}", "", cleaned).strip("_- ")
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    stripped_compact = re.sub(r"[^a-z0-9]+", "", stripped)
    return [
        key
        for key in dict.fromkeys((lowered, cleaned, stripped, compact, stripped_compact))
        if key
    ]


def _add_speaker_name(mapping: dict[object, list[str]], key: object, name: str) -> None:
    if not name:
        return
    bucket = mapping.setdefault(key, [])
    if name not in bucket:
        bucket.append(name)


def _lookup_speaker_name(
    value: str,
    resource: DialogueResource,
    speaker_names: SpeakerNameIndex | None,
) -> str:
    if not value or speaker_names is None:
        return ""

    module = (resource.module_name or "").lower()
    module_candidates = [candidate for candidate in ("override", module) if candidate]
    keys = _speaker_lookup_keys(value)

    for key in keys:
        for module_key in module_candidates:
            resolved = _unique_speaker_name(speaker_names.names_by_module_key.get((module_key, key), []))
            if resolved:
                return resolved

    for key in keys:
        resolved = _unique_speaker_name(speaker_names.names_by_key.get(key, []))
        if resolved:
            return resolved
    return ""


def _unique_speaker_name(names: list[str]) -> str:
    cleaned = [_clean_resource_speaker_name(name) for name in names]
    cleaned = [name for name in cleaned if name]
    if not cleaned:
        return ""

    by_lower: dict[str, str] = {}
    for name in cleaned:
        by_lower.setdefault(name.lower(), name)
    return next(iter(by_lower.values())) if len(by_lower) == 1 else ""


def _conversation_speaker_hint(
    resource: DialogueResource,
    root: GffStruct,
    speaker_names: SpeakerNameIndex | None = None,
) -> str:
    vo_id = _plain_text(root.get("VO_ID"))
    if vo_id:
        resolved = _lookup_speaker_name(vo_id, resource, speaker_names)
        if resolved:
            return resolved
        return _pretty_label(vo_id)

    stem = Path(resource.dlg_name).stem.lower()
    labels = {
        "admoff": "Administration Officer",
        "kolto": "Exile's Kolto Tank",
        "medlog": "Medical Computer",
        "va_offic": "Vaklu Officer",
        "101atton": "Atton",
        "102atton": "Atton",
        "103atton": "Atton",
        "104atton": "Atton",
        "106atton": "Atton",
        "hk50": "HK-50",
        "hk47": "HK-47",
        "t3m4": "T3-M4",
        "kredead": "",
    }
    if stem.endswith("atton"):
        return "Atton"
    if stem in labels:
        return labels[stem]
    resolved = _lookup_speaker_name(stem, resource, speaker_names)
    if resolved:
        return resolved
    return _speaker_hint_from_stem(stem)


def _speaker_hint_from_stem(stem: str) -> str:
    normalized = stem.lower()
    stripped = re.sub(r"^\d{3,}", "", normalized).strip("_-")
    known = {
        "3cfd": "3C-FD",
        "attpazzak": "Atton",
        "attond": "Atton",
        "attonend": "Atton",
        "admlog": "Administration Console",
        "board": "Ebon Hawk Console",
        "cahhmakt": "Cahhmakt",
        "cmp_drd": "Droid Control Station",
        "cmp_tur": "Turret Control Station",
        "drdparts": "HK Unit",
        "drocon": "Maintenance Console",
        "emerstat": "Emergency Field Station",
        "emrhatch": "Emergency Hatch",
        "fuelcon": "Fuel Control Console",
        "galaxy": "Galaxy Map",
        "galaxy2": "Galaxy Map",
        "hangterm": "Hangar Control Console",
        "kreia": "Kreia",
        "mindrd": "Mining Droid",
        "secsys": "Security System",
        "secter": "Security Console",
        "shfcon": "Mining Control Console",
        "test_droid": "Mining Droid",
        "vissparrpc": "Visas",
        "visasend": "Visas",
        "atton": "Atton",
        "t3m4": "T3-M4",
        "t3": "T3-M4",
        "hk47": "HK-47",
        "hk50": "HK-50",
        "mand": "Mandalore",
        "mandalor": "Mandalore",
        "handma": "Handmaiden",
        "disc": "Disciple",
        "disciple": "Disciple",
        "bao": "Bao-Dur",
        "baodur": "Bao-Dur",
        "mira": "Mira",
        "visas": "Visas",
        "g0t0": "G0-T0",
        "goto": "G0-T0",
        "hanharr": "Hanharr",
        "kavar": "Kavar",
        "vrook": "Vrook",
        "vash": "Vash",
        "zez": "Zez-Kai Ell",
        "sion": "Sion",
        "atris": "Atris",
        "atrend3": "Atris",
        "talia": "Queen Talia",
        "vaklu": "General Vaklu",
        "tobin": "Colonel Tobin",
        "kelborn": "Kelborn",
        "kex": "Kex",
        "zherron": "Zherron",
        "nikko": "Nikko",
        "grenn": "Grenn",
        "batono": "Batono",
        "luxa": "Luxa",
        "ramana": "Ramana",
        "harra": "Harra",
        "vogga": "Vogga",
        "visquis": "Visquis",
    }
    if normalized in known:
        return known[normalized]
    if stripped in known:
        return known[stripped]
    for token, label in known.items():
        if re.search(rf"(^|_){re.escape(token)}($|_)", stripped):
            return label

    prefix_labels = (
        ("att", "Atton"),
        ("vis", "Visas"),
        ("disc", "Disciple"),
        ("hand", "Handmaiden"),
        ("bao", "Bao-Dur"),
        ("mand", "Mandalore"),
        ("hk", "HK-47"),
        ("t3", "T3-M4"),
        ("kre", "Kreia"),
        ("sion", "Sion"),
        ("atris", "Atris"),
        ("kavar", "Kavar"),
        ("vrook", "Vrook"),
    )
    for prefix, label in prefix_labels:
        if stripped.startswith(prefix):
            return label

    object_labels = (
        ("workbnch", "Workbench"),
        ("terminal", "Terminal"),
        ("comcon", "Communications Console"),
        ("seccon", "Security Console"),
        ("console", "Console"),
        ("hyper", "Hyperdrive"),
        ("cmp_fix", "Engine Console"),
        ("door_ovr", "Door Override"),
        ("lift", "Lift"),
        ("sparkwir", "Sparking Wires"),
        ("treatinj", "Medpac"),
        ("intro", "System"),
        ("outro", "System"),
    )
    for token, label in object_labels:
        if token in stripped:
            return label

    return _pretty_label(stripped)


def _is_terminal_label(label: str) -> bool:
    return any(word in label.lower() for word in ("computer", "console", "terminal", "workbench", "station"))


def _pretty_label(value: str) -> str:
    normalized = value.strip()
    canonical = _canonical_speaker_name(normalized)
    if canonical:
        return canonical
    known = {
        "909sion": "Sion",
        "atton": "Atton",
        "admoff": "Administration Officer",
        "adm_console": "Administration Console",
        "atristemp": "Atris",
        "atriscut": "Atris",
        "b4d4": "B-4D4",
        "b5d8": "B-5D8",
        "b2term": "Droid Maintenance Chamber Control",
        "benc-99": "Workbench",
        "benc99": "Workbench",
        "baodur": "Bao-Dur",
        "bao_dur": "Bao-Dur",
        "bh_rodian": "Rodian Bounty Hunter",
        "222tel": "Shuttle",
        "col_tobin": "Colonel Tobin",
        "comchannel": "Comlink",
        "darthnihilus": "Darth Nihilus",
        "darthsion": "Darth Sion",
        "darthtraya": "Darth Traya",
        "discip": "Disciple",
        "drdp": "Protocol Droid",
        "drdith": "Ithorian Droid",
        "exchangethug": "Exchange Thug",
        "exchangethug302_1": "Exchange Thug",
        "exchangethug302_2": "Exchange Thug",
        "fake_kreia": "Kreia",
        "g0t0": "G0-T0",
        "gandfind": "Gand",
        "gam_enforcer": "Exchange Enforcer",
        "goto": "G0-T0",
        "gotoholo": "G0-T0",
        "gotovoic": "G0-T0",
        "grennph": "Grenn",
        "gsoldier": "Onderon Soldier",
        "handma": "Handmaiden",
        "ithholo": "Ithorian",
        "jedimaster1": "Master Vash",
        "jedimaster2": "Master Vrook",
        "jedimaster3": "Master Kavar",
        "jedimaster4": "Master Zez-Kai Ell",
        "jedimaster-1": "Master Vash",
        "jedimaster-2": "Master Vrook",
        "jedimaster-3": "Master Kavar",
        "jedimaster-4": "Master Zez-Kai Ell",
        "jedimaster": "Jedi Master",
        "kreiaevil": "Kreia",
        "kreiainv": "Kreia",
        "kreiavoi": "Kreia",
        "medoff": "Medical Officer",
        "mainof": "Maintenance Officer",
        "mand": "Mandalore",
        "mandalore": "Mandalore",
        "medlog": "Medical Computer",
        "npc_kelborn": "Kelborn",
        "npc_nikko": "Nikko",
        "npc_vrook": "Vrook",
        "npc_zherron": "Zherron",
        "ond_soldier_ri": "Onderon Soldier",
        "psoldier": "Palace Soldier",
        "recapt": "Republic Captain",
        "ref-2": "Refugee",
        "ref2": "Refugee",
        "redeclipsecrew-1": "Red Eclipse Crew",
        "redeclipsecrew-2": "Red Eclipse Crew",
        "redeclipsecrew-3": "Red Eclipse Crew",
        "redeclipsecrew": "Red Eclipse Crew",
        "rethug-4": "Exchange Thug",
        "sec_terminal50": "Security Terminal",
        "sister-1": "Handmaiden Sister",
        "sister-2": "Handmaiden Sister",
        "sister": "Handmaiden Sister",
        "sister1cut": "Handmaiden Sister",
        "sister1wind": "Handmaiden Sister",
        "sister2cut": "Handmaiden Sister",
        "sister2wind": "Handmaiden Sister",
        "talia": "Queen Talia",
        "tempin-0001": "Refugee",
        "tempin0001": "Refugee",
        "thgd": "Exchange Thug",
        "t1n1": "T1-N1",
        "tobin": "Colonel Tobin",
        "trma-2": "Airlock 2 Terminal",
        "trma2": "Airlock 2 Terminal",
        "twiholo": "Twi'lek",
        "twilek_servant": "Twi'lek Servant",
        "twinsun-1": "Twin Sun",
        "twinsun-2": "Twin Sun",
        "twinsun": "Twin Sun",
        "tsfb": "TSF Officer",
        "tsf_smuggling": "TSF Officer",
        "vaklu": "General Vaklu",
        "vandar_holo": "Vandar",
        "visasmarr": "Visas Marr",
        "voggathug-1": "Vogga's Thug",
        "voggathug-2": "Vogga's Thug",
        "voggathug": "Vogga's Thug",
        "vrook_holo": "Vrook",
        "zezkaiel": "Zez-Kai Ell",
        "zezkaiell": "Zez-Kai Ell",
        "zhugbro": "Zhug Brother",
        "zhugshooter": "Zhug Shooter",
        "zhugthug-1": "Zhug Thug",
        "zhugthug": "Zhug Thug",
        "coortathug1": "Coorta's Thug",
        "coortathug2": "Coorta's Thug",
        "hk50": "HK-50",
        "hk47": "HK-47",
        "hk501": "HK-50",
        "hk502": "HK-50",
        "hk502cs": "HK-50",
        "hk503": "HK-50",
        "hk503cs": "HK-50",
        "hk50cs": "HK-50",
        "hk50cut2": "HK-50",
        "hk50intv": "HK-50",
        "hk50m1": "HK-50",
        "hk50t1": "HK-50",
        "hk50t2": "HK-50",
        "hk50t3": "HK-50",
        "hk50vic": "HK-50",
        "t3m4": "T3-M4",
        "3cfd": "3C-FD",
    }
    lowered = normalized.lower()
    if lowered in known:
        return known[lowered]

    droid = re.fullmatch(r"([a-z])(\d)([a-z])(\d)", lowered)
    if droid:
        return f"{droid.group(1).upper()}-{droid.group(2)}{droid.group(3).upper()}-{droid.group(4)}"

    numbered_group = re.fullmatch(r"([a-z_]+)[-_]?\d+", lowered)
    if numbered_group and numbered_group.group(1) in known:
        return known[numbered_group.group(1)]

    cleaned = _clean_speaker_tag(normalized)
    lowered_cleaned = cleaned.lower()
    if lowered_cleaned in known:
        return known[lowered_cleaned]

    droid = re.fullmatch(r"([a-z])(\d)([a-z])(\d)", lowered_cleaned)
    if droid:
        return f"{droid.group(1).upper()}-{droid.group(2)}{droid.group(3).upper()}-{droid.group(4)}"

    compact = re.fullmatch(r"([a-z]+)(\d+)", lowered)
    if compact:
        return f"{compact.group(1).upper()}-{compact.group(2)}"

    parts = re.split(r"[_\s-]+", cleaned)
    readable = " ".join(_title_label_part(part) for part in parts if part)
    return readable or normalized


def _clean_speaker_tag(value: str) -> str:
    cleaned = re.sub(r"^\d+_", "", value.strip())
    cleaned = re.sub(r"^(npc|n|g|m|c|plc|mp)_", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(_ph|_cut|cut)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip("_- ") or value.strip()


def _title_label_part(part: str) -> str:
    acronyms = {
        "tsf": "TSF",
        "ith": "Ithorian",
        "rod": "Rodian",
        "merc": "Mercenary",
        "drd": "Droid",
        "tel": "Telosian",
        "dan": "Dantooine",
        "ond": "Onderon",
        "ebo": "Ebon Hawk",
        "czerka": "Czerka",
    }
    lowered = part.lower()
    if lowered in acronyms:
        return acronyms[lowered]
    if re.fullmatch(r"[a-z]\d[a-z]\d", lowered):
        return f"{lowered[0].upper()}-{lowered[1]}{lowered[2].upper()}-{lowered[3]}"
    if re.fullmatch(r"[a-z]+\d+", lowered):
        letters = re.sub(r"\d+", "", lowered)
        digits = re.sub(r"\D+", "", lowered)
        return f"{letters.upper()}-{digits}"
    return part[:1].upper() + part[1:]


def _conversation_context(resource: DialogueResource) -> str:
    module = resource.module_name or ""
    area = _module_area(module)
    dlg = resource.dlg_name.removesuffix(".dlg")
    if area:
        return f"{module} ({area}) / {dlg}"
    return f"{module} / {dlg}" if module else dlg


def _module_area(module_name: str) -> str:
    module = module_name.upper()
    suffix = re.sub(r"^\d+", "", module)
    areas = {
        "EBO": "Ebon Hawk",
        "PER": "Peragus",
        "HAR": "Harbinger",
        "TEL": "Telos",
        "NAR": "Nar Shaddaa",
        "DXN": "Dxun",
        "OND": "Onderon",
        "DAN": "Dantooine",
        "KOR": "Korriban",
        "NIH": "Ravager",
        "MAL": "Malachor V",
        "COR": "Coruscant",
    }
    return areas.get(suffix, "")


def _script_fields(node: GffStruct) -> list[tuple[str, str]]:
    wanted = ("script", "active", "action", "condition")
    found: list[tuple[str, str]] = []
    for field in node.raw_fields:
        label = field.label.lower()
        if not any(token in label for token in wanted):
            continue
        if any(token in label for token in ("param", "delay", "wait", "quest")):
            continue
        if not isinstance(field.value, str):
            continue
        text = _plain_text(field.value)
        if text and text not in {"****", "-1"}:
            found.append((field.label, text))
    return found


def _index_from_link(link: GffStruct) -> int | None:
    for label in ("Index", "EntryIndex", "ReplyIndex"):
        value = link.get(label)
        if isinstance(value, int) and value >= 0:
            return value
    for field in link.raw_fields:
        if field.label.lower().endswith("index") and isinstance(field.value, int) and field.value >= 0:
            return field.value
    return None


def _write_all_dialogue(path: Path, dumped: list[DumpedDialogue]) -> None:
    parts = ["# KOTOR II Dialogue Dump", ""]
    for item in dumped:
        parts.append(item.markdown.rstrip())
        parts.append("")
    path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def _validate_game_dir(game_dir: Path) -> None:
    if not game_dir.is_dir():
        raise FileNotFoundError(f"Game directory does not exist: {game_dir}")


def _validate_output_dir(out_dir: Path, game_dir: Path) -> None:
    protected = {
        Path.cwd().resolve(),
        Path.home().resolve(),
        game_dir,
    }
    anchor = Path(out_dir.anchor).resolve()
    protected.add(anchor)

    if out_dir in protected:
        raise ValueError(f"Refusing to use protected directory as output: {out_dir}")
    if _is_relative_to(out_dir, game_dir):
        raise ValueError(f"Refusing to write output inside the game directory: {out_dir}")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _prepare_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for generated in ("all_dialogue.md", "by_module", "by_dlg"):
        path = out_dir / generated
        if path.is_dir():
            _validate_generated_tree(path)
            shutil.rmtree(path)
        elif path.exists():
            _validate_generated_file(path)
            path.unlink()
    (out_dir / OUTPUT_MARKER).write_text("Generated by k2dialog-dump.\n", encoding="utf-8")


def _validate_generated_tree(path: Path) -> None:
    for child in path.rglob("*"):
        if child.is_dir():
            continue
        if child.suffix.lower() != ".md":
            raise ValueError(f"Refusing to delete non-Markdown file in generated output tree: {child}")


def _validate_generated_file(path: Path) -> None:
    if path.name != "all_dialogue.md":
        raise ValueError(f"Refusing to delete unexpected generated file: {path}")
    try:
        first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except IndexError:
        first_line = ""
    if first_line != "# KOTOR II Dialogue Dump":
        raise ValueError(f"Refusing to overwrite non-dump file: {path}")


def _write_by_module(path: Path, dumped: list[DumpedDialogue]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[DumpedDialogue]] = {}
    for item in dumped:
        grouped.setdefault(item.resource.module_name or "unknown", []).append(item)

    for module, items in grouped.items():
        parts = [f"# {_md_escape(module)}", ""]
        for item in items:
            parts.append(item.markdown.rstrip())
            parts.append("")
        (path / f"{_safe_name(module)}.md").write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")


def _write_by_dlg(path: Path, dumped: list[DumpedDialogue]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    for item in dumped:
        stem = f"{item.resource.module_name or 'unknown'}_{item.resource.dlg_name}"
        name = _safe_name(stem) + ".md"
        if name in seen:
            seen[name] += 1
            name = _safe_name(stem) + f"_{seen[name]}.md"
        else:
            seen[name] = 1
        (path / name).write_text(item.markdown, encoding="utf-8")


def _as_list(value: object) -> list[GffStruct]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, GffStruct)]
    return []


def _plain_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int):
        return str(value)
    return ""


def _paragraph(text: str) -> str:
    return _md_escape(re.sub(r"\s+", " ", text).strip())


def _choice_text(text: str, prefix_tags: list[str] | None = None) -> str:
    return _paragraph(_insert_choice_prefix_tags(_replace_cost_tokens(text), prefix_tags or []))


def _insert_choice_prefix_tags(text: str, prefix_tags: list[str]) -> str:
    if not prefix_tags:
        return text

    stripped = text.strip()
    position = 0
    existing_tags: list[str] = []
    while True:
        match = re.match(r"\s*\[([^\]]+)\]\s*", stripped[position:])
        if not match:
            break
        existing_tags.append(match.group(1).strip())
        position += match.end()

    kept_existing_tags = [
        tag for tag in existing_tags if not any(_prefix_replaces_existing_tag(tag, prefix) for prefix in prefix_tags)
    ]
    rest = stripped[position:].lstrip()
    inserted_tags = kept_existing_tags + prefix_tags
    inserted = " ".join(f"[{tag}]" for tag in inserted_tags)
    return f"{inserted} {rest}".strip()


def _prefix_replaces_existing_tag(existing_tag: str, prefix_tag: str) -> bool:
    existing = _normalize_check_label(existing_tag)
    prefix = _normalize_check_label(prefix_tag)
    return bool(existing and existing != prefix and prefix.startswith(existing))


def _replace_cost_tokens(text: str) -> str:
    spike_base_costs = {
        "CUSTOM35": "base cost: 2 spikes; reduced by Computer Use",
    }

    def replace_spikes(match: re.Match[str]) -> str:
        token = match.group("token").upper()
        label = spike_base_costs.get(token, "variable spike cost; reduced by Computer Use")
        return f" [{label}]"

    def replace_parts(match: re.Match[str]) -> str:
        return " [variable repair part cost; reduced by Repair]"

    text = re.sub(
        r"[\[(]?\s*<(?P<token>CUSTOM\d+)>\s+spikes?(?:\(s\))?\s*[\])]?",
        replace_spikes,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"[\[(]?\s*<(?P<token>CUSTOM\d+)>\s+(?:repair\s+)?part\(s\)\s*[\])]?",
        replace_parts,
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"[\[(]?\s*<(?P<token>CUSTOM\d+)>\s+parts?\s+needed\s*[\])]?",
        replace_parts,
        text,
        flags=re.IGNORECASE,
    )


def _display_text(text: str, speaker: str) -> str:
    if _is_terminal_label(speaker):
        return _md_escape(_terminal_lines(text))
    if "\n" in text:
        return _md_escape(_multiline_text(text))
    return _paragraph(text)


def _speaker_text_lines(speaker: str, text: str) -> list[str]:
    if "\n" not in text:
        return [f"**{_md_escape(speaker)}:** {text}"]
    return [f"**{_md_escape(speaker)}:**", text]


def _terminal_lines(text: str) -> str:
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.strip().splitlines())
    if not text:
        return text
    text = re.sub(r"\s+MEDICAL BAY FUNCTIONS\s+", "\nMEDICAL BAY FUNCTIONS\n", text)
    text = re.sub(r"\s+EMERGENCY LOCKDOWN\s+", "\nEMERGENCY LOCKDOWN\n", text)
    text = re.sub(r"\s+ENTER COMMAND$", "\nENTER COMMAND", text)
    return text


def _multiline_text(text: str) -> str:
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.strip().splitlines())


def _split_designer_notes(text: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    remaining = text.strip()
    for match in re.finditer(r"\{([^{}]*)\}", remaining):
        note = match.group(1).strip()
        if note:
            notes.append(note)
    remaining = re.sub(r"[ \t]*\{[^{}]*\}[ \t]*", " ", remaining)
    remaining = re.sub(r"[ \t]+", " ", remaining).strip()
    remaining = re.sub(r"[ \t]+([,.;:?!])", r"\1", remaining)
    return remaining, notes


def _md_escape(text: str) -> str:
    return text.replace("\\", "\\\\")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "unknown"
