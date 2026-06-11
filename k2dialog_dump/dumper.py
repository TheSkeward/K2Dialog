from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil

from .archives import DialogueResource, find_dialogue_resources
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


LOGGER = logging.getLogger(__name__)
OUTPUT_MARKER = ".k2dialog_dump_output"
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

    dumped: list[DumpedDialogue] = []
    for resource in resources:
        try:
            root = read_gff(resource.data)
            markdown = render_dialogue(resource, root, tlk, show_unresolved_checks=options.show_unresolved_checks)
            if not markdown.strip():
                continue
            dumped.append(DumpedDialogue(resource=resource, markdown=markdown))
        except Exception as exc:
            LOGGER.warning("failed to parse %s::%s: %s", resource.source_path, resource.dlg_name, exc)

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
    show_unresolved_checks: bool = False,
) -> str:
    entries = _as_list(root.get("EntryList"))
    replies = _as_list(root.get("ReplyList"))
    speaker_hint = _conversation_speaker_hint(resource, root)
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
                branch_targets = _auto_choice_targets(last_entry, entries, replies, tlk)
                if branch_targets:
                    block = _render_transcript_chain_with_branches(
                        transcript_chain,
                        branch_targets,
                        entries,
                        replies,
                        tlk,
                        speaker_hint,
                        show_unresolved_checks=show_unresolved_checks,
                    )
                    if _block_seen(block, seen_blocks):
                        skip_entries.update(transcript_chain)
                        skip_entries.update(target for _label, target in branch_targets)
                        continue
                    lines.extend(block)
                    skip_entries.update(transcript_chain)
                    skip_entries.update(target for _label, target in branch_targets)
                else:
                    block = _render_transcript_chain(transcript_chain, entries, tlk, speaker_hint)
                    if _block_seen(block, seen_blocks):
                        skip_entries.update(transcript_chain)
                        continue
                    lines.extend(block)
                    skip_entries.update(transcript_chain)
                lines.append("---")
                lines.append("")
                rendered_any = True
                continue
        if _is_orphan_entry(entry, entries, replies, tlk):
            if not _auto_route_reaches_meaningful_replies(entry, entries, replies, tlk):
                continue
            continue
        block = _render_entry(
            index,
            entry,
            entries,
            replies,
            tlk,
            speaker_hint,
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
    links = _as_list(entry.get("RepliesList"))
    if len(links) != 1:
        return []

    link = links[0]
    reply_index = _index_from_link(link)
    if reply_index is None or not (0 <= reply_index < len(replies)):
        return []
    if _visibility_check_lines(link, "") or _link_detail_lines(link):
        return []

    reply = replies[reply_index]
    reply_text, _notes = _split_designer_notes(_resolve_text(reply, tlk))
    if reply_text and reply_text.lower() != "[continue]":
        return []
    if _effect_lines(reply) or _reply_check_lines(reply, reply_text, [], replies, tlk):
        return []

    next_links = _as_list(reply.get("EntriesList"))
    if not next_links:
        return []

    next_entries: list[int] = []
    for next_link in next_links:
        if _link_detail_lines(next_link):
            continue
        next_index = _index_from_link(next_link)
        if next_index is not None:
            next_entries.append(next_index)
    return next_entries


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


def _auto_choice_targets(
    entry: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> list[tuple[str, int]]:
    queue: list[tuple[str, int]] = _auto_next_entry_labels(entry, replies, tlk)
    seen: set[int] = set()
    targets: list[tuple[str, int]] = []
    while queue:
        label, entry_index = queue.pop(0)
        if entry_index in seen or not (0 <= entry_index < len(entries)):
            continue
        seen.add(entry_index)
        candidate = entries[entry_index]
        if _entry_has_meaningful_replies(candidate, entries, replies, tlk):
            targets.append((label, entry_index))
            continue
        queue.extend(_inherit_label(label, child_label, child_index) for child_label, child_index in _auto_next_entry_labels(candidate, replies, tlk))
    return targets


def _auto_next_entry_labels(entry: GffStruct, replies: list[GffStruct], tlk: TlkTable) -> list[tuple[str, int]]:
    links = _as_list(entry.get("RepliesList"))
    if len(links) != 1:
        return []

    link = links[0]
    reply_index = _index_from_link(link)
    if reply_index is None or not (0 <= reply_index < len(replies)):
        return []
    if _visibility_check_lines(link, "") or _link_detail_lines(link):
        return []

    reply = replies[reply_index]
    reply_text, _notes = _split_designer_notes(_resolve_text(reply, tlk))
    if reply_text and reply_text.lower() != "[continue]":
        return []
    if _effect_lines(reply) or _reply_check_lines(reply, reply_text, [], replies, tlk):
        return []

    targets: list[tuple[str, int]] = []
    for next_link in _as_list(reply.get("EntriesList")):
        if _link_detail_lines(next_link):
            continue
        next_index = _index_from_link(next_link)
        if next_index is not None:
            targets.append((_condition_label(_plain_text(next_link.get("Active"))), next_index))
    return targets


def _inherit_label(parent: str, child: str, index: int) -> tuple[str, int]:
    return _condition_label_join(parent, child), index


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
    }
    return labels.get(script.lower(), "")


def _render_transcript_chain(
    chain: list[int],
    entries: list[GffStruct],
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

    if not turns:
        if speaker_hint:
            lines.append(f"**{_md_escape(speaker_hint)}:** [no prompt shown]")
        else:
            lines.append("[no prompt shown]")
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
    *,
    show_unresolved_checks: bool = False,
) -> list[str]:
    lines = _render_transcript_chain(chain, entries, tlk, speaker_hint)
    reply_lines = _reply_lines(
        entries[chain[-1]],
        entries,
        replies,
        tlk,
        show_unresolved_checks=show_unresolved_checks,
    )
    if reply_lines:
        lines.append("")
        lines.extend(reply_lines)
    return lines


def _render_transcript_chain_with_branches(
    chain: list[int],
    branch_targets: list[tuple[str, int]],
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    speaker_hint: str,
    *,
    show_unresolved_checks: bool = False,
) -> list[str]:
    target_indices = [target for _label, target in branch_targets]
    lines = [_entry_chain_heading(chain + target_indices), ""]

    common_parts = _chain_turns(chain, entries, tlk, speaker_hint)
    rendered_variants: list[tuple[str, list[str]]] = []
    seen_variants: dict[str, int] = {}
    for label, target in branch_targets:
        variant_lines: list[str] = []
        target_turns = _chain_turns([target], entries, tlk, speaker_hint)
        for speaker, parts in _merge_turns(common_parts + target_turns):
            text = _paragraph(" ".join(parts))
            if speaker:
                variant_lines.append(f"**{_md_escape(speaker)}:** {text}")
            else:
                variant_lines.append(text)
        reply_lines = _reply_lines(
            entries[target],
            entries,
            replies,
            tlk,
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
    *,
    show_unresolved_checks: bool = False,
) -> list[str]:
    lines = [f"## Entry {index}", ""]
    speaker = _entry_speaker(entry, speaker_hint)
    text, notes = _split_designer_notes(_resolve_text(entry, tlk))
    line_text = _display_text(text or "[no text]", speaker)
    if speaker:
        lines.extend(_speaker_text_lines(speaker, line_text))
    else:
        lines.append(line_text)

    reply_lines = _reply_lines(entry, entries, replies, tlk, show_unresolved_checks=show_unresolved_checks)
    if reply_lines:
        lines.append("")
        lines.extend(reply_lines)
    return lines


def _reply_lines(
    entry: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
    *,
    show_unresolved_checks: bool = False,
) -> list[str]:
    linked_replies = _as_list(entry.get("RepliesList"))
    if not linked_replies:
        return []
    if len(linked_replies) == 1 and _is_trivial_continue_reply(linked_replies[0], replies, tlk):
        return []

    reply_lines: list[str] = []
    for link in linked_replies:
        reply_index = _index_from_link(link)
        if reply_index is None:
            continue
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
        prefix_tags: list[str] = []
        check_lines: list[str] = []
        if reply is not None:
            annotations.extend(_effect_lines(reply))
            check_lines = _reply_check_lines(reply, reply_text, entries, replies, tlk)
            prefix_tags.extend(_check_prefix_tags(check_lines, choice_text))
            annotations.extend(line for line in check_lines if not _check_prefix_tags_for_line(line, choice_text))
            if not check_lines:
                annotations.extend(_routed_entry_effect_lines(reply, entries, replies, tlk))
        elif show_unresolved_checks and _tag_without_check_line(reply_text, link, reply):
            annotations.append(_tag_without_check_line(reply_text, link, reply))
        if show_unresolved_checks and reply is not None and not check_lines and not visibility_lines:
            tag_line = _tag_without_check_line(reply_text, link, reply)
            if tag_line:
                annotations.append(tag_line)
        annotations = list(dict.fromkeys(annotations))
        detail = f" [{'; '.join(annotations)}]" if annotations else ""
        suffix = _choice_text(choice_text, prefix_tags)
        reply_lines.append(f"- {suffix}{detail}")
    return reply_lines


def _routed_entry_effect_lines(
    reply: GffStruct,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> list[str]:
    effects: list[str] = []
    for entry_link in _as_list(reply.get("EntriesList")):
        entry_index = _index_from_link(entry_link)
        if entry_index is None or not (0 <= entry_index < len(entries)):
            continue
        effects.extend(_automatic_route_effect_lines(entry_index, entries, replies, tlk))
    return list(dict.fromkeys(effects))


def _automatic_route_effect_lines(
    start_index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> list[str]:
    effects: list[str] = []
    seen: set[int] = set()
    current = start_index
    while 0 <= current < len(entries) and current not in seen:
        seen.add(current)
        entry = entries[current]
        effects.extend(_effect_lines(entry))

        links = _as_list(entry.get("RepliesList"))
        if len(links) != 1:
            break
        link = links[0]
        if _link_detail_lines(link) or _visibility_check_lines(link, ""):
            break

        reply_index = _index_from_link(link)
        if reply_index is None or not (0 <= reply_index < len(replies)):
            break
        reply = replies[reply_index]
        reply_text, notes = _split_designer_notes(_resolve_text(reply, tlk))
        if notes or (reply_text and reply_text.lower() != "[continue]"):
            break

        effects.extend(_effect_lines(reply))
        if _reply_check_lines(reply, reply_text, entries, replies, tlk):
            break

        next_links = _as_list(reply.get("EntriesList"))
        if len(next_links) != 1 or _link_detail_lines(next_links[0]):
            break
        next_index = _index_from_link(next_links[0])
        if next_index is None:
            break
        current = next_index
    return list(dict.fromkeys(effects))


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
) -> list[str]:
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
        success_checks.sort(key=lambda item: int(item[0]["dc"]), reverse=True)
        for check, entry_index in success_checks:
            condition = _outcome_check_condition(check, reply_text)
            lines.append(f"{condition}: {_entry_outcome(entry_index, entries, replies, tlk)}")
        if fallback_entries:
            lines.append(f"otherwise: {_entry_outcomes(fallback_entries, entries, replies, tlk)}")
        for check, entry_index in lt_checks:
            lines.append(f"DC {check['dc']}")
            lines.append(f"failure: {_entry_outcome(entry_index, entries, replies, tlk)}")
        for check, entry_index in other_checks:
            lines.append(_skill_check_label(check))
            lines.append(f"success: {_entry_outcome(entry_index, entries, replies, tlk)}")
        return lines

    for check, entry_index in gt_checks:
        lines.append(_outcome_check_condition(check, reply_text))
        lines.append(f"success: {_entry_outcome(entry_index, entries, replies, tlk)}")
        if fallback_entries:
            lines.append(f"failure: {_entry_outcomes(fallback_entries, entries, replies, tlk)}")

    for check, entry_index in lt_checks:
        lines.append(_outcome_check_condition(check, reply_text))
        if fallback_entries:
            lines.append(f"success: {_entry_outcomes(fallback_entries, entries, replies, tlk)}")
        lines.append(f"failure: {_entry_outcome(entry_index, entries, replies, tlk)}")

    for check, entry_index in other_checks:
        lines.append(_skill_check_label(check))
        lines.append(f"success: {_entry_outcome(entry_index, entries, replies, tlk)}")
        if fallback_entries:
            lines.append(f"failure: {_entry_outcomes(fallback_entries, entries, replies, tlk)}")
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
    return [label.strip(), dc_tag]


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
) -> str:
    return ", ".join(_entry_outcome(index, entries, replies, tlk) for index in indices)


def _entry_outcome(
    index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> str:
    summary = ""
    if 0 <= index < len(entries):
        summary = _outcome_summary(index, entries, replies, tlk)
    return summary or f"Entry {index}"


def _outcome_summary(
    index: int,
    entries: list[GffStruct],
    replies: list[GffStruct],
    tlk: TlkTable,
) -> str:
    effects = _automatic_route_effect_lines(index, entries, replies, tlk)
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
    return bool(_reply_lines(entry, entries, replies, tlk))


def _is_trivial_end_continue(link: GffStruct, replies: list[GffStruct], tlk: TlkTable) -> bool:
    if _link_detail_lines(link) or _visibility_check_lines(link, ""):
        return False
    reply_index = _index_from_link(link)
    if reply_index is None or not (0 <= reply_index < len(replies)):
        return False

    reply = replies[reply_index]
    reply_text, notes = _split_designer_notes(_resolve_text(reply, tlk))
    if notes:
        return False
    if reply_text and reply_text.lower() != "[continue]":
        return False
    if _effect_lines(reply) or _reply_check_lines(reply, reply_text, [], replies, tlk):
        return False
    return not _as_list(reply.get("EntriesList"))


def _is_trivial_continue_reply(link: GffStruct, replies: list[GffStruct], tlk: TlkTable) -> bool:
    if _link_detail_lines(link) or _visibility_check_lines(link, ""):
        return False
    reply_index = _index_from_link(link)
    if reply_index is None or not (0 <= reply_index < len(replies)):
        return False

    reply = replies[reply_index]
    reply_text, notes = _split_designer_notes(_resolve_text(reply, tlk))
    if notes:
        return False
    if reply_text and reply_text.lower() != "[continue]":
        return False
    if _effect_lines(reply) or _reply_check_lines(reply, reply_text, [], replies, tlk):
        return False
    return True


def _entry_speaker(entry: GffStruct, speaker_hint: str) -> str:
    speaker = _plain_text(entry.get("Speaker"))
    if speaker:
        return _pretty_label(speaker)
    return speaker_hint


def _conversation_speaker_hint(resource: DialogueResource, root: GffStruct) -> str:
    vo_id = _plain_text(root.get("VO_ID"))
    if vo_id:
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
    known = {
        "909sion": "Sion",
        "atton": "Atton",
        "admoff": "Administration Officer",
        "adm_console": "Administration Console",
        "atristemp": "Atris",
        "atriscut": "Atris",
        "b4d4": "B-4D4",
        "b5d8": "B-5D8",
        "baodur": "Bao-Dur",
        "bao_dur": "Bao-Dur",
        "bh_rodian": "Rodian Bounty Hunter",
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
        "thgd": "Exchange Thug",
        "t1n1": "T1-N1",
        "tobin": "Colonel Tobin",
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
    while True:
        match = re.match(r"\s*\[[^\]]+\]\s*", stripped[position:])
        if not match:
            break
        position += match.end()

    prefix = stripped[:position].rstrip()
    rest = stripped[position:].lstrip()
    inserted = " ".join(f"[{tag}]" for tag in prefix_tags)
    if prefix:
        return f"{prefix} {inserted} {rest}".strip()
    return f"{inserted} {rest}".strip()


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
    for match in re.finditer(r"\{([^{}]+)\}", remaining):
        note = match.group(1).strip()
        if note:
            notes.append(note)
    remaining = re.sub(r"[ \t]*\{[^{}]+\}[ \t]*", " ", remaining)
    remaining = re.sub(r"[ \t]+", " ", remaining).strip()
    remaining = re.sub(r"[ \t]+([,.;:?!])", r"\1", remaining)
    return remaining, notes


def _md_escape(text: str) -> str:
    return text.replace("\\", "\\\\")


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._") or "unknown"
