# K2Dialog Dump

Small local CLI for dumping KOTOR II `.dlg` dialogue files to searchable Markdown.

```powershell
python -m k2dialog_dump dump --game-dir "C:\path\to\Knights of the Old Republic II" --out out
```

Optional output switches:

```powershell
python -m k2dialog_dump dump --game-dir ... --out out --single-file
python -m k2dialog_dump dump --game-dir ... --out out --by-module
python -m k2dialog_dump dump --game-dir ... --out out --by-dlg
python -m k2dialog_dump dump --game-dir ... --out out --quiet
```

By default, all outputs are written:

```text
out/
  all_dialogue.md
  by_module/
  by_dlg/
```

The dumper reads `dialog.tlk`, finds loose `.dlg` files in `Override/`, and extracts `.dlg` resources from `.mod` / `.erf` archives under `modules/`.

The output directory is regenerated on each run. The tool refuses obviously dangerous output locations, such as the game directory, your home directory, the current working directory, or a drive root.

For parser debugging, `--show-unresolved-checks` annotates skill-tagged options whose check script is not yet decoded. Normal output omits those diagnostics.
