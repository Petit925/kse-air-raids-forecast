"""Export the current Claude Code session JSONL as a clean readable txt.

Run:
    python -m src.export_transcript --in <path-to-session.jsonl> --out <path-to-output.txt>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def block_to_text(block: dict, compact: bool = False) -> str:
    btype = block.get("type")
    if btype == "text":
        return block.get("text", "")
    if btype == "image":
        return "[IMAGE]"
    if btype == "tool_use":
        name = block.get("name", "?")
        inp = block.get("input", {})
        if compact:
            # One-line per tool call in compact mode
            if name == "Bash" or name == "PowerShell":
                return f"[{name}: {inp.get('description','')}]"
            if name in ("Write", "Edit", "Read"):
                return f"[{name}: {Path(inp.get('file_path','')).name}]"
            if name == "TodoWrite":
                todos = inp.get("todos", [])
                inp_progress = [t.get('content','') for t in todos if t.get('status')=='in_progress']
                if inp_progress:
                    return f"[Todo: → {inp_progress[0]}]"
                return "[Todo update]"
            if name == "Agent":
                return f"[Agent: {inp.get('description','')}]"
            if name.startswith("mcp__playwright"):
                short = name.replace("mcp__playwright__browser_", "")
                return f"[browser.{short}]"
            return f"[{name}]"
        # Full mode below
        if name == "Bash" or name == "PowerShell":
            cmd = inp.get("command", "")
            desc = inp.get("description", "")
            return f"[TOOL {name}] {desc}\n    $ {cmd[:300]}"
        if name in ("Write", "Edit"):
            fp = inp.get("file_path", "")
            return f"[TOOL {name}] {fp}"
        if name == "Read":
            fp = inp.get("file_path", "")
            return f"[TOOL Read] {fp}"
        if name == "Grep":
            return f"[TOOL Grep] pattern={inp.get('pattern','')[:80]!r}"
        if name == "Glob":
            return f"[TOOL Glob] {inp.get('pattern','')}"
        if name == "TodoWrite":
            todos = inp.get("todos", [])
            lines = [f"  {('✓' if t.get('status')=='completed' else '▸' if t.get('status')=='in_progress' else '○')} {t.get('content','')}" for t in todos]
            return "[TOOL TodoWrite]\n" + "\n".join(lines)
        if name == "Agent":
            return f"[TOOL Agent] {inp.get('description','')}"
        if name.startswith("mcp__playwright"):
            short = name.replace("mcp__playwright__browser_", "browser.")
            return f"[TOOL {short}] {json.dumps({k:v for k,v in inp.items() if k != 'function'}, ensure_ascii=False)[:200]}"
        return f"[TOOL {name}] {json.dumps(inp, ensure_ascii=False)[:200]}"
    if btype == "tool_result":
        if compact:
            return ""  # skip tool results entirely in compact mode
        content = block.get("content")
        if isinstance(content, str):
            txt = content
        elif isinstance(content, list):
            parts = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    elif c.get("type") == "image":
                        parts.append("[IMAGE returned]")
                else:
                    parts.append(str(c))
            txt = "\n".join(parts)
        else:
            txt = str(content)
        if len(txt) > 2000:
            txt = txt[:1500] + f"\n... [truncated {len(txt) - 1500} chars]"
        return f"[TOOL RESULT]\n{txt}"
    return f"[block type={btype}]"


def format_message(record: dict, compact: bool = False) -> str | None:
    rtype = record.get("type")
    if rtype not in ("user", "assistant"):
        return None
    msg = record.get("message", {})
    role = msg.get("role", rtype)
    content = msg.get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    parts = [block_to_text(b, compact=compact) for b in content if isinstance(b, dict)]
    parts = [p for p in parts if p]
    if not parts:
        return None
    body = ("\n" if compact else "\n\n").join(parts)
    if compact:
        prefix = "USER:" if role == "user" else "CLAUDE:"
        return f"\n{prefix} {body}\n"
    ts = record.get("timestamp", "")
    header = f"{'='*78}\n[{ts}] {role.upper()}\n{'='*78}"
    return f"{header}\n{body}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--compact", action="store_true",
                    help="Skip tool results, one-line tool calls — fits Google-Forms char limits.")
    ap.add_argument("--max-chars", type=int, default=0,
                    help="Hard cap on output length (truncate with note). 0 = no cap.")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    formatted = []
    n_total = 0
    n_kept = 0
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = format_message(rec, compact=args.compact)
            if txt:
                formatted.append(txt)
                n_kept += 1

    header = (
        f"KSE AI Agentic Summer School — Stage 2\n"
        f"Air-raid alerts forecast + workforce planner\n"
        f"AI conversation log{' (compact, see repo for full)' if args.compact else ''}\n"
        f"Date: 2026-06-21\n"
        f"User: galabitskiy@gmail.com / Petit925\n"
        f"Repo: https://github.com/Petit925/kse-air-raids-forecast\n"
        f"Full log in repo: https://github.com/Petit925/kse-air-raids-forecast/blob/main/kse_ai_conversation_log.txt\n\n"
        f"Total session events: {n_total}, kept (user+assistant): {n_kept}\n"
        f"{'='*78}\n"
    )
    body = "\n".join(formatted)
    full = header + body
    if args.max_chars and len(full) > args.max_chars:
        cut = args.max_chars - 200
        full = full[:cut] + (
            f"\n\n... [TRUNCATED — {len(body) - cut + len(header)} chars removed to fit submission form. "
            f"Full log in repo URL above.]"
        )
    out.write_text(full, encoding="utf-8")
    print(f"Wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {n_kept} messages, {len(full):,} chars)")


if __name__ == "__main__":
    main()
