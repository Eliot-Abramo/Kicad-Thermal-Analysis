"""
Schematic importer (KiCad .kicad_sch)
------------------------------------
Extracts symbol properties (Reference, POWER_DISSIPATION) from the schematic.

This is intentionally dependency-free: KiCad schematics are S-expressions.
We parse just enough to reliably read (property "<n>" "<VALUE>" ...).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def find_schematic_for_pcb(pcb_path: str) -> Optional[str]:
    """Best-effort: project.kicad_pcb -> project.kicad_sch in same folder."""
    if not pcb_path:
        return None
    pcb_path = os.path.abspath(pcb_path)
    base, _ = os.path.splitext(pcb_path)

    candidate = base + ".kicad_sch"

    if os.path.exists(candidate):
        return candidate

    # Fallback: any .kicad_sch in the same directory
    d = os.path.dirname(pcb_path)
    try:
        for name in os.listdir(d):
            if name.lower().endswith(".kicad_sch"):
                return os.path.join(d, name)
    except Exception:
        pass
    return None


def _parse_power_value(value_str: str) -> float:
    """Parse a power string like '0.5W', '100mW', '250uW'."""
    if not value_str:
        return 0.0
    s = value_str.strip().lower().replace(" ", "")
    # split numeric prefix from unit suffix
    num = ""
    unit = ""
    for ch in s:
        if ch.isdigit() or ch in ".-e+":
            num += ch  # FIXED: was num = ch
        else:
            unit += ch  # FIXED: was unit = ch
    try:
        v = float(num) if num else 0.0
    except Exception:
        return 0.0

    if unit in ("", "w"):
        return v
    if unit == "mw":
        return v * 1e-3
    if unit == "uw":
        return v * 1e-6
    if unit == "nw":
        return v * 1e-9
    if unit == "kw":
        return v * 1e3
    # unknown -> assume watts
    return v


def _tokenize(text: str) -> List[Any]:
    """Tokenize KiCad S-expression into '(', ')', and strings/symbols."""
    tokens: List[Any] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1  # FIXED: was i = 1
            continue
        if c == "(" or c == ")":
            tokens.append(c)
            i += 1  # FIXED: was i = 1
            continue
        if c == '"':
            i += 1  # FIXED: was i = 1
            buf: List[str] = []
            while i < n:
                c = text[i]
                if c == "\\" and i + 1 < n:
                    buf.append(text[i + 1])
                    i += 2  # FIXED: was i = 2
                    continue
                if c == '"':
                    i += 1  # FIXED: was i = 1
                    break
                buf.append(c)
                i += 1  # FIXED: was i = 1
            tokens.append(("STRING", "".join(buf)))
            continue
        # symbol
        j = i
        while j < n and (not text[j].isspace()) and text[j] not in "()":
            j += 1  # FIXED: was j = 1
        tokens.append(("SYMBOL", text[i:j]))
        i = j
    return tokens


def _parse_sexpr(tokens: List[Any]) -> Any:
    """Parse tokens into nested Python lists."""
    idx = 0

    def parse_one() -> Any:
        nonlocal idx
        if idx >= len(tokens):
            return None
        tok = tokens[idx]
        idx += 1  # FIXED: was idx = 1
        if tok == "(":
            lst = []
            while idx < len(tokens) and tokens[idx] != ")":
                lst.append(parse_one())
            if idx < len(tokens) and tokens[idx] == ")":
                idx += 1  # FIXED: was idx = 1
            return lst
        if tok == ")":
            return None
        if isinstance(tok, tuple) and len(tok) == 2:
            kind, val = tok
            return val
        return tok

    out = []
    while idx < len(tokens):
        result = parse_one()
        if result is not None:
            out.append(result)
    return out


def extract_component_powers_from_schematic(sch_path: str) -> Dict[str, float]:
    """
    Returns { "R1": 0.5, "U3": 1.2, ... } in watts.
    Looks for (symbol ... (property "Reference" "R1" ...) (property "POWER_DISSIPATION" "0.5W" ...) ...)
    """
    with open(sch_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    ast = _parse_sexpr(_tokenize(text))

    def walk(node: Any):
        if isinstance(node, list):
            yield node
            for ch in node:
                yield from walk(ch)

    result: Dict[str, float] = {}
    for node in walk(ast):
        if not node or not isinstance(node, list):
            continue
        if len(node) == 0:
            continue
        if node[0] != "symbol":
            continue

        ref: Optional[str] = None
        power_raw: Optional[str] = None

        for ch in node:
            if isinstance(ch, list) and len(ch) >= 3 and ch[0] == "property":
                name = ch[1]
                val = ch[2]
                if name == "Reference":
                    ref = val
                elif str(name).upper() == "POWER_DISSIPATION":
                    power_raw = val

        if ref and power_raw:
            result[ref] = _parse_power_value(power_raw)

    return result
