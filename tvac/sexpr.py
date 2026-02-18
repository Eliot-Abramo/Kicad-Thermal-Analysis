"""
Minimal, robust S-expression parser for KiCad .kicad_pcb files.

Design goals:
- Accept KiCad 6-9 board files
- Preserve strings exactly (including spaces when quoted)
- Avoid recursion blowups on very large zone polygons (iterative stack parser)

This module returns a nested Python structure:
- atoms are strings
- lists are Python lists
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Union, Optional

Atom = str
SExpr = Union[Atom, List["SExpr"]]


def tokenize(text: str) -> Iterator[str]:
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c == '(' or c == ')':
            yield c
            i += 1
            continue
        if c == '"':
            # quoted string with escapes
            i += 1
            out = []
            while i < n:
                c = text[i]
                if c == '"':
                    i += 1
                    break
                if c == '\\' and i + 1 < n:
                    # Keep escaped char as-is (KiCad uses \" and \\)
                    i += 1
                    out.append(text[i])
                    i += 1
                    continue
                out.append(c)
                i += 1
            yield ''.join(out)
            continue
        # bare atom
        j = i
        while j < n and (not text[j].isspace()) and text[j] not in '()':
            j += 1
        yield text[i:j]
        i = j


def parse(text: str) -> SExpr:
    """
    Parse S-expression text into a nested list.
    Uses an explicit stack to avoid recursion depth limits.
    """
    stack: List[List[SExpr]] = []
    cur: List[SExpr] = []
    for tok in tokenize(text):
        if tok == '(':
            stack.append(cur)
            new_list: List[SExpr] = []
            cur.append(new_list)
            cur = new_list
        elif tok == ')':
            if not stack:
                # tolerate stray ')'
                continue
            cur = stack.pop()
        else:
            cur.append(tok)
    # the file root is the first element
    return cur[0] if cur else []


def is_list(x) -> bool:
    return isinstance(x, list)


def find(node: SExpr, tag: str) -> Optional[List[SExpr]]:
    if not isinstance(node, list):
        return None
    for c in node:
        if isinstance(c, list) and c and c[0] == tag:
            return c
    return None


def find_all(node: SExpr, tag: str) -> List[List[SExpr]]:
    out: List[List[SExpr]] = []
    if not isinstance(node, list):
        return out
    for c in node:
        if isinstance(c, list):
            if c and c[0] == tag:
                out.append(c)
            out.extend(find_all(c, tag))
    return out
