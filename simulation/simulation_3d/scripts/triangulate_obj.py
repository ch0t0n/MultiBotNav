#!/usr/bin/env python3
"""Triangulate n-gon faces in OBJ files for Ursina compatibility."""

from __future__ import annotations

import argparse
import os
import sys


def _parse_face_vertex(token: str) -> str:
    return token.split("/")[0]


def _simple_face_tokens(parts: list[str]) -> list[str]:
    """OBJ face indices only (no vt/vn) for Ursina compatibility."""
    return [_parse_face_vertex(p) for p in parts]


def triangulate_obj_text(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        if not line.startswith("f "):
            out_lines.append(line)
            continue
        parts = line.strip().split()[1:]
        verts = _simple_face_tokens(parts)
        if len(verts) < 3:
            continue
        if len(verts) == 3:
            out_lines.append("f " + " ".join(verts))
            continue
        anchor = verts[0]
        for i in range(1, len(verts) - 1):
            tri = [anchor, verts[i], verts[i + 1]]
            out_lines.append("f " + " ".join(tri))
    return "\n".join(out_lines) + "\n"


def triangulate_obj_file(src: str, dst: str) -> None:
    with open(src, "r", encoding="utf-8") as f:
        text = f.read()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(triangulate_obj_text(text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst", nargs="?")
    args = parser.parse_args()
    dst = args.dst or args.src.replace(".obj", "_ursina.obj")
    triangulate_obj_file(args.src, dst)
    print(f"Wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
