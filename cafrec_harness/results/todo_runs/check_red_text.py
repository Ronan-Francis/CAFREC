"""Check that no red text remains in the journal paper (source and compiled PDF). Read-only.

Red text in this paper comes from the draft macros in main.tex, \\todores and \\todocite, both
\\textcolor{red}. The check has two layers:

1. Source: every .tex file (sections and main.tex) is scanned, comments stripped, for calls
   to \\todores or \\todocite and for any red colour command (\\textcolor{red}, \\color{red},
   \\colorbox{red}, xcolor mixes naming red). The two \\newcommand definitions in main.tex are
   reported separately; they print nothing unless called.
2. PDF: a copy of the paper is compiled (pdflatex, bibtex, pdflatex x2) in a scratch folder.
   Every content stream (images skipped) is decompressed and searched for pdfTeX's red colour
   operators, "1 0 0 rg" (fill) and "1 0 0 RG" (stroke), and their CMYK form "0 1 1 0 k/K".
   Where pypdf is installed, page text is also searched for the macros' printed prefixes
   "[RESULT:" and "[CITE:".

A positive control runs the same PDF check on a copy that still has markers, to show the
detector finds red text when it is there.

Usage: python check_red_text.py <paper_dir> [<positive_control_dir>] [--scratch <dir>]
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import zlib

MIKTEX = r"C:/Users/msc/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
COMMENT = re.compile(r"(?<!\\)%.*")
RED_SRC = re.compile(r"\\(textcolor|color|colorbox|fcolorbox|pagecolor)\s*(\[[^\]]*\])?\s*\{[^}]*\bred\b[^}]*\}")
MARKER_CALL = re.compile(r"\\(todores|todocite)\s*\{")
RED_PDF = re.compile(rb"(?<![\d.])(?:1 0 0 (?:rg|RG)|0 1 1 0 (?:k|K))(?![\w])")


def source_check(paper):
    hits, defs = [], []
    for f in sorted(glob.glob(os.path.join(paper, "sections", "*.tex"))) + [os.path.join(paper, "main.tex")]:
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            code = COMMENT.sub("", line)
            if "\\newcommand" in code and MARKER_CALL.search(code.replace("\\newcommand{\\", "{\\")) is None \
                    and re.search(r"\\newcommand\{\\(todores|todocite|todonum)\}", code):
                defs.append(f"{os.path.basename(f)}:{n}: {code.strip()}")
                continue
            for m in list(MARKER_CALL.finditer(code)) + list(RED_SRC.finditer(code)):
                hits.append(f"{os.path.basename(f)}:{n}: {m.group(0)}")
    return hits, defs


def compile_pdf(paper, scratch):
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    shutil.copytree(paper, scratch)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run([os.path.join(MIKTEX, cmd[0])] + cmd[1:], cwd=scratch,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(os.path.join(scratch, "main.log"), encoding="latin-1").read()
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    return os.path.join(scratch, "main.pdf"), {
        "errors": len(re.findall(r"^!", log, re.M)),
        "undefined": len(re.findall(r"(Reference|Citation) `[^']+' on page \d+ undefined", log)),
        "pages": int(pages.group(1)) if pages else None}


def pdf_check(pdf):
    raw = open(pdf, "rb").read()
    red_ops, streams = 0, 0
    for m in re.finditer(rb"<<(.*?)>>\s*stream\r?\n", raw, re.S):
        head = m.group(1)
        end = raw.find(b"endstream", m.end())
        data = raw[m.end():end]
        if b"/Image" in head:
            continue
        if b"/FlateDecode" in head:
            try:
                data = zlib.decompress(data)
            except zlib.error:
                continue
        streams += 1
        red_ops += len(RED_PDF.findall(data))
    text_hits = None
    try:
        from pypdf import PdfReader
        text_hits = []
        for i, page in enumerate(PdfReader(pdf).pages, 1):
            t = page.extract_text() or ""
            for tag in ("[RESULT:", "[CITE:"):
                if tag in t:
                    text_hits.append(f"page {i}: {tag}")
    except ImportError:
        pass
    return {"streams_scanned": streams, "red_colour_operators": red_ops, "text_hits": text_hits}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scratch = sys.argv[sys.argv.index("--scratch") + 1] if "--scratch" in sys.argv else os.path.join(
        os.environ.get("TEMP", "."), "red_check")
    paper = args[0]
    hits, defs = source_check(paper)
    print(f"SOURCE {paper}\n  marker calls or red colour commands: {len(hits)}")
    for h in hits:
        print("   ", h)
    print(f"  macro definitions (print nothing unless called): {len(defs)}")
    for d in defs:
        print("   ", d)
    pdf, log = compile_pdf(paper, os.path.join(scratch, "paper"))
    res = pdf_check(pdf)
    print(f"PDF {log} {res}")
    if len(args) > 1:
        ctrl_pdf, ctrl_log = compile_pdf(args[1], os.path.join(scratch, "control"))
        ctrl = pdf_check(ctrl_pdf)
        ctrl_hits, _ = source_check(args[1])
        print(f"POSITIVE CONTROL {args[1]}\n  source marker calls {len(ctrl_hits)}; PDF {ctrl_log} {ctrl}")
    ok = not hits and res["red_colour_operators"] == 0 and not (res["text_hits"] or [])
    print("RESULT:", "NO RED TEXT" if ok else "RED TEXT FOUND")


if __name__ == "__main__":
    main()
