"""Validation for the results-filling pass (step 8): brace, $ and environment balance per file,
unresolved \\ref and \\cite keys, dashes, remaining markers. Read-only on the paper."""
import glob
import os
import re
import sys

PAPER = sys.argv[1] if len(sys.argv) > 1 else r"C:/Users/msc/Desktop/G00403092/JournalPaper"
COMMENT = re.compile(r"(?<!\\)%.*")


def strip(text):
    return "\n".join(COMMENT.sub("", line) for line in text.splitlines())


files = sorted(glob.glob(os.path.join(PAPER, "sections", "*.tex"))) + [os.path.join(PAPER, "main.tex")]
labels, refs, cites = set(), set(), set()
for f in files:
    raw = open(f, encoding="utf-8").read()
    body = strip(raw)
    braces = (body.count("{") - body.count(r"\{")) - (body.count("}") - body.count(r"\}"))
    dollars = len(re.findall(r"(?<!\\)\$", body))
    envs = {}
    for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", body):
        envs[name] = envs.get(name, 0) + (1 if kind == "begin" else -1)
    bad = {k: v for k, v in envs.items() if v}
    dash = ("---" in raw) or ("\u2014" in raw)
    print(f"{os.path.basename(f):28s} braces {braces:+d}  $ {dollars} ({'even' if dollars % 2 == 0 else 'ODD'})"
          f"  envs {bad or 'ok'}  dashes {'FOUND' if dash else 'none'}")
    labels |= set(re.findall(r"\\label\{([^}]+)\}", body))
    refs |= set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body))
    for c in re.findall(r"\\cite\{([^}]+)\}", body):
        cites |= {x.strip() for x in c.split(",")}
bib = set(re.findall(r"@\w+\{([^,\s]+),", open(os.path.join(PAPER, "references.bib"), encoding="utf-8").read()))
print(f"labels {len(labels)}; unresolved refs {sorted(refs - labels)}; "
      f"cites {len(cites)}; unresolved cites {sorted(cites - bib)}")
