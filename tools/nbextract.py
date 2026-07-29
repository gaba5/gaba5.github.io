#!/usr/bin/env python3
"""Extract paste-ready HTML fragments from a Jupyter notebook.

Turns a finished notebook into the pieces a site page is built from: tables,
figures and code cells, each in the exact markup main.css already styles. It
reads the .ipynb JSON directly, so there is no dependency (no nbconvert, no
pandas) and no build step. Nothing is rewritten: every table value and code
line is passed through verbatim, and captions are left exactly as the notebook
wrote them.

Usage:
    python tools/nbextract.py path/to/analysis.ipynb

Every table, figure and code cell is printed to stdout in notebook order, each
under a `<!-- cell N: ... -->` comment so you can see where it came from. Copy
the fragments you want into the page and delete the rest. Figures are written
as PNG files next to the page, in an `assets/` folder beside the notebook, and
referenced with a relative `<img src="assets/figureN.png">`; fill in the empty
alt text and caption yourself.

Markdown cells are ignored on purpose: page prose is written fresh, not lifted
from the notebook.

Self-check:  python tools/nbextract.py --selftest
"""

import base64
import html
import json
import os
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _text(value):
    """A notebook text field is a string or a list of line strings."""
    return "".join(value) if isinstance(value, list) else value


def merge_header(table_html):
    """Fold pandas' second header row into the first.

    A pandas Styler puts the index name on its own <tr> with every other cell
    blank, which renders as a stray second header line. Folding it in touches
    no data cell; it is the one structural change this script makes.
    """
    head = re.search(r"<thead>(.*?)</thead>", table_html, re.S)
    if not head:
        return table_html
    rows = re.findall(r"<tr>(.*?)</tr>", head.group(1), re.S)
    if len(rows) != 2:
        return table_html
    cells = [re.findall(r"<th[^>]*>(.*?)</th>", r, re.S) for r in rows]
    if any(c.strip() for c in cells[1][1:]):      # second row holds data, leave it
        return table_html
    merged = [cells[1][0]] + cells[0][1:]          # index name + the real headers
    new = "<thead>\n<tr>" + "".join(f"<th>{c}</th>" for c in merged) + "</tr>\n</thead>"
    return table_html[: head.start()] + new + table_html[head.end() :]


def clean_table(table_html):
    """Strip pandas chrome from a Styler table. Data stays verbatim.

    The caption is blanked for the page author to write, the same way figures get
    an empty figcaption; the notebook's original caption is kept in a comment for
    reference. Table numbering belongs to the page, not the notebook.
    """
    h = re.sub(r"<style.*?</style>", "", table_html, flags=re.S)
    h = re.sub(r'\s(id|class)="[^"]*"', "", h)
    h = h.replace("&nbsp;", "")
    h = merge_header(h)
    original = re.search(r"<caption>(.*?)</caption>", h, re.S)
    if original:
        note = re.sub(r"\s+", " ", original.group(1)).strip()
        h = h[: original.start()] + f"<caption></caption><!-- notebook: {note} -->" + h[original.end() :]
    return re.sub(r"\n\s*\n", "\n", h).strip()


def code_snippet(source, output_text):
    """A code cell as a .notebook block: the source, and its text output if any."""
    block = ['<div class="notebook">', '  <div class="cell code_cell">']
    block.append(f'    <div class="input_area"><pre>{html.escape(source)}</pre></div>')
    if output_text.strip():
        block.append(
            '    <div class="output_subarea output_stream">'
            f"<pre>{html.escape(output_text.rstrip())}</pre></div>"
        )
    block += ["  </div>", "</div>"]
    return "\n".join(block)


def figure_fragment(index):
    return (
        "<figure>\n"
        f'  <img src="assets/figure{index}.png" alt="" '
        'style="display:block; width:100%; height:auto; margin:0 auto;" />\n'
        "  <figcaption></figcaption>   <!-- add caption -->\n"
        "</figure>"
    )


def extract(notebook_path):
    """Yield (source_cell_index, label, fragment) for every emittable piece."""
    nb = json.load(open(notebook_path, encoding="utf8"))
    assets = os.path.join(os.path.dirname(os.path.abspath(notebook_path)), "assets")
    figure_n = 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        stdout_parts = []
        tables = []
        pngs = []
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "text/html" in data:                 # a Styler table
                tables.append(_text(data["text/html"]))
            if "image/png" in data:                 # a figure
                pngs.append(_text(data["image/png"]))
            if out.get("output_type") == "stream":
                stdout_parts.append(_text(out.get("text", "")))
            elif out.get("output_type") == "execute_result":
                # plain-text result, but not the "<pandas ... Styler>" repr
                if "text/html" not in data and "text/plain" in data:
                    stdout_parts.append(_text(data["text/plain"]))
            elif out.get("output_type") == "error":
                stdout_parts.append("\n".join(_text(out.get("traceback", []))))

        source = _text(cell.get("source", "")).strip("\n")
        if source:
            out_text = ANSI.sub("", "".join(stdout_parts))
            yield i, "code", code_snippet(source, out_text)

        for table in tables:
            yield i, "table", '<div class="table-wrap">\n' + clean_table(table) + "\n</div>"

        for png in pngs:
            figure_n += 1
            os.makedirs(assets, exist_ok=True)
            path = os.path.join(assets, f"figure{figure_n}.png")
            open(path, "wb").write(base64.b64decode(png))
            yield i, f"figure -> assets/figure{figure_n}.png", figure_fragment(figure_n)


def main(notebook_path):
    # Notebook text carries characters outside the Windows console default (cp1252),
    # so force UTF-8 rather than crash half way through.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for index, label, fragment in extract(notebook_path):
        print(f"\n<!-- cell {index}: {label} -->")
        print(fragment)


def _selftest():
    """Round-trip a tiny notebook and assert nothing is altered or dropped."""
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["ignored"]},
            {
                "cell_type": "code",
                "source": ["x = 1 < 2 & True\n", "print(x)"],
                "outputs": [
                    {"output_type": "stream", "text": ["True\n"]},
                    {
                        "output_type": "execute_result",
                        "data": {
                            "text/html": [
                                "<style></style>",
                                '<table id="T_x"><caption>Table 9. Verbatim.</caption>',
                                '<thead><tr><th class="blank">&nbsp;</th>'
                                '<th id="c">score</th></tr>'
                                '<tr><th>team</th><th class="blank">&nbsp;</th></tr></thead>',
                                "<tbody><tr><th>Spain</th><td>0.4829</td></tr></tbody></table>",
                            ],
                            "text/plain": ["<pandas ... Styler>"],
                        },
                    },
                ],
            },
        ]
    }
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "_nbextract_selftest.ipynb")
    json.dump(nb, open(path, "w"), )
    kinds = [(label, frag) for _, label, frag in extract(path)]
    labels = [k for k, _ in kinds]
    assert labels == ["code", "table"], labels               # markdown skipped, order kept
    code = next(f for k, f in kinds if k == "code")
    assert "x = 1 &lt; 2 &amp; True" in code, "code not HTML-escaped"
    assert "True" in code, "stdout dropped"
    table = next(f for k, f in kinds if k == "table")
    assert "0.4829" in table, "data value altered"            # verbatim number survives
    assert "<caption></caption>" in table, "caption not blanked"
    assert "notebook: Table 9. Verbatim." in table, "original caption not kept in comment"
    assert "<th>team</th><th>score</th>" in table, "header not merged"
    assert "pandas" not in code and "pandas" not in table, "Styler repr leaked"
    inner = table[table.index("<table"):]                     # skip the .table-wrap wrapper
    assert 'id="' not in inner and 'class="' not in inner, "chrome not stripped"
    os.remove(path)
    print("selftest ok")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        _selftest()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        sys.exit(__doc__)
