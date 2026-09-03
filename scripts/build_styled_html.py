"""Export notebooks/*.ipynb to standalone, custom-styled HTML files.

Why this exists: GitHub's hosted .ipynb renderer sanitizes embedded CSS in
markdown cells, so a notebook's typography and background can't be
controlled from within the .ipynb file itself when viewed there. This
script instead post-processes nbconvert's own HTML export - a file this
repo fully owns - with a "textbook" stylesheet (Times New Roman prose on a
warm parchment background, monospace code left untouched) that renders
identically in any browser, independent of any third-party renderer's
sanitization rules.

Run from anywhere:
    python scripts/build_styled_html.py

Regenerate after editing either notebook so the .html twins stay in sync.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / "venv" / "Scripts" / "python.exe"
NOTEBOOKS_DIR = REPO / "notebooks"
NOTEBOOK_NAMES = ["HTGR_Systems_Textbook.ipynb", "HTGR_Analysis.ipynb"]

CUSTOM_CSS = """
<style>
/* ===================== HTGR Textbook custom styling ===================== */
html, body, .container {
  background-color: #FAF6EC !important;
  color: #2B2620 !important;
  font-family: "Times New Roman", Times, Georgia, serif !important;
}
.text_cell_render, .rendered_html {
  font-family: "Times New Roman", Times, Georgia, serif !important;
  color: #2B2620 !important;
  font-size: 18px !important;
  line-height: 1.65 !important;
}
.text_cell_render h1, .rendered_html h1 {
  font-family: "Times New Roman", Times, Georgia, serif !important;
  color: #17304D !important;
  border-bottom: 2px solid #B8860B;
  padding-bottom: 0.25em;
}
.text_cell_render h2, .rendered_html h2,
.text_cell_render h3, .rendered_html h3,
.text_cell_render h4, .rendered_html h4 {
  font-family: "Times New Roman", Times, Georgia, serif !important;
  color: #1F3A5F !important;
  border-bottom: 1px solid #D8D0BC;
  padding-bottom: 0.2em;
}
.text_cell_render a, .rendered_html a { color: #7B241C !important; text-decoration: none; border-bottom: 1px dotted #7B241C; }
.text_cell_render a:hover, .rendered_html a:hover { color: #a52e21 !important; }
.text_cell_render strong, .rendered_html strong { color: #17304D; }
.text_cell_render blockquote, .rendered_html blockquote {
  border-left: 3px solid #B8860B; color: #4A4436; background: #F3EEDF; padding: 0.6em 1em; margin-left: 0;
}
.text_cell_render table, .rendered_html table, table.dataframe {
  font-family: "Times New Roman", Times, Georgia, serif !important;
  border-collapse: collapse !important;
  border: 1px solid #D8D0BC !important;
}
table.dataframe th, .text_cell_render table th, .rendered_html table th {
  background-color: #E9E0C8 !important; color: #17304D !important; border: 1px solid #D8D0BC !important;
}
table.dataframe td, .text_cell_render table td, .rendered_html table td {
  border: 1px solid #E4DDC9 !important;
}
table.dataframe tbody tr:nth-child(even) { background-color: #F3EEDF !important; }
table.dataframe tbody tr:nth-child(odd) { background-color: #FAF6EC !important; }

.cell { background-color: transparent !important; border: none !important; }
hr { border-top: 1px solid #D8D0BC; }

/* Code stays monospace - only the surrounding panel adopts the palette */
.input_area, .highlight, pre, code, .output_text, .output_stream, .CodeMirror {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace !important;
  font-size: 14px !important;
}
.input_area {
  background-color: #F0EBDD !important;
  border: 1px solid #D8D0BC !important;
  border-radius: 5px;
}
.output_area pre, .output_stream {
  background-color: #F5F1E6 !important;
  color: #2B2620 !important;
  border-radius: 4px;
  padding: 4px 8px;
}
.prompt { font-family: "SFMono-Regular", Consolas, monospace !important; color: #8A7F63 !important; }
</style>
"""


def build(nb_name: str) -> None:
    nb_path = NOTEBOOKS_DIR / nb_name
    out_html = nb_path.with_suffix(".html")

    subprocess.run(
        [
            str(PYTHON), "-m", "jupyter", "nbconvert", "--to", "html", "--template", "classic",
            str(nb_path), "--output", out_html.name,
        ],
        cwd=str(NOTEBOOKS_DIR),
        check=True,
    )

    html = out_html.read_text(encoding="utf-8")
    if "</head>" not in html:
        raise RuntimeError(f"Unexpected nbconvert output for {nb_name}: no </head> tag found")
    html = html.replace("</head>", CUSTOM_CSS + "</head>", 1)
    out_html.write_text(html, encoding="utf-8")
    print(f"Wrote styled {out_html.relative_to(REPO)} ({out_html.stat().st_size:,} bytes)")


if __name__ == "__main__":
    for name in NOTEBOOK_NAMES:
        build(name)
