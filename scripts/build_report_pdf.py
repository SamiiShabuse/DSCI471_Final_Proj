"""Build print-ready HTML and PDF from docs/reports/final_report.md."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = PROJECT_ROOT / "docs" / "reports" / "final_report.md"
REPORT_HTML = PROJECT_ROOT / "docs" / "reports" / "final_report.html"
REPORT_PDF = PROJECT_ROOT / "docs" / "reports" / "final_report.pdf"
FIGURES_DIR = PROJECT_ROOT / "docs" / "reports" / "figures"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&family=Source+Sans+3:wght@400;600;700&display=swap');

:root {
  --text: #1a1a1a;
  --muted: #555;
  --accent: #1e40af;
  --border: #ddd;
  --bg: #fff;
}

* { box-sizing: border-box; }

body {
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 11pt;
  line-height: 1.55;
  color: var(--text);
  max-width: 7.5in;
  margin: 0 auto;
  padding: 0.75in 0.85in;
  background: var(--bg);
}

h1, h2, h3, h4 {
  font-family: 'Source Sans 3', Arial, sans-serif;
  line-height: 1.25;
  page-break-after: avoid;
}

h1 { font-size: 22pt; margin-top: 0; border-bottom: 2px solid var(--accent); padding-bottom: 0.25em; }
h2 { font-size: 15pt; margin-top: 1.4em; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 0.15em; }
h3 { font-size: 12pt; margin-top: 1.1em; }
h4 { font-size: 11pt; }

p, li { orphans: 3; widows: 3; }
blockquote {
  margin: 1em 0;
  padding: 0.6em 1em;
  border-left: 4px solid var(--accent);
  background: #f8fafc;
  font-style: italic;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 1em 0;
  font-size: 9.5pt;
  page-break-inside: avoid;
}

th, td {
  border: 1px solid var(--border);
  padding: 0.35em 0.5em;
  text-align: left;
}

th { background: #f1f5f9; font-family: 'Source Sans 3', sans-serif; font-weight: 600; }

code, pre {
  font-family: Consolas, 'Courier New', monospace;
  font-size: 9pt;
}

pre {
  background: #f8fafc;
  border: 1px solid var(--border);
  padding: 0.75em;
  overflow-x: auto;
  page-break-inside: avoid;
}

figure {
  margin: 1.2em 0;
  page-break-inside: avoid;
  text-align: center;
}

figure img {
  max-width: 100%;
  height: auto;
  border: 1px solid var(--border);
}

figcaption {
  font-family: 'Source Sans 3', sans-serif;
  font-size: 9pt;
  color: var(--muted);
  margin-top: 0.4em;
}

hr { border: none; border-top: 1px solid var(--border); margin: 1.5em 0; }

a { color: var(--accent); }

.title-meta {
  font-family: 'Source Sans 3', sans-serif;
  color: var(--muted);
  margin-bottom: 1.5em;
}

@media print {
  body { padding: 0; max-width: none; }
  a { color: var(--text); text-decoration: none; }
  h2 { page-break-before: auto; }
}
"""


def md_to_html(md_text: str) -> str:
    try:
        import markdown
        from markdown.extensions.tables import TableExtension
    except ImportError:
        print("Installing markdown...", file=sys.stderr)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown", "-q"])
        import markdown
        from markdown.extensions.tables import TableExtension

    # Pre-process figure blocks: ![caption](path) stays as-is for markdown
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )
    return html_body


def build_html() -> str:
    md_text = REPORT_MD.read_text(encoding="utf-8")
    body = md_to_html(md_text)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Final Report — Multimodal Fashion Search</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def try_pdf_from_html(html_path: Path, pdf_path: Path) -> bool:
    """Attempt PDF via weasyprint, then playwright."""
    html_content = html_path.read_text(encoding="utf-8")

    try:
        from weasyprint import HTML
        HTML(string=html_content, base_url=str(html_path.parent)).write_pdf(str(pdf_path))
        return True
    except Exception as exc:
        print(f"  weasyprint unavailable: {exc}")

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.as_uri())
            page.pdf(path=str(pdf_path), format="Letter", margin={"top": "0.75in", "bottom": "0.75in", "left": "0.85in", "right": "0.85in"}, print_background=True)
            browser.close()
        return True
    except Exception as exc:
        print(f"  playwright unavailable: {exc}")

    return False


def main() -> None:
    print("Building HTML report...")
    html = build_html()
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"  wrote {REPORT_HTML.relative_to(PROJECT_ROOT)}")

    print("Attempting PDF export...")
    if try_pdf_from_html(REPORT_HTML, REPORT_PDF):
        print(f"  wrote {REPORT_PDF.relative_to(PROJECT_ROOT)}")
    else:
        print("  PDF not generated automatically.")
        print("  Open final_report.html in a browser -> Print -> Save as PDF.")


if __name__ == "__main__":
    main()
