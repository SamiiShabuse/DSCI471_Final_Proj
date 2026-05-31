"""Build HTML and PDF presentation from docs/presentation_slides.md content."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLIDES_HTML = PROJECT_ROOT / "docs" / "presentation.html"
SLIDES_PDF = PROJECT_ROOT / "docs" / "presentation.pdf"
FIGURES = PROJECT_ROOT / "docs" / "reports" / "figures"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Source Sans 3', 'Segoe UI', Arial, sans-serif;
  background: #0f172a;
  color: #f8fafc;
}

.deck { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }

.slide {
  background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
  border: 1px solid #334155;
  border-radius: 12px;
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  min-height: 520px;
  page-break-after: always;
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.slide h1 {
  font-size: 2rem;
  color: #93c5fd;
  margin-bottom: 0.5rem;
  line-height: 1.2;
}

.slide h2 {
  font-size: 1.6rem;
  color: #bfdbfe;
  margin-bottom: 1rem;
  padding-bottom: 0.4rem;
  border-bottom: 2px solid #2563eb;
}

.slide p, .slide li { font-size: 1.05rem; line-height: 1.55; color: #e2e8f0; }
.slide ul { margin: 0.75rem 0 0 1.5rem; }
.slide li { margin-bottom: 0.45rem; }

.slide blockquote {
  border-left: 4px solid #3b82f6;
  padding: 0.75rem 1rem;
  margin: 1rem 0;
  background: rgba(59,130,246,0.1);
  font-style: italic;
  color: #cbd5e1;
}

.slide table {
  width: 100%;
  border-collapse: collapse;
  margin: 1rem 0;
  font-size: 0.95rem;
}

.slide th, .slide td {
  border: 1px solid #475569;
  padding: 0.45rem 0.6rem;
  text-align: left;
}

.slide th { background: #334155; color: #f1f5f9; }

.slide img {
  max-width: 100%;
  max-height: 380px;
  display: block;
  margin: 1rem auto 0;
  border-radius: 8px;
  border: 1px solid #475569;
}

.slide pre {
  background: #0f172a;
  border: 1px solid #475569;
  padding: 0.75rem;
  border-radius: 6px;
  font-size: 0.85rem;
  overflow-x: auto;
  color: #94a3b8;
}

.subtitle { color: #94a3b8; font-size: 1.1rem; margin-bottom: 1.5rem; }
.footer { color: #64748b; font-size: 0.9rem; margin-top: 2rem; }

@media print {
  body { background: white; color: black; }
  .slide {
    background: white;
    color: black;
    border: 1px solid #ccc;
    box-shadow: none;
    min-height: auto;
  }
  .slide h1, .slide h2 { color: #1e40af; }
  .slide p, .slide li { color: #1a1a1a; }
}
"""

SLIDES = [
    {
        "title": "Multimodal Deep Learning Search Engine for E-commerce Fashion",
        "subtitle": "Richardson Chhin · Samii Shabuse · DSCI471 · May 2026",
        "body": "",
    },
    {
        "h2": "Problem & motivation",
        "body": """
<ul>
<li>Shoppers describe products in natural language — e.g. <em>"a flowy floral dress with cap sleeves"</em> — but catalogs rely on keywords.</li>
<li>We build a retriever that maps <strong>text queries → product images</strong> in a shared embedding space.</li>
<li>Dataset: <strong>44,265</strong> fashion products (Kaggle Fashion Product Images).</li>
</ul>""",
    },
    {
        "h2": "Research question",
        "body": """
<blockquote>Can a dual-encoder deep learning model, trained on paired fashion images and text, <strong>outperform traditional keyword-based e-commerce search</strong>?</blockquote>
<ul>
<li><strong>Baseline:</strong> TF-IDF on product_text (caption + JSON description).</li>
<li><strong>Proposed:</strong> Dual-encoder — text query → embedding ← product image embedding.</li>
</ul>""",
    },
    {
        "h2": "Dataset & preprocessing",
        "body": """
<table>
<tr><th>Item</th><th>Detail</th></tr>
<tr><td>Source</td><td>Kaggle Fashion Product Images (~44k)</td></tr>
<tr><td>Splits</td><td>80/10/10 train / val / test</td></tr>
<tr><td>Test gallery</td><td><strong>4,427</strong> products</td></tr>
</table>
<p>Pipeline: <code>src/prepare_data.py</code></p>""",
    },
    {
        "h2": "Query styles",
        "body": """
<table>
<tr><th>Style</th><th>Example</th></tr>
<tr><td>templated</td><td>A men navy blue shirts, for casual wear in fall...</td></tr>
<tr><td>shopper</td><td>men's navy blue shirt</td></tr>
<tr><td>brand</td><td>Turtle Check Men Navy Blue Shirt</td></tr>
<tr><td>short</td><td>navy shirt</td></tr>
</table>""",
    },
    {
        "h2": "Architecture (v4)",
        "body": """
<pre>Text query  →  MiniLM (frozen)  →  384-d embedding
                                        ↓ cosine similarity
Product image → EfficientNetB0 (fine-tuned) → 384-d embedding</pre>
<ul>
<li>Contrastive loss, τ = 0.07</li>
<li>Selected after v1→v5 ablations — pretrained text encoder = largest gain</li>
</ul>""",
    },
    {
        "h2": "Ablation progression (validation)",
        "img": "ablation_progression.png",
        "caption": "Recall@1 on templated queries — v4 (MiniLM) nearly doubled v1",
    },
    {
        "h2": "Results — test set (4,427 products)",
        "img": "test_metrics_comparison.png",
        "caption": "TF-IDF wins Top-1 on every query style",
    },
    {
        "h2": "Top-1 accuracy",
        "body": """
<table>
<tr><th>Query type</th><th>TF-IDF</th><th>Dual-encoder</th></tr>
<tr><td>templated</td><td><strong>0.52</strong></td><td>0.17</td></tr>
<tr><td>shopper</td><td><strong>0.08</strong></td><td>0.07</td></tr>
<tr><td>brand</td><td><strong>0.65</strong></td><td>0.14</td></tr>
<tr><td>short</td><td><strong>0.07</strong></td><td>0.05</td></tr>
</table>""",
    },
    {
        "h2": "Where dual-encoder wins (shopper)",
        "img": "shopper_metrics_comparison.png",
        "caption": "Higher Top-5 (0.24 vs 0.22) and MRR (0.167 vs 0.161)",
    },
    {
        "h2": "Demo — success (templated query)",
        "img": "demo_success_templated.png",
    },
    {
        "h2": "Demo — head-to-head (brand query)",
        "img": "demo_head_to_head_brand.png",
    },
    {
        "h2": "Demo — failure (shopper query)",
        "img": "demo_failure_shopper.png",
    },
    {
        "h2": "Limitations",
        "body": """
<ul>
<li><strong>Modality gap</strong> — text→image vs TF-IDF text→text</li>
<li><strong>Low Top-1</strong> — both models struggle on shopper/short (~5–8%)</li>
<li><strong>Synthetic queries</strong> — not real search logs</li>
<li><strong>Compute</strong> — ~40 min CPU training</li>
</ul>""",
    },
    {
        "h2": "Conclusion & future work",
        "body": """
<p><strong>Conclusion:</strong> Keyword search wins Top-1 when the gallery is text-indexed. Dual-encoder is viable for cross-modal retrieval and wins on shopper Top-5 / MRR.</p>
<p><strong>Future work:</strong> fine-tune text tower, hard negatives, CLIP-style pretraining, hybrid TF-IDF → dual reranking.</p>
<p class="footer">Full report: docs/reports/final_report.pdf · Repro: docs/GRADING.md</p>
<p class="footer"><strong>Thank you!</strong></p>""",
    },
]


def render_slide(slide: dict) -> str:
    parts = ['<section class="slide">']
    if "title" in slide:
        parts.append(f'<h1>{slide["title"]}</h1>')
        if slide.get("subtitle"):
            parts.append(f'<p class="subtitle">{slide["subtitle"]}</p>')
    if slide.get("h2"):
        parts.append(f'<h2>{slide["h2"]}</h2>')
    if slide.get("body"):
        parts.append(slide["body"])
    if slide.get("img"):
        rel = f"reports/figures/{slide['img']}"
        parts.append(f'<img src="{rel}" alt="{slide.get("h2", "figure")}">')
        if slide.get("caption"):
            parts.append(f'<p class="footer">{slide["caption"]}</p>')
    parts.append("</section>")
    return "\n".join(parts)


def build_html() -> str:
    body = "\n".join(render_slide(s) for s in SLIDES)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Multimodal Fashion Search — Presentation</title>
  <style>{CSS}</style>
</head>
<body>
<div class="deck">
{body}
</div>
</body>
</html>
"""


def export_pdf(html_path: Path, pdf_path: Path) -> bool:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            page.goto(html_path.as_uri(), wait_until="networkidle")
            page.pdf(
                path=str(pdf_path),
                format="A4",
                landscape=True,
                margin={"top": "0.4in", "bottom": "0.4in", "left": "0.5in", "right": "0.5in"},
                print_background=True,
            )
            browser.close()
        return True
    except Exception as exc:
        print(f"  PDF export failed: {exc}")
        return False


def main() -> None:
    html = build_html()
    SLIDES_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote {SLIDES_HTML.relative_to(PROJECT_ROOT)}")
    if export_pdf(SLIDES_HTML, SLIDES_PDF):
        print(f"Wrote {SLIDES_PDF.relative_to(PROJECT_ROOT)}")
    else:
        print("Open presentation.html in browser -> Print -> Save as PDF")


if __name__ == "__main__":
    main()
