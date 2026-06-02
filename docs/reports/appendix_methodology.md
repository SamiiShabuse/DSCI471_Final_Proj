# Appendix: Methodology Rationale and Suggested Small Ablations

This appendix expands the methodological justification and lists reproducible small ablations that support design choices.

1) Backbone choice: EfficientNetB0
- Rationale: good accuracy-to-compute tradeoff for course-scale experiments; smaller memory footprint makes reproducible CPU runs practical.
- Evidence: final model uses EfficientNetB0 in `src/train.py` and `src/model.py`.

Suggested small ablation (fast sample): compare EfficientNetB0 vs MobileNetV2 for 1 epoch on 800-sample subset (see `scripts/run_sample_pipeline.py`). Record validation R@1/R@5 in `docs/reports/ablations/backbone_sample.csv`.

2) Text encoder decision: frozen `all-MiniLM-L6-v2`
- Rationale: compact sentence embeddings produce strong semantic features with low compute; freezing avoids expensive text fine-tuning and instability on small course GPUs.
- Evidence: use of `sentence-transformers/all-MiniLM-L6-v2` in `src/config.py` and `src/train.py`.

Suggested small ablation: unfreeze MiniLM for 1-2 epochs on sample dataset; compare validation R@1 vs frozen variant and report delta. Use `scripts/run_sample_pipeline.py` to produce sample artifacts.

3) Embedding dimension and temperature
- `TEXT_DIM=384` and `TEMPERATURE=0.07` are set in `src/config.py`. These are standard CLIP-like choices; to justify, run a grid of {256,384,512} embedding dims on samples or report literature citations (CLIP, Sentence-BERT) in the report.

4) Contrastive loss and batch size
- In-course constraints set `BATCH_SIZE=64`. Larger batch sizes improve contrastive negatives; note this as a limitation in the main report and propose larger-batch experiments as future work.

References to committed ablations
- See `docs/reports/ablations/v1_v2_v3_v4_comparison.csv` and `docs/reports/ablations/final_comparison.csv` for validation results that motivated v4 selection.

Implementation notes
- Add `scripts/ablate_backbones.py` and `scripts/ablate_text_ft.py` to run small-sample ablations; they should use `--sample` CLI flags so they run quickly for graders.
