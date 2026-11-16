# Negative Results for Reciprocity-Based Over-the-Air Split Learning Under Channel Aging: Stability, Alignment, and Geometry

## Requirements

Python 3.9 on CPU. Install with `python -m pip install -r requirements.txt`.
All experiments use synthetic data; no third-party datasets or pretrained models are required.

## Reproduction

Run `bash run_all.sh quick` for the available numerical checks (under a second on the tested CPU). Run `bash run_all.sh figure-b` to generate the two-panel figure in `output/` from the archived figure summaries. Intervals in the checks and figure are archived, not independently bootstrapped. D/E simulations take approximately 3–10 seconds each on the tested CPU and write approximately 450 MB in total.

| Paper item | Command | Scope |
|---|---|---|
| Table A, A text | `bash run_all.sh a` | Selected recorded analytic outputs and logged SGD results |
| Figure B, B text | `bash run_all.sh figure-b`; `bash run_all.sh b` | Figure from logged summaries; selected numeric claims checked |
| Table C | `bash run_all.sh c` | Seven logged seed means and archived intervals |
| Table D | `bash run_all.sh d1`; `bash run_all.sh d2`; `bash run_all.sh d3` | Full eight-seed simulations |
| Table E | `bash run_all.sh e1`; `bash run_all.sh e2` | Full eight-seed simulations |
| F text | `bash run_all.sh f` | Selected logged results |
| Table H, G text | `bash run_all.sh g` | Selected logged seed means and archived intervals |
| C8–C10 and remaining text/abstract | `bash run_all.sh c8` (also `c9`, `c10`) | Not checked |

The A–G experiment snapshots in `experiments/` provide the computational source and configurations. Except D/E, these are archival sources, **not independently runnable release pipelines**: some retain freeze-manifest, setup, or upstream-run dependencies not shipped here. The lettered targets above check shipped logs, not raw reruns. `c1`–`c7` all run the Table C check; `c8`–`c10` report their unchecked status. Unchecked claims should not be inferred from a successful quick check. The A proxy values are recorded intermediate outputs rather than an independent matrix recomputation. C6's archived ZF-SIC and MMSE-SIC values coincide; the value check alone cannot establish attribution.

Citation: arXiv:2610.37188.
