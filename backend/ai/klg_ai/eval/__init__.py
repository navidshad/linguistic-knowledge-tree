"""Phase 5 — validation on open data (Duolingo SLAM).

Runs the activation engine as a knowledge-tracing predictor on the Duolingo SLAM
en_es track and scores next-step correctness prediction against the shared-task
metrics, with the RQ ablations:

* **RQ1** — graph (engine) vs. sequence (in-repo DKT) vs. simple baselines.
* **RQ3** — GNN propagation on/off.
* **RQ4** — forgetting/decay on/off.

Submodules: ``metrics`` (SLAM-comparable AUROC/F1/acc/logloss), ``dataset``
(SLAM -> per-learner train/eval split), ``predict`` (engine-as-predictor +
calibration), ``baselines`` (DKT + difficulty/ability baselines), ``ablations``
(the RQ runs), ``run`` (CLI).
"""
from .dataset import EvalInstance, Interaction, LearnerData, load_track
from .metrics import evaluate, roc_curve

__all__ = [
    "Interaction", "EvalInstance", "LearnerData", "load_track",
    "evaluate", "roc_curve",
]
