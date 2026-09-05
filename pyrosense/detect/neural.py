"""
Stage 2: optional neural verifier.

Deliberately a thin, pluggable slot rather than a bundled model.

The honest position on fire-detection CNNs is that the public datasets (FireNet,
BoWFire, FiSmo, the Kaggle fire sets) are dominated by outdoor bonfires, forest
fires and dramatic building fires photographed from the front. A model trained on
those transfers poorly to a ceiling-mounted 704x576 camera looking down a dim
aisle of racking. Shipping such a model and calling it "AI fire detection" would
add a confident-sounding number without adding real accuracy.

So the design is: this stage does nothing until someone drops an ONNX file in
`data/models/`, and the intended path to that file is site footage. A deployment
accumulates labelled crops from its own review queue - every confirmed alert and
every dismissal is a training example, already cropped and already labelled by the
adjudicator - and after a few weeks a small classifier fine-tuned on *that* is
worth far more than anything pre-trained on bonfires.

Any ONNX image classifier works. The expected contract:
  input   NCHW float32, 3x224x224 (or whatever the model declares), 0..1 scaled
  output  either a 2-logit [not_fire, fire] tensor, or a single sigmoid probability
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:                                          # pragma: no cover
    cv2 = None


@dataclass
class NeuralConfig:
    flame_model: str = ""
    smoke_model: str = ""
    size: int = 224
    mean: tuple = (0.0, 0.0, 0.0)
    std: tuple = (1.0, 1.0, 1.0)
    providers: tuple = ("CPUExecutionProvider",)


class NeuralVerifier:
    """Loads ONNX models if present; otherwise `predict` returns None and the
    cascade simply skips stage 2. Never raises on a missing model."""

    def __init__(self, cfg: NeuralConfig | None = None):
        self.cfg = cfg or NeuralConfig()
        self._sess: dict[str, object] = {}
        self._inputs: dict[str, str] = {}
        self._shape: dict[str, tuple] = {}
        self.loaded: list[str] = []
        self.error = ""
        self._load()

    def _load(self) -> None:
        paths = {"flame": self.cfg.flame_model, "smoke": self.cfg.smoke_model}
        have = {k: p for k, p in paths.items() if p and os.path.exists(p)}
        if not have:
            self.error = "no ONNX models present - stage 2 disabled"
            return
        try:
            import onnxruntime as ort
        except Exception as e:
            self.error = f"onnxruntime unavailable: {e}"
            return
        for kind, path in have.items():
            try:
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1        # we parallelise over cameras
                sess = ort.InferenceSession(path, so, providers=list(self.cfg.providers))
                inp = sess.get_inputs()[0]
                self._sess[kind] = sess
                self._inputs[kind] = inp.name
                shape = tuple(d if isinstance(d, int) else -1 for d in inp.shape)
                self._shape[kind] = shape
                self.loaded.append(kind)
            except Exception as e:
                self.error = f"{kind}: {type(e).__name__}: {e}"[:200]

    @property
    def available(self) -> bool:
        return bool(self._sess)

    def _pre(self, crop: np.ndarray, kind: str) -> np.ndarray:
        shape = self._shape.get(kind, ())
        size = self.cfg.size
        if len(shape) == 4 and shape[2] > 0:
            size = shape[2]
        img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.asarray(self.cfg.mean, np.float32)) / np.asarray(self.cfg.std, np.float32)
        return np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)

    def predict(self, crop: np.ndarray, kind: str = "flame") -> float | None:
        """Probability that `crop` shows the given class. None if unavailable."""
        sess = self._sess.get(kind)
        if sess is None or crop is None or crop.size == 0:
            return None
        try:
            out = sess.run(None, {self._inputs[kind]: self._pre(crop, kind)})[0]
            v = np.asarray(out).ravel()
            if v.size == 1:
                return float(1.0 / (1.0 + np.exp(-v[0]))) if not (0.0 <= v[0] <= 1.0) \
                    else float(v[0])
            e = np.exp(v - v.max())
            return float((e / e.sum())[-1])
        except Exception:
            return None

    def status(self) -> dict:
        return {"available": self.available, "loaded": self.loaded,
                "error": self.error}


def export_training_crops(events, out_dir: str = "data/training") -> int:
    """Turn the review queue into a labelled dataset.

    Every adjudicated event already carries a crop and a label that a vision model
    assigned and (usually) a human confirmed. Writing them into class folders is
    all that stands between a running deployment and a site-specific stage-2
    model, so the plumbing is here from day one."""
    import shutil
    n = 0
    for ev in events:
        label = ev.get("verdict") if isinstance(ev, dict) else getattr(ev, "verdict", "")
        snap = ev.get("snapshot") if isinstance(ev, dict) else getattr(ev, "snapshot", "")
        if not snap or not os.path.exists(snap) or label not in ("confirmed", "dismissed"):
            continue
        cls = "fire" if label == "confirmed" else "not_fire"
        d = os.path.join(out_dir, cls)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(snap, os.path.join(d, os.path.basename(snap)))
        n += 1
    return n
