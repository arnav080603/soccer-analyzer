"""A small trainable team classifier: logistic regression on jersey hue.

Scoped per clip, not global: two different games have two different pairs of jersey
colors, so a model trained on one game's colors is meaningless (often actively wrong —
e.g. classifying every player as one team) applied to a different game. Corrections you
make on a clip only ever train that clip's own model; re-running detection on the same
clip later reuses it, but a brand new clip always starts fresh from k-means.

Features are HSV hue (as cos/sin, since hue wraps around) + saturation — deliberately
dropping brightness/value. Raw BGR bakes in lighting: on shadow-heavy footage a plain
color classifier learns "bright vs. shadowed" instead of "red vs. blue" and misclassifies
constantly. Hue is what actually distinguishes jerseys and stays stable across shadow.
"""
import json
import os
import cv2
import numpy as np

MIN_EXAMPLES = 4


def _model_path(clip_name):
    base = os.path.splitext(clip_name)[0]
    return os.path.join(os.path.dirname(__file__), "data", f"{base}_team_model.json")


class TeamModel:
    def __init__(self, clip_name):
        self.clip_name = clip_name
        self.examples = []  # list of (color [b,g,r], label)
        self.weights = None  # (4,) = [w_cosH, w_sinH, w_S, bias]
        self._load()

    def _load(self):
        path = _model_path(self.clip_name)
        if os.path.exists(path):
            data = json.load(open(path))
            self.examples = [(list(c), l) for c, l in data.get("examples", [])]
            w = data.get("weights")
            self.weights = np.array(w) if w is not None else None

    def _save(self):
        path = _model_path(self.clip_name)
        data = {
            "examples": [(c, l) for c, l in self.examples],
            "weights": self.weights.tolist() if self.weights is not None else None,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(data, open(path, "w"), indent=2)

    @staticmethod
    def _features(colors):
        bgr = np.array(colors, dtype=np.uint8).reshape(-1, 1, 3)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float64)
        h_rad = hsv[:, 0] / 180.0 * 2 * np.pi
        s = hsv[:, 1] / 255.0
        X = np.stack([np.cos(h_rad), np.sin(h_rad), s], axis=1)
        return np.hstack([X, np.ones((len(X), 1))])

    def retrain(self):
        """Force a full retrain on the currently stored examples (e.g. after a feature-space change)."""
        self._train()
        self._save()

    def _train(self):
        labels = [l for _, l in self.examples]
        if len(self.examples) < MIN_EXAMPLES or len(set(labels)) < 2:
            return
        X = self._features([c for c, _ in self.examples])
        y = np.array(labels, dtype=np.float64)
        w = np.zeros(X.shape[1])
        lr = 0.5
        for _ in range(500):
            p = 1 / (1 + np.exp(-(X @ w)))
            grad = X.T @ (p - y) / len(y)
            w -= lr * grad
        self.weights = w

    def is_trained(self):
        return self.weights is not None

    def predict(self, colors):
        """colors: list of (b,g,r) -> list of 0/1 team labels."""
        X = self._features(colors)
        p = 1 / (1 + np.exp(-(X @ self.weights)))
        return (p >= 0.5).astype(int).tolist()

    def add_correction(self, color, label):
        self.examples.append((list(color), int(label)))
        self._train()
        self._save()

    def add_corrections(self, examples):
        """examples: list of (color, label). Trains once and saves once, for batches."""
        for color, label in examples:
            self.examples.append((list(color), int(label)))
        self._train()
        self._save()
