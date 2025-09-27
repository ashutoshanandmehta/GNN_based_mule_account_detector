import bisect
from .config import Config

class DecisionEngine:
    def __init__(self, cfg: Config):
        t = cfg.thresholds
        self.cuts = [t.ALLOW_T, t.STEP_UP_T, t.HOLD_T, t.BLOCK_T]
        self.labels = ["ALLOW", "STEP_UP", "HOLD", "BLOCK", "BLOCK"]

    def decide(self, fused_score: float) -> str:
        i = bisect.bisect_left(self.cuts, fused_score)
        return self.labels[i]
