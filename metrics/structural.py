import numpy as np
from scipy.stats import entropy

# base class
class StructuralMetric:
    def eval(self, sample):
        raise NotImplementedError
    
    def batch_eval(self, samples):
        results = []
        for s in samples:
            results.append(self.eval(s))
        return np.mean(results)

# Structural diversity based on tile frequency
class EntropyMetric(StructuralMetric):
    def __init__(self, base=2):
        self.base = base

    def eval(self, level):
        # tile frequency
        unique, counts = np.unique(level, return_counts=True)
        pk = counts / counts.sum()
        return entropy(pk, base=self.base)

# Distributional divergence from baseline samples
class KLDivergence(StructuralMetric):
    def __init__(self, baseline, num_tiles=3, base=2,  eps=0.001):
        self.num_tiles = num_tiles
        self.eps = eps
        self.base = base
        # baseline = qx
        self.baseline = np.array(baseline) + eps
        self.baseline = self.baseline / self.baseline.sum()

    def level_distribution(self, level):
        counts = np.zeros(self.num_tiles)
        for t in range(self.num_tiles):
            counts[t] = np.sum(level == t)
        return counts / counts.sum()

    def eval(self, level):
        pk = self.level_distribution(level) + self.eps
        pk = pk / pk.sum()
        return entropy(pk, self.baseline, base=self.base)


level1 = np.array([[1,0,2],[0,1,0],[2,0,1]])
level2 = np.array([[1,1,0],[0,0,1],[1,2,0]])
level3 = np.array([[0,2,1],[2,1,0],[0,1,0]])
levels = [level1, level2, level3]

# entropy
cal_entropy = EntropyMetric()
print("Entropy:", cal_entropy.batch_eval(levels))

# kl divergence
baseline = [0.5, 0.3, 0.2]   
kl = KLDivergence(baseline, num_tiles=3)
for i, lv in enumerate(levels):
    print(f"Level {i+1} KL Divergence:", kl.eval(lv))
