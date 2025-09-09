import numpy as np
from dtw import dtw

# base class
class BehavioralMetric:
    def eval(self, trajs):
        raise NotImplementedError
    
    def batch_eval(self, samples):
        results = []
        for s in samples:
            results.append(self.eval(s))
        return np.mean(results)

def hamming_dis(seq1, seq2):
    assert len(seq1) == len(seq2)
    differences = np.sum(np.array(seq1) != np.array(seq2))
    return differences

# Behavioral path coverage for agents
class TrajectorySpread(BehavioralMetric):
    def __init__(self, dist_func=hamming_dis, w=5):
        self.dist_func = dist_func
        self.w = w

    def eval(self, traj1, traj2):
        res, *_ = dtw(traj1, traj2, dist=lambda x,y: self.dist_func(x,y), w=self.w)
        return res / max(len(traj1), len(traj2))

traj1 = [(0,0), (1,1), (2,2), (3,1), (3,3)]
traj2 = [(0,0), (1,0), (1,2), (2,1), (3,2)]

traj_dtw = TrajectorySpread()
print("Trajectory Spread (DTW):", traj_dtw.eval(traj1, traj2))