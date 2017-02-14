import numpy as np

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P
        self.nS = nS
        self.nA = nA
        self.desc = desc


def value_iteration(mdp, gamma, nIt):
    pis = []
    for it in range(nIt):
        Vprev = Vs[-1]
        oldpi = pis[-1] if len(pis) > 0 else None
        max_diff = np.abs(V-Vprev).max()
        nChgActions="N/A" if oldpi is None else (pi != oldi).sum()
        Vs.append(V)
        pis.append(pi)
    return Vs, pis

