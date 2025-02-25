import sys
import numpy as np

import nsfg
import first
import stats
import plot

def PmfMean(pmf):
    mean = 0.0
    for x, p in pmf.d.items():
        mean += p * x
    return mean

def PmfVar(pmf, mu=None):
    if mu is None:
        mu = pmf.Mean()
    
    var = 0.0
    for x, p in pmf.d.items():
        var += p * (x - mu) ** 2
    return var

def Diffs(t):
    first = t[0]
    rest = t[1:]
    diffs = [first - x for x in rest]
    return diffs

def PairWiseDifferences(live):
    live = live[live.prglngth >= 37]
    preg_map = nsfg.MakePregMap(live)

    diffs = []
    for caseid, indices in preg_map.items():
        lengths = live.loc[indices].prglngth.values
        if len(lengths) >= 2:
            diffs.extend(Diffs(lengths))

    mean = stats.Mean(diffs)
    print('Mean difference between pairs', mean)

    pmf = stats.Pmf(diffs)
    plot.Hist(pmf, align='center')
    plot.Show(xlabel='Difference in weeks', ylabel='PMF')

def main(script):

    live, firsts, others = first.MakeFrames()
    PairWiseDifferences(live)

    prglngth = live.prglngth
    pmf = stats.Pmf(prglngth)
    mean = PmfMean(pmf)
    var = PmfVar(pmf)

    assert(mean == pmf.Mean())
    assert(var == pmf.Var())
    print('mean/var preg length', mean, var)

    print('%s: All tests passed' % script)

if __name__ == '__main__':
    main(*sys.argv)