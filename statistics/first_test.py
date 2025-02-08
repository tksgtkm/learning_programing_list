import sys
from operator import itemgetter

import first
import stats

def Mode(hist):
    p, x = max([(p, x) for x, p in hist.Items()])
    return x

def AllModes(hist):
    return sorted(hist.Items(), key=itemgetter(1), reverse=True)

def WeightDifference(live, firsts, others):
    mean0 = live.totalwgt_lb.mean()
    mean1 = firsts.totalwgt_lb.mean()
    mean2 = others.totalwgt_lb.mean()

    var1 = firsts.totalwgt_lb.var()
    var2 = others.totalwgt_lb.var()

    print('Mean')
    print('First babies', mean1)
    print('Others', mean2)

    print('Variance')
    print('First babies', var1)
    print('Others', var2)

    print('Difference in lbs', mean1 - mean2)
    print('Difference in oz', (mean1 - mean2) * 16)

    print('Difference relative to mean (%age points)',
          (mean1 - mean2) / mean0 * 100)
    
    d = stats.CohenEffectSize(firsts.totalwgt_lb, others.totalwgt_lb)
    print('Cohen d', d)

def main(script):
    live, firsts, others = first.MakeFrames()
    hist = stats.Hist(live.prglngth)

    WeightDifference(live, firsts, others)

    mode = Mode(hist)
    print('Mode of preg length', mode)
    assert(mode == 39)

    modes = AllModes(hist)
    assert(modes[0][1] == 4693)

    for value, freq in modes[:5]:
        print(value, freq)

    print('%s: All tests passed.' % script)

if __name__ == '__main__':
    main(*sys.argv)