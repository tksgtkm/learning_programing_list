import math
import sys

import numpy as np

import nsfg
import stats
import plot

def MakeFrames():
    preg = nsfg.ReadFemPreg()

    live = preg[preg.outcome == 1]
    firsts = live[live.birthord == 1]
    others = live[live.birthord != 1]

    assert len(live) == 9148
    assert len(firsts) == 4413
    assert len(others) == 4735

    return live, firsts, others

def Summarize(live, firsts, others):
    mean = live.prglngth.mean()
    var = live.prglngth.var()
    std = live.prglngth.std()

    print('Live mean', mean)
    print('Live variance', var)
    print('Live std', std)

    mean1 = firsts.prglngth.mean()
    mean2 = others.prglngth.mean()

    var1 = firsts.prglngth.var()
    var2 = others.prglngth.var()

    print('Mean')
    print('First babies', mean1)
    print('Others', mean2)

    print('Variance')
    print('First babies', var1)
    print('Others', var2)

    print('Difference in weeks', mean1 - mean2)
    print('Difference in hours', (mean1 - mean2) * 7 * 24)

    print('Difference relative to 39 weeks', (mean1 - mean2) / 39 * 100)

    d = stats.CohenEffectSize(firsts.prglngth, others.prglngth)
    print('Cohen d', d)

def PrintExtremes(live):
    hist = stats.Hist(live.prglngth)
    plot.Hist(hist, label='live births')

    plot.Save(
        root='savefig/first_nsfg_hist_live',
        title='Histgram',
        xlabel='weeks',
        ylabel='frequency'
    )

    print('Shortest length:')
    for weeks, freq in hist.Smallest(10):
        print(weeks, freq)

    print('Longest length')
    for weeks, freq in hist.Largest(10):
        print(weeks, freq)

def MakeHists(live):
    hist = stats.Hist(live.birthwgt_lb, label='birthwgt_lb')
    plot.Hist(hist)
    plot.Save(
        root='savefig/first_wgt_lb_hist',
        xlabel='pounds',
        ylabel='frequency',
        axis=[-1, 14, 0, 3200]
    )

    hist = stats.Hist(live.birthwgt_oz, label='birthwgt_oz')
    plot.Hist(hist)
    plot.Save(
        root='savefig/first_wgt_oz_hist',
        xlabel='ounces',
        ylabel='frequency',
        axis=[-1, 16, 0, 3200]
    )

    hist = stats.Hist(np.floor(live.agepreg), label='agepreg')
    plot.Hist(hist)
    plot.Save(
        root='savefig/first_agepreg_hist',
        xlabel='years',
        ylabel='frequency'
    )

    hist = stats.Hist(live.prglngth, label='prglngth')
    plot.Hist(hist)
    plot.Save(
        root='savefig/first_prglngth_hist',
        xlabel='weeks',
        ylabel='frequency',
        axis=[-1, 53, 0, 5000]
    )

def MakeComparison(firsts, others):
    first_hist = stats.Hist(firsts.prglngth, label='first')
    other_hist = stats.Hist(others.prglngth, label='other')

    width = 0.45
    plot.PrePlot(2)
    plot.Hist(first_hist, align='right', width=width)
    plot.Hist(other_hist, align='left', width=width)

    plot.Save(
        root='first_nsfg_hist',
        title='Histgram',
        xlabel='weeks',
        ylabel='frequency',
        axis=[27, 46, 0, 2700]
    )

def main(script):
    live, firsts, others = MakeFrames()

    MakeHists(live)
    PrintExtremes(live)
    MakeComparison(firsts, others)
    Summarize(live, firsts, others)

if __name__ == '__main__':
    main(*sys.argv)