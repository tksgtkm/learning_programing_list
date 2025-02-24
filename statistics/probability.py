import math

import numpy as np

import nsfg
import first
import stats
import plot

# pmf = stats.Pmf([1, 2, 2, 3, 5])

# print(pmf)

# print(pmf.Prob(2))

# pmf.Incr(2, 0.2)
# print(pmf.Prob(2))

# pmf.Mult(2, 0.5)
# print(pmf.Prob(2))

# print(pmf.Total())

# pmf.Normalize()
# print(pmf.Total())

def MakeHists(live):
    hist = stats.Hist(np.floor(live.agepreg), label='agepreg')
    plot.PrePlot(2, cols=2)

    plot.SubPlot(1)
    plot.Hist(hist)
    plot.Config(
        xlabel='years',
        ylabel='frequency',
        axis=[0, 45, 0, 700]
    )

    plot.SubPlot(2)
    plot.Hist(hist)
    plot.Save(
        root='savefig/probability_agepreg_hist',
        xlabel='years',
        axis=[0, 45, 0, 700]
    )

def MakeFigures(firsts, others):
    first_pmf = stats.Pmf(firsts.prglngth, label='first')
    other_pmf = stats.Pmf(others.prglngth, label='other')
    width = 0.45

    plot.PrePlot(2, cols=2)
    plot.Hist(first_pmf, align='right', width=width)
    plot.Hist(other_pmf, align='left', width=width)
    plot.Config(
        xlabel='weeks',
        ylabel='probability',
        axis=[27, 46, 0, 0.6]
    )

    plot.PrePlot(2)
    plot.SubPlot(2)
    plot.Pmfs([first_pmf, other_pmf])
    plot.Save(
        root='savefig/probability_nsfg_pmf',
        xlabel='weeks',
        axis=[27, 46, 0, 0.6]
    )

    weeks = range(35, 46)
    diffs = []
    for week in weeks:
        p1 = first_pmf.Prob(week)
        p2 = other_pmf.Prob(week)
        diff = 100 * (p1 - p2)
        diffs.append(diff)

    plot.Bar(weeks, diffs)
    plot.Save(
        root='savefig/probability_nsfg_diffs',
        title='Difference in PMFs',
        xlabel='weeks',
        ylabel='savefig/percentage points',
        legend=False
    )

def BiasPmf(pmf, label=''):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf.Mult(x, x)
    
    new_pmf.Normalize()
    return new_pmf

def UnbiasPmf(pmf, label=''):
    new_pmf = pmf.Copy(label=label)

    for x, p in pmf.Items():
        new_pmf.Mult(x, 1.0/x)

    new_pmf.Normalize()
    return new_pmf

def ClassSizes():

    d = { 7: 8, 12: 8, 17: 14, 22: 4, 
          27: 6, 32: 12, 37: 8, 42: 3, 47: 2 }
    
    pmf = stats.Pmf(d, label='actual')
    print('mean', pmf.Mean())
    print('var', pmf.Var())

    biased_pmf = BiasPmf(pmf, label='observed')
    print('mean', biased_pmf.Mean())
    print('var', biased_pmf.Var())

    unbiased_pmf = UnbiasPmf(biased_pmf, label='unbiased')
    print('mean', unbiased_pmf.Mean())
    print('var', unbiased_pmf.Var())

    plot.PrePlot(2)
    plot.Pmfs([pmf, biased_pmf])
    plot.Save(
        root='savefig/class_size1',
        xlabel='class size',
        ylabel='PMF',
        axis=[0, 52, 0, 0.27]
    )

def main(script):
    live, firsts, others = first.MakeFrames()
    MakeFigures(firsts, others)
    MakeHists(live)

    ClassSizes()

if __name__ == '__main__':
    import sys
    main(*sys.argv)