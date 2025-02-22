import math

import numpy as np

import nsfg
import first
import stats
import plot

pmf = stats.Pmf([1, 2, 3, 4, 5])

print(pmf)

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
        root='probability_agepreg_hist',
        xlabel='years',
        axis=[0, 45, 0, 700]
    )