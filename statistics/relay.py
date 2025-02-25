import sys

import numpy as np

import stats
import plot

def ConvertPaceToSpeed(pace):
    m, s = [int(x) for x in pace.split(':')]
    secs = m*60 + s
    mph = 1 / secs * 60 * 60
    return mph

def CleanLine(line):
    t = line.split()
    if len(t) < 6:
        return None
    
    place, divtot, div, gun, net, pace = t[0:6]

    if not '/' in divtot:
        return None
    
    for time in [gun, net, pace]:
        if ':' not in time:
            return None
        
    return place, divtot, div, gun, net, pace

def ReadResults(filename='dataset/Apr25_27thAn_set1.shtml'):
    results = []
    for line in open(filename):
        t = CleanLine(line)
        if t:
            results.append(t)
    return results

def GetSpeeds(results, column=5):
    speeds = []
    for t in results:
        pace = t[column]
        speed = ConvertPaceToSpeed(pace)
        speeds.append(speed)
    return speeds

def BinData(data, low, high, n):
    data = (np.array(data) - low) / (high - low) * n
    data = np.round(data) * (high - low) / n + low
    return data

def ObservePmf(pmf, speed, label=None):
    new = pmf.Copy(label=label)
    for val in new.Values():
        diff = abs(val - speed)
        new.Mult(val, diff)
    new.Normalize()
    return new

def main(script):
    results = ReadResults()
    speeds = GetSpeeds(results)
    speeds = BinData(speeds, 3, 12, 100)

    pmf = stats.Pmf(speeds, 'actual speeds')

    biased = ObservePmf(pmf, 7.5, label='observed speeds')

    plot.Pmf(biased)
    plot.Save(
        root='savefig/observed_speeds',
        title='PMF of running speed',
        xlabel='speed (mph)',
        ylabel='PMF'
    )

if __name__ == '__main__':
    import sys
    main(*sys.argv)