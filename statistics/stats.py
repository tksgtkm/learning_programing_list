import re
import math
import logging
import warnings
import random
import copy
import bisect

from collections import Counter

import numpy as np
import pandas as pd

DEFAULT_LABEL = "_nolegend_"

class _DictWrapper:

    def __init__(self, obj=None, label=None):
        self.label = label if label is not None else DEFAULT_LABEL
        self.d = {}

        self.log = False

        if obj is None:
            return
        
        if isinstance(obj, dict):
            self.d.update(obj.items())
        elif isinstance(obj, pd.Series):
            self.d.update(obj.value_counts().items())
        else:
            self.d.update(Counter(obj))

        if len(self) > 0 and isinstance(self, Pmf):
            self.Normalize()

    def __hash__(self):
        return id(self)
    
    def __str__(self):
        cls = self.__class__.__name__
        if self.label == DEFAULT_LABEL:
            return "%s(%s)" % (cls, str(self.d))
        else:
            return self.label
        
    def __repr__(self):
        cls = self.__class__.__name__
        if self.label == DEFAULT_LABEL:
            return "%s(%s)" % (cls, repr(self.d))
        else:
            return "%s(%s, %s)" % (cls, repr(self.d), repr(self.label))
        
    def __eq__(self, other):
        try:
            return self.d
        except AttributeError:
            return False
        
    def __len__(self):
        return len(self.d)
    
    def __iter__(self):
        return iter(self.d)
    
    def iterkeys(self):
        return iter(self.d)
    
    def __contains__(self, value):
        return value in self.d
    
    def __getitem__(self, value):
        return self.d.get(value, 0)
    
    def __setitem__(self, value, prob):
        self.d[value] = prob

    def __delitem__(self, value):
        del self.d[value]

    def Copy(self, label=None):
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.label = label if label is not None else self.label
        return new
    
    def Scale(self, factor):
        new = self.Copy()
        new.d.clear()

        for val, prob in self.Items():
            new.Set(val * factor, prob)
        return new
    
    def Log(self, m=None):
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            if p:
                self.Set(x, math.log(p / m))
            else:
                self.Remove(x)

    def Exp(self, m=None):
        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.log = False

        if m is None:
            m = self.MaxLike()

        for x, p in self.d.items():
            self.Set(x, math.exp(p - m))

    def GetDict(self):
        return self.d
    
    def SetDict(self, d):
        self.d = d

    def Values(self):
        return self.d.keys()
        
    def Items(self):
        return self.d.items()
    
    def SortedItems(self):

        def isnan(x):
            try:
                return math.isnan(x)
            except TypeError:
                return False
            
        if any([isnan(x) for x in self.Values()]):
            msg = "Keys contain NaN, may not sort correctly"
            logging.warning(msg)

        try:
            return sorted(self.d.items())
        except TypeError:
            return self.d.items()
    
    def Render(self, **options):
        return zip(*self.SortedItems())
    
    def MakeCdf(self, label=None):
        label = label if label is not None else self.label
        return Cdf(self, label=label)
    
    def Print(self):
        for val, prob in self.SortedItems():
            print(val, prob)

    def Set(self, x, y=0):
        self.d[x] = y
    
    def Incr(self, x, term=1):
        self.d[x] = self.d.get(x, 0) + term

    def Mult(self, x, factor):
        self.d[x] = self.d.get(x, 0) * factor
    
    def Remove(self, x):
        del self.d[x]
    
    def Total(self):
        total = sum(self.d.values())
        return total
    
    def MaxLike(self):
        return max(self.d.values())
    
    def Largest(self, n=10):
        return sorted(self.d.items(), reverse=True)[:n]
    
    def Smallest(self, n=10):
        return sorted(self.d.items(), reverse=False)[:n]
        
class Hist(_DictWrapper):

    def Freq(self, x):
        return self.d.get(x, 0)
    
    def Freqs(self, xs):
        return [self.Freq(x) for x in xs]
    
    def IsSubset(self, other):
        for val, freq in self.Items():
            if freq > other.Freq(val):
                return False
        return True
    
    def Subtract(self, other):
        for val, freq in other.Items():
            self.Incr(val, -freq)

class Pmf(_DictWrapper):
    """
    pmf = stats.Pmf([1, 2, 2, 3, 5])
    
    print(pmf)

    print(pmf.Prob(2))

    pmf.Incr(2, 0.2)
    print(pmf.Prob(2))

    pmf.Mult(2, 0.5)
    print(pmf.Prob(2))

    print(pmf.Total())

    pmf.Normalize()
    print(pmf.Total())
    """

    def Prob(self, x, default=0):
        return self.d.get(x, default)
    
    def Probs(self, xs):
        return [self.Probs(x) for x in xs]
    
    def Percentile(self, percentage):
        p  = percentage / 100
        total = 0
        for val, prob in sorted(self.Items()):
            total += prob
            if total >= p:
                return val
            
    def ProbGreater(self, x):
        if isinstance(x, _DictWrapper):
            return PmfProbGreater(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val > x]
            return sum(t)
        
    def ProbLess(self, x):
        if isinstance(x, _DictWrapper):
            return PmfProbLess(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val < x]
            return sum(t)
        
    def ProbEqual(self, x):
        if isinstance(x, _DictWrapper):
            return PmfProbEqual(self, x)
        else:
            return self[x]
        
    def Normalize(self, fraction=1):
        if self.log:
            raise ValueError("Normalize: Pmf is under a log transform")
        
        total = self.Total()
        if total == 0:
            raise ValueError("Normalize: total probability is zero.")
        
        factor = fraction / total
        for x in self.d:
            self.d[x] *= factor

        return total
    
    def Random(self):
        target = random.random()
        total = 0
        for x, p in self.d.items():
            total += p
            if total >= target:
                return x
            
        raise ValueError("Random: Pmf might not be normalized")
    
    def Sample(self, n):
        return self.MakeCdf().Sample(n)

    def Mean(self):
        return sum(p * x for x, p in self.Items())
    
    def Median(self):
        return self.MakeCdf().Percentile(50)

    def Var(self, mu=None):
        if mu is None:
            mu = self.Mean()

        return sum(p * (x - mu) ** 2 for x, p in self.Items())

    def Except(self, func):
        return np.sum(p * func(x) for x, p in self.Items())

    def Std(self, mu=None):
        var = self.Var(mu)
        return math.sqrt(var)

    def Mode(self):
        _, val = max((prob, val) for val, prob in self.Items())
        return val

    MAP = Mode

    Maximumlikelihood = Mode

    def CredibleInterval(self, percentage=90):
        cdf = self.MakeCdf()
        return cdf.CredibleInterval(percentage)

    def __add__(self, other):
        try:
            return self.AddPmf(other)
        except AttributeError:
            return self.AddConstant(other)

    __radd__ = __add__

    def AddPmf(self, other):
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf[v1 + v2] += p1 * p2
        return pmf

    def AddConstant(self, other):
        if other == 0:
            return self.Copy()

        pmf = Pmf()
        for v1, p1 in self.Items():
            pmf.Set(v1 + other, p1)
        return pmf

    def __sub__(self, other):
        try:
            return self.SubPmf(other)
        except AttributeError:
            return self.AddConstant(-other)

    def SubPmf(self, other):
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 - v2, p1 * p2)
        return pmf

    def __mul__(self, other):
        try:
            return self.MulPmf(other)
        except AttributeError:
            return self.MulConstant(other)

    def MulPmf(self, other):
        pmf = Pmf()
        for  v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 * v2, p1 * p2)
        return pmf

    def MulConstant(self, other):
        pmf = Pmf()
        for v1, p1 in self.Items():
            pmf.Set(v1 * other, p1)
        return pmf

    def __div__(self, other):
        try:
            return self.DivPmf(other)
        except AttributeError:
            return self.MulConstant(1 / other)

    __truediv__ = __div__

    def DivPmf(self, other):
        pmf = Pmf()
        for v1, p1 in self.Items():
            for v2, p2 in other.Items():
                pmf.Incr(v1 / v2, p1 * p2)
        return pmf

    def Max(self, k):
        cdf = self.MakeCdf()
        cdf.ps **= k
        return cdf
    
class Cdf:
    
    def __init__(self, obj=None, ps=None, label=None):
        self.label = label if label is not None else DEFAULT_LABEL

        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            if not label:
                self.label = label if label is not None else obj.label

        if obj is None:
            self.xs = np.asarray([])
            self.ps = np.asarray([])
            if ps is not None:
                logging.warning("Cdf: can't pass ps without also passing xs.")
            return
        else:
            if ps is not None:
                if isinstance(ps, str):
                    logging.warning("Cdf: ps can't be a string")

                self.xs = np.asarray(obj)
                self.ps = np.asarray(ps)
                return

        if isinstance(obj, Cdf):
            self.xs = copy.copy(obj.xs)
            self.ps = copy.copy(obj.ps)
            return

        if isinstance(obj, _DictWrapper):
            dw = obj
        else:
            dw = Hist(obj)

        if len(dw) == 0:
            self.xs = np.asarray([])
            self.ps = np.asarray([])
            return

        xs, freqs = zip(*sorted(dw.Items()))
        self.xs = np.asarray(xs)
        self.ps = np.cumsum(freqs, dtype=float)
        self.ps /= self.ps[-1]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, x):
        return self.Prob(x)

    def __setitem__(self):
        raise UnimplementedMethodException()

    def __delitem__(self):
        raise UnimplementedMethodException()

    def __eq__(self, other):
        return np.all(self.xs == other.ps) and np.all(self.ps == other.ps)

    def Print(self):
        for val, prob in zip(self.xs, self.ps):
            print(val, prob)

    def Copy(self, label=None):
        if label is None:
            label = self.label
        return Cdf(list(self.xs), list(self.ps), label=label)

    def MakePmf(self, label=None):
        if label is None:
            label = self.label
        return Pmf(self, label=label)

    def Prob(self, x):
        if x < self.xs[0]:
            return 0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
        return p

    def Probs(self, xs):
        xs = np.asarray(xs)
        index = np.searchsorted(self.xs, xs, side="right")
        ps = self.ps[index - 1]
        ps[xs < self.xs[0]] = 0
        return ps

    ProbArray = Probs

    def Value(self, p):
        if p < 0 or p > 1:
            raise ValueError("Probability p must be in range [0, 1]")

        index = bisect.bisect_left(self.ps, p)
        return self.xs[index]

    def Values(self, ps=None):
        if ps is None:
            return self.ps

        ps = np.asarray(ps)
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError("Probability p must be in range [0, 1]")

        index = np.searchsorted(self.ps, ps, size="left")
        return self.xs[index]

    ValueArray = Values

    def Percentile(self, p):
        return self.Value(p / 100)

    def Percentiles(self, ps):
        ps = np.asarray(ps)
        return self.Values(ps / 100)

    def PercentileRank(self, x):
        return self.Prob(x) * 100

    def PercentileRanks(self, xs):
        return self.Prob(xs) * 100

    def Random(self):
        return self.Value(random.random())

    def Sample(self, n):
        ps = np.random.random(n)
        return self.ValueArray(ps)

    def Mean(self):
        old_p = 0
        total = 0
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total

    def CredibleInterval(self, percentage=90):
        prob = (1 - percentage / 100) / 2
        interval = self.Value(prob), self.Value(1 - prob)
        return interval

    ConfidenceInterval = CredibleInterval

    def _Round(self, multiplier=1000):
        raise UnimplementedMethodException()

    def Render(self, **options):

        def interleave(a, b):
            c = np.empty(a.shape[0] + b.shape[0])
            c[::2] = a
            c[1::2] = b
            return c

        a = np.array(self.xs)
        xs = interleave(a, a)
        shift_ps = np.roll(self.ps, 1)
        shift_ps[0] = 0
        ps = interleave(shift_ps, self.ps)
        return xs, ps

    def Max(self, k):
        cdf = self.Copy()
        cdf.ps **= k
        return cdf

class UnimplementedMethodException(Exception):
    "上書されるべきメソッドを呼び出した場合の例外"

class Pdf:
    pass

class FixedWidthVariables(object):

    def __init__(self, variables, index_base=0):
        self.variables = variables

        self.colspecs = variables[["start", "end"]] - index_base

        self.colspecs = self.colspecs.astype(int).values.tolist()
        self.names = variables["name"]

    def ReadFixedWidth(self, filename, **options):
        df = pd.read_fwf(
            filename, colspecs=self.colspecs, names=self.names, **options
        )
        return df

def ReadStataDct(dct_file, **options):
    type_map = dict(
        byte=int, int=int, long=int, float=float, double=float, numeric=float
    )

    var_info = []
    with open(dct_file, **options) as f:
        for line in f:
            match = re.search(r"_column\(([^)]*)\)", line)
            if not match:
                continue
            start = int(match.group(1))
            t = line.split()
            vtype, name, fstring = t[1:4]
            name = name.lower()
            if vtype.startswith("str"):
                vtype = str
            else:
                vtype = type_map[vtype]
            long_desc = " ".join(t[4:]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))

    columns = ["start", "type", "name", "fstring", "desc"]
    variables = pd.DataFrame(var_info, columns=columns)

    variables["end"] = variables.start.shift(-1)
    variables.loc[len(variables) - 1, "end"] = -1

    dct = FixedWidthVariables(variables, index_base=1)

    return dct

def PmfProbLess(pmf1, pmf2):
    total = 0
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            if v1 < v2:
                total += p1 * p2
    return total

def PmfProbGreater(pmf1, pmf2):
    total = 0
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            if v1 > v2:
                total += p1 * p2
    return total

def PmfProbEqual(pmf1, pmf2):
    total = 0
    for v1, p1 in pmf1.Items():
        for v2, p2 in pmf2.Items():
            if v1 == v2:
                total += p1 * p2
    return total

def CohenEffectSize(group1, group2):
    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / math.sqrt(pooled_var)
    return d