import re
import math
import logging
import warnings
import random
import copy

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
    
    # TODO Cdfクラス実装後から行う
    def Sample(self, n):
        pass

    def Mean(self):
        return sum(p * x for x, p in self.Items())
    
    # def Median(self):
    
class Cdf:
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