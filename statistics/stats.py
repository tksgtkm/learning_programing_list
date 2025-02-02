import re

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
        
    def Items(self):
        return self.d.items()
    
    def Incr(self, x, term=1):
        self.d[x] = self.d.get(x, 0) + term
        
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