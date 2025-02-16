import functools
import heapq
import operator
import random
import bisect

import numpy as np

# ______________________________________________________________________________
# Functions on Sequences and Iterables

def is_in(elt, seq):
    return any(x is elt for x in seq)

# ______________________________________________________________________________
# argmin and argmax

identity = lambda x: x

def argmin_random_tie(seq, key=identity):
    return min(shuffled(seq), key=key)

def argmax_random_tie(seq, key=identity):
    return max(shuffled(seq), key=key)

def shuffled(iterable):
    items = list(iterable)
    random.shuffle(items)
    return items
# ______________________________________________________________________________

# ______________________________________________________________________________
# Statistical and mathematical functions

def vector_add(a, b):
    return tuple(map(operator.add, a, b))

def probability(p):
    return p > random.uniform(0.0, 1.0)

def weighted_sampler(seq, weights):
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]

def gaussian_kernel(l=5, sig=1.0):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))
    return kernel

# ______________________________________________________________________________
# Grid Functions

def distance(a, b):
    xA, yA = a
    xB, yB = b
    return (xA - xB) ** 2 + (yA - yB) ** 2

# ______________________________________________________________________________



# ______________________________________________________________________________
# Misc Functions

def memoize(fn, slot=None, maxsize=32):
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)
    
    return memoized_fn

# ______________________________________________________________________________

# ______________________________________________________________________________
# Queues: Stack, FIFOQueue, PriorityQueue
# Stack and FIFOQueue are implemented as list and collection.deque
# PriorityQueue is implemented here

class PriorityQueue:

    def __init__(self, order='min', f=lambda x: x):
        self.heap = []
        if order == 'min':
            self.f = f
        elif order == 'max':
            self.f = lambda x: -f(x)
        else:
            raise ValueError("Order must be either 'min' or 'max'")
        
    def append(self, item):
        heapq.heappush(self.heap, (self.f(item), item))

    def extend(self, items):
        for item in items:
            self.append(item)

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        else:
            raise Exception('Trying to pop from empty PriorityQueue')
        
    def __len__(self):
        return len(self.heap)
    
    def __contains__(self, key):
        return any([item == key for _, item in self.heap])
    
    def __getitem__(self, key):
        for value, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key))
    
    def __delitem__(self, key):
        try:
            del self.heap[[item == key for _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + "is not in the priority queue")
        heapq.heapify(self.heap)

# ______________________________________________________________________________