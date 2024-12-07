"""
単一の特徴量を持つ回帰モデルや列集合間の相互作用を
定義するのに用いる
"""

from itertools import product
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from scipy.interpolate import splev

class Poly(TransformerMixin, BaseEstimator):

    def __init__(self, degree=1, intercept=False, raw=False):
        self.degree = degree
        self.raw = raw
        self.intercept = intercept

    def fit(self, X, y=None):
        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single feature')
        self.mean_ = X.mean()

        if not self.raw:
            powX = np.power.outer(X - self.mean_, np.arange(0, self.degree + 1))
            Q, R = np.linalg.qr(powX)
            Z = Q * np.diag(R)[None, :]
            self.norm2_ = (Z**2).sum(0)
            self.alpha_ = ((X[:, None] * Z**2).sum(0) / self.norm2_)[:self.degree]
            self.norm2_ = np.hstack([1, self.norm2_])

        self.columns_ = range(self.degree+1)
        if not self.intercept:
            self.columns_ = self.columns_[:-1]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            if isinstance(X_orig, pd.Series):
                name = X_orig.name
            else:
                name = X_orig.columns[0]
            self.columns_ = ['{0}[{1}]'.format(self, d) for d in self.columns_]

        return self
    
    def transform(self, X):
        check_is_fitted(self)

        X_orig = X

        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.shape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')
        
        if not self.raw:
            Z = np.ones((n, self.degree+1))
            Z[:, 1] = X - self.alpha_[0]

            if self.degree > 1:
                for i in range(1, self.degree):
                    Z[:, i+1] = (
                        (X - self.alpha_[i]) * Z[:, i] - self.norm2_[i+1] / self.norm2_[i] * Z[:, i-1]
                    )
            Z /= np.sqrt(self.norm2_[1:])
            powX = Z
        else:
            powX = np.power.outer(X, np.arange(0, self.degree+1))

        if not self.intercept:
            powX = powX[:, 1:]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            df = pd.DataFrame(powX, columns=self.columns_)
            df.index = X_orig.index
            return df
        else:
            return powX
        
class Interaction(TransformerMixin, BaseEstimator):

    def __init__(self, variables, columns, column_names):
        self.variables = variables
        self.columns = columns
        self.column_name = column_names
        self.column_names_ = {}
        self.columns_ = []

        variable_names = []
        for variable in self.variables:
            cols = self.columns[variable]
            col_names = ['{0}[{1}]'.format(variable, i) for i in range(len(self.columns[variable]))]
            if variable in column_names:
                col_names = [str(c) for c in column_names[variable]]
            if len(cols) > 1:
                variable_names.append(col_names)
            else:
                variable_names.append(['{0}'.format(variable)])

        for names in product(*variable_names):
            self.columns_.append(':'.join(names))

    def fit(self, X, y):
        return self
    
    def transform(self, X):
        check_is_fitted(self)

        X_orig = X
        X = np.asarray(X)

        X_lists = []
        for variable in self.variables:
            X_lists.append(X[:, self.columns[variable]].T)

        cols = []
        for X_list in product(*X_lists):
            col = np.ones(X.shape[0])
            for x in X_list:
                col *= x
            cols.append(col)

        df = pd.DataFrame(np.column_stack(cols), columns=self.columns_)
        if isinstance(X_orig, (pd.DataFrame, pd.Series)):
            df.index = X_orig.index

        return df
    
def _onehot(p, j):
    v = np.zeros(p)
    v[j] = 1
    return v

def _splevf(x, tk, der=0, ext=0):
    x = np.asarray(x)
    knots, degree = tk
    nbasis = len(knots) - (degree + 1)
    tcks = [(knots, _onehot(nbasis, j), degree) for j in range(nbasis)]
    return np.column_stack([splev(x, tck, ext=ext, der=der) for tck in tcks])

def _splev_taylor(x, basept, tk, order):
    x = np.asarray(x)
    dervis = np.array([_splevf([basept], tk, der=o, ext=0).reshape(-1) for o in range(order+1)])
    polys = np.power.outer(x-basept, np.arange(order+1))
    fact = np.concatenate([[1], np.cumprod(np.arange(1, order+1))])
    polys /= fact[None,:]

    return polys.dot(dervis)

class BSpline(TransformerMixin, BaseEstimator):

    def __init__(
            self, degree=3, 
            intercept=False, 
            lower_bound=None, 
            upper_bound=None, 
            internal_knots=None, 
            df=None, 
            ext=0
        ):
        self.degree = degree
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.internal_knots = internal_knots
        self.df = df
        self.ext = ext
        self.intercept = intercept

    def fit(self, X, y=None):
        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')
        
        order = self.degree + 1
        
        if self.lower_bound is None:
            self.lower_bound = X.min()

        if self.upper_bound is None:
            self.upper_bound = X.max()

        if self.df is not None:
            if self.df < order - 1 + self.intercept:
                raise ValueError('df must be greator than or equal to %d' % (order - 1 + self.intercept))
            ninternal = self.df - (order - 1 + self.intercept)
            percs = 100*np.linspace(0, 1, ninternal+2)[1:-1]
            internal_knots = np.percentile(X, percs)
            if self.internal_knots is not None:
                raise ValueError('only one of df or internal_knots should be specified')
        else:
            internal_knots = np.asarray(sorted(self.internal_knots))
            if self.internal_knots is not None:
                raise ValueError('if df not specified then need internal_knots')
            
        if self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound must be smaller than upper_bound")
        
        self.internal_knots_ = internal_knots

        self.knots_ = np.sort(np.concatenate(
            [[self.lower_bound]*order,
             [self.upper_bound]*order,
             internal_knots]
        ))

        if self.knots_[0] < self.lower_bound:
            raise ValueError('internal_knots should be greater than our equal to lower_bound')
        if self.knots_[-1] > self.upper_bound:
            raise ValueError('internal_knots should be less than our equal to upper_bound')
        
        self.boundary_knots_ = [self.lower_bound, self.upper_bound]

        self.nbasis_ = len(self.knots_) - (self.degree + 1)
        self.columns_ = range(self.nbasis_)

        if not self.intercept:
            self.columns_ = self.columns_[:-1]

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            if isinstance(X_orig, pd.Series):
                name = X_orig.name
            else:
                name = X_orig.columns[0]
            self.columns_ = ['{0}[{1}]'.format(self, d) for d in self.columns_]

        return self
    
    def transform(self, X):
        check_is_fitted(self)

        X_orig = X
        X = np.squeeze(np.asarray(X).astype(float).copy())
        n = X.shape[0]
        if X.reshape(-1).shape[0] != n:
            raise ValueError('expecting a single column feature')
        
        value = _splevf(X, (self.knots_, self.degree), der=0, ext=self.ext)

        if not self.intercept:
            value = value[:, 1:]
        columns_ = self.columns_

        if isinstance(X_orig, (pd.Series, pd.DataFrame)):
            df = pd.DataFrame(value, columns=columns_)
            df.index = X_orig.index
            return df
        else:
            return value