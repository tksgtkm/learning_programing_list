from typing import NamedTuple, Any
from copy import copy

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

class Column(NamedTuple):

    idx: Any
    name: str
    is_categorical: bool = False
    is_ordinal: bool = False
    columns: tuple = ()
    encoder: Any = None

    def get_columns(self, X, fit=False):
        """
        Xから関連する列を抽出する
        Noneでなければ`self.encoder`とエンコーディングする
        """
        cols = _get_column(self.idx, X)
        if fit:
            self.fit_encoder(X)
            
        if self.encoder is not None:
            try:
                cols = self.encoder.transform(cols)
            except ValueError:
                cols = self.encoder.transform(np.asarray(cols).reshape(-1, 1))

        cols = np.asarray(cols)

        names = self.columns
        if hasattr(self.encoder, 'columns_'):
            names = ['{0}[{1}]'.format(self.name, c) for c in self.encoder.columns_]

        if not names:
            names = ['{0}[{1}]'.format(self.name, i) for i in range(cols.shape[1])]

        return cols
    
    def fit_encoder(self, X):
        cols = _get_column(self.idx, X)
        if self.encoder is not None:
            try:
                check_is_fitted(self.encoder)
            except NotFittedError:
                self.encoder.fit(cols)
        return np.asarray(cols)

def _get_column(idx, X, loc=True):
    """
    二次元のndarrayまたはpd.DataFrameから
    Xからidx列を抽出する
    """
    if isinstance(X, np.ndarray):
        col = X[:, [idx]]
    elif hasattr(X, 'loc'):
        if loc:
            col = X.loc[:, [idx]]
        else:
            col = X.iloc[:, [idx]]
    else:
        raise ValueError('expecting an ndarray or a ' +
                         '"loc/iloc" methods, got %s' % str(X))
    return col

def _get_column_info(X, columns, is_categorical, is_ordinal, default_encoders={'ordinal': OrdinalEncoder(), 'categorical': OneHotEncoder()}):
    column_info = {}
    for i, col in enumerate(columns):
        if type(col) == int:
            name = f'X{col}'
        else:
            name = str(col)
        if is_categorical[i]:
            if is_ordinal[i]:
                Xcol = _get_column(col, X)
                encoder = clone(default_encoders['ordinal'])
                encoder.fit(Xcol)
                columns = ['{0}'.format(col)]
            else:
                Xcol = _get_column(col ,X)
                encoder = clone(default_encoders['categorical'])
                cols = encoder.fit_transform(Xcol)
                if hasattr(encoder, 'columns_'):
                    columns_ = encoder.columns_
                else:
                    columns_ = range(cols.shape[1])
                columns = ['{0}[{1}]'.format(col, c) for c in columns_]

            column_info[col] = Column(
                col,
                name,
                is_categorical[i],
                is_ordinal[i],
                tuple(columns),
                encoder
            )
        else:
            Xcol = _get_column(col, X)
            column_info[col] = Column(
                col,
                name,
                columns=(name,)
            )
    return column_info

def _check_categories(categorical_features, X):
    if categorical_features is None:
        return None, None
    
    categorical_features = np.asarray(categorical_features)

    if categorical_features.size == 0:
        return None, None
    
    if categorical_features.dtype.kind not in ('i', 'b'):
        raise ValueError("categorical_features must be an array-like of "
                         "bools or array-like of ints")
    
    n_features = X.shape[1]

    if categorical_features.dtype.kind == 'i':
        if (np.max(categorical_features) >= n_features or np.min(categorical_features) < 0):
            raise ValueError("categorical_features set as integer "
                             "indices must be in [0, n_features - 1]")
        is_categorical = np.zeros(n_features, dtype=bool)
        is_categorical[categorical_features] = True
    else:
        if categorical_features.shape[0] != n_features:
            raise ValueError("categorical_features set as a boolean mask "
                             "must have shape (n_features,), got")
        is_categorical = categorical_features

    if not np.any(is_categorical):
        return None, None
    
    known_categories = []

    for f_idx in range(n_features):
        if is_categorical[f_idx]:
            categories = np.array([v for v in set(_get_column(f_idx, X, loc=False))])

            missing = []
            for c in categories:
                try:
                    missing.append(np.isnan(c))
                except TypeError:
                    missing.append(False)
            missing = np.array(missing)
            if missing.any():
                categories = sorted(categories[~missing])
        else:
            categories = None
        known_categories.append(categories)

    return is_categorical, known_categories

def _categorical_from_df(df):
    is_categorical = []
    is_ordinal = []
    for c in df.columns:
        try:
            if df[c].dtype == 'category':
                is_categorical.append(True)
                is_ordinal.append(df[c].cat_ordered)
            else:
                is_categorical.append(False)
                is_ordinal.append(False)
        except (TypeError, AttributeError):
            is_categorical.append(False)
            is_ordinal.append(False)
    is_categorical = np.asarray(is_categorical)
    is_ordinal = np.asarray(is_ordinal)

    return is_categorical, is_ordinal