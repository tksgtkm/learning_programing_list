from collections import namedtuple
from itertools import product
from typing import NamedTuple, Any
from copy import copy

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.exceptions import NotFittedError
from joblib import hash as joblib_hash