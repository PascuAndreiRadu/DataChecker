from collections import defaultdict

# Lazy-loaded module cache
_np = _pd = _pl = _torch = _tf = None


def get_np():
    global _np
    if _np is None:
        import numpy as np
        _np = np
    return _np


def get_pd():
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd


def get_pl():
    global _pl
    if _pl is None:
        import polars as pl
        _pl = pl
    return _pl


def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def get_tf():
    global _tf
    if _tf is None:
        import tensorflow as tf
        _tf = tf
    return _tf


DEFAULT_DATA_TYPES = (
    "numpy.ndarray",
    "pandas.DataFrame",
    "polars.DataFrame",
    "pandas.Series",
    "polars.Series",
    "torch.Tensor",
    "tensorflow.Tensor",
)

class DataChecker:
    """
    A utility class for validating ML/data pipeline variables for NaNs, Infs, and Nulls.
 
    Supports numpy arrays, pandas and polars DataFrames/Series, and PyTorch tensors.
    Can be used inline during training loops or data preprocessing to catch corrupt
    values early, either by raising immediately or accumulating a report.
 
    Args:
        sporadic (int | None): Controls how often checks actually run.
            - None: run on every call (default).
            - -1: run only on the very first call, then skip all subsequent ones.
            - N (int > 0): run every Nth call (e.g. sporadic=10 checks every 10th call).
        reportCard (bool): If False (default), raise a RuntimeError on the first bad value.
            If True, collect all issues and print a summary report at the end of each
            inspect() call instead of raising.
 
    """
    def __init__(self, sporadic=None, report_card=False):
        self.report_card = report_card
        self.sporadic = sporadic

        self.passes = 0
        self.report = []

    def __call__(self, vars: list):
        return self.inspect(vars)

    def inspect(self, vars: list) -> None:
        self.passes += 1

        if self.sporadic == -1 and self.passes != 1:
            return
        elif self.sporadic and self.passes % self.sporadic != 0:
            return

        self.report = []

        groups = defaultdict(list)

        for var in vars:
            module_name = type(var).__module__

            if module_name.startswith("numpy"):
                groups["np"].append(var)

            elif module_name.startswith("pandas"):
                groups["pd"].append(var)

            elif module_name.startswith("polars"):
                groups["pl"].append(var)

            elif module_name.startswith("torch"):
                groups["torch"].append(var)

            elif module_name.startswith("tensorflow"):
                groups["tf"].append(var)

            else:
                self._handle_err(ERR_MSG['Type'], type(var))

        self.np_check(groups["np"])
        self.pd_check(groups["pd"])
        self.pl_check(groups["pl"])
        self.tf_check(groups["tf"])
        self.torch_check(groups["torch"])

        if self.report:
            print("\n".join(self.report))

    def np_check(self, vars: list):
        if not vars:
            return
        np = get_np()

        for arr in vars:
            if np.any(np.isinf(arr)):
                self._handle_err(ERR_MSG['Inf'], "np.ndarray")

            if np.any(np.isnan(arr)):
                self._handle_err(ERR_MSG['Nan'], "np.ndarray")

    def pd_check(self, vars: list):
        if not vars:
            return
        pd = get_pd()
        np = get_np()

        for df in vars:
            if df.isnull().values.any():
                self._handle_err(ERR_MSG['Nan'], "pd.DataFrame or pd.Series")

            if np.isinf(df.select_dtypes('number')).values.any():
                self._handle_err(ERR_MSG['Inf'], "pd.DataFrame or pd.Series")

    def pl_check(self, vars: list):
        if not vars:
            return
        pl = get_pl()

        for df in vars:
            if isinstance(df, pl.Series):
                df = df.to_frame()

            if any(df.null_count().row(0)):
                self._handle_err(ERR_MSG['Null'], "pl.DataFrame or pl.Series")

            if any(df.select(pl.all().is_nan().any()).row(0)):
                self._handle_err(ERR_MSG['Nan'], "pl.DataFrame or pl.Series")

            if any(df.select(pl.all().is_infinite().any()).row(0)):
                self._handle_err(ERR_MSG['Inf'], "pl.DataFrame or pl.Series")

    def torch_check(self, vars: list):
        if not vars:
            return
        torch = get_torch()

        for t in vars:
            if torch.isinf(t).any():
                self._handle_err(ERR_MSG['Inf'], "torch.Tensor")

            if torch.isnan(t).any():
                self._handle_err(ERR_MSG['Nan'], "torch.Tensor")

    def tf_check(self, vars: list):
        if not vars:
            return
        tf = get_tf()

        for t in vars:
            if any(tf.math.is_inf(t)):
                self._handle_err(ERR_MSG["Inf"], "tf.Tensor")

            if any(tf.math.is_nan(t)):
                self._handle_err(ERR_MSG['Nan'], "tf.Tensor")

    def _handle_err(self, err_msg, var=None):
        msg = err_msg.format(var=var)

        if self.report_card:
            self.report.append(msg)
        else:
            raise RuntimeError(msg)


ERR_MSG = {
    "Nan": "Encountered a {var} that contains NaNs",
    "Inf": "Encountered a {var} that contains inf",
    "Null": "Encountered a {var} that contains Nulls",
    "Type": f"Only {DEFAULT_DATA_TYPES} are supported but encountered " + "{var}",
}