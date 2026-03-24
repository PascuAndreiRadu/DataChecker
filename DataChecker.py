import numpy as np
import pandas as pd
import polars as pl
import torch
import tensorflow as tf
from collections import defaultdict
import random


DEFAULT_DATA_TYPES=(np.ndarray,pd.DataFrame,pl.DataFrame,pd.Series,pl.Series,torch.Tensor,tf.Tensor)

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
    def __init__(self,sporadic=None,report_card=False):

        self.report_card=report_card
        self.sporadic=sporadic
        self.TypeCheck=None
        
        self.passes=0
        self.report=[]
        self.cntr=0

    def __call__(self,vars:list):
        return self.inspect(vars)

    def inspect(self,vars:list)->None:    
        self.passes+=1

        if self.sporadic == -1 and self.passes !=1:
            return
        elif self.sporadic and self.passes % self.sporadic != 0:
            return
        
        self.report=[]
    
        groups: defaultdict[str, list] = defaultdict(list)
        for index, var in enumerate(vars): 
            match var:
                case np.ndarray():
                    groups["np"].append(var)
                case pd.DataFrame() | pd.Series():
                    groups["pd"].append(var)
                case pl.DataFrame() | pl.Series():
                    groups["pl"].append(var)
                case torch.Tensor():
                    groups["torch"].append(var)
                case tf.Tensor():
                    groups["tf"].append(var)
                case _:
                    self._handle_err(ERR_MSG['Type'],type(var))

        self.np_check(groups['np'])
        self.pd_check(groups['pd'])
        self.pl_check(groups['pl'])
        self.tf_check(groups['tf'])
        self.torch_check(groups['torch'])

        if self.report:
            print("\n".join(self.report))

    def torch_check(self,vars:list):
        for index,arg in enumerate(vars):
            try:
                assert not torch.isinf(arg).any()
            except AssertionError:
                self._handle_err(ERR_MSG['inf'],"torch.Tensor")
            try:
                assert not torch.isnan(arg).any()
            except AssertionError:
                self._handle_err(ERR_MSG['Nan'],"torch.Tensor")

    def pd_check(self,vars:list):
        for df in vars:
            try:
                assert not df.isnull().values.any()
            except AssertionError:
                self._handle_err(ERR_MSG['Nan'],"pd.Dataframe or pd.Series")

            try:
                assert not np.isinf(df.select_dtypes('number')).values.any()
            except AssertionError:
                self._handle_err(ERR_MSG['Inf'],"pd.Dataframe or pd.Series")

    def pl_check(self, vars: list):
        for index, df in enumerate(vars):
            if isinstance(df, pl.Series):
                df = df.to_frame()
            try:
                assert not any(df.null_count().row(0))
            except AssertionError:
                self._handle_err(ERR_MSG['Null'], "pl.Dataframe or pl.Series")

            try:
                assert not any(df.select(pl.all().is_nan().any()).row(0))
            except AssertionError:
                self._handle_err(ERR_MSG['Nan'], "pl.Dataframe or pl.Series")

            try:
                assert not any(df.select(pl.all().is_infinite().any()).row(0))
            except AssertionError:
                self._handle_err(ERR_MSG['Inf'], "pl.Dataframe or pl.Series")
    def np_check(self, vars:list):
        for index,np_array in enumerate(vars):
            try:
                assert not np.any(np.isinf(np_array))
            except AssertionError:
                self._handle_err(ERR_MSG['Inf'],"np.ndarray")
            
            try:
                assert not np.any(np.isnan(np_array))
            except AssertionError:
                self._handle_err(ERR_MSG['Nan'],"np.ndarray")

    def tf_check(self,vars:list):
        for index,tf_tensor in enumerate(vars):
            try:
                assert not any(tf.math.is_inf(tf_tensor))
            except AssertionError:
                self._handle_err(ERR_MSG["Inf"],"tf.Tensor")

            try:
                assert not any(tf.math.is_nan(tf_tensor))
            except AssertionError:
                self._handle_err(ERR_MSG['Nan'],"tf.Tensor")
                
    def _handle_err(self,err_msg,var=None):
        
        msg=err_msg.format(var=var)
        if self.report_card:
            self.report.append(msg)
        else:
            raise RuntimeError(msg)

ERR_MSG={   
    "Nan":"Encountered a {var} that contains Nans",
    "Inf":"Encountered a {var} that contains inf",
    'Null' : "Encountered a {var} that contains Nulls",
    "Type":f"Only {DEFAULT_DATA_TYPES} are supported but encountered " + "{var}",
}