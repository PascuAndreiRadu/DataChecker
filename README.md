# DataChecker
A utility class for validating ML/data pipeline variables for NaNs, Infs, and Nulls.

Supports numpy arrays, pandas and polars DataFrames/Series, and PyTorch tensors.
Can be used inline during training loops or data preprocessing to catch corrupt
values early, either by raising immediately or accumulating a report.
 
