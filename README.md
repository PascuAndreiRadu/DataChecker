# DataChecker

A lightweight, zero-dependency validation utility that catches **NaNs, Infs, and Nulls** across every major ML data type — before they silently corrupt your training run.

---

## Table of Contents

- [Why use this?](#why-use-this)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Modes](#usage-modes)
  - [Raise immediately (default)](#raise-immediately-default)
  - [Report card mode](#report-card-mode)
  - [Sporadic checking](#sporadic-checking)
- [API Reference](#api-reference)
  - [Constructor](#constructor)
  - [inspect() / \_\_call\_\_()](#inspect--__call__)
- [Supported Types](#supported-types)
- [Error Messages](#error-messages)
- [Recipes](#recipes)

---

## Why use this?

NaNs and Infs are silent killers in ML pipelines. A corrupt batch can propagate through an entire training run, produce nonsensical losses, and only surface hours later — or never. `DataChecker` gives you a single drop-in call that validates every variable you care about, regardless of whether it's a NumPy array, a Pandas DataFrame, a Polars Series, or a PyTorch / TensorFlow tensor.

---

## Installation

```bash
pip install numpy pandas polars torch tensorflow
```

`DataChecker` has no package of its own — just drop `data_checker.py` into your project and import it directly.

---

## Quick Start

```python
from data_checker import DataChecker
import numpy as np
import pandas as pd

checker = DataChecker()

X = np.array([[1.0, 2.0], [3.0, 4.0]])
y = pd.Series([0, 1, 0, 1])

checker([X, y])  # passes silently — all clean
```

```python
X_corrupt = np.array([[1.0, float("nan")], [float("inf"), 4.0]])
checker([X_corrupt])
# RuntimeError: Encountered a np.ndarray that contains Nans
```

---

## Usage Modes

### Raise immediately (default)

The default behaviour — the moment a bad value is found, a `RuntimeError` is raised and execution stops. Ideal for catching problems as early as possible during development.

```python
checker = DataChecker()

for epoch in range(100):
    X_batch, y_batch = get_batch()
    checker([X_batch, y_batch])   # raises on first bad batch
    train_step(X_batch, y_batch)
```

---

### Report card mode

Set `report_card=True` to collect all issues across the entire variable list and print a summary at the end of the call rather than raising. Useful when you want to audit a full dataset without stopping at the first problem.

```python
checker = DataChecker(report_card=True)

checker([X_train, y_train, X_test, y_test])
# Encountered a np.ndarray that contains Nans
# Encountered a pd.Dataframe or pd.Series that contains inf
```

No exception is raised — all issues are printed together so you can see the full picture at once.

---

### Sporadic checking

Running full validation on every step of a tight training loop can be expensive. The `sporadic` parameter lets you control how often checks actually execute.

| Value | Behaviour |
|---|---|
| `None` (default) | Check on every call |
| `-1` | Check only on the **first** call, skip all subsequent ones |
| `N` (int > 0) | Check every **N**th call |

```python
# Check only once — useful for validating the dataset at startup
checker = DataChecker(sporadic=-1)

# Check every 10th batch — low overhead in a long training loop
checker = DataChecker(sporadic=10)

for step in range(10_000):
    checker([X_batch, y_batch])   # only runs on steps 0, 10, 20, ...
    train_step(X_batch, y_batch)
```

---

## API Reference

### Constructor

```python
DataChecker(sporadic=None, report_card=False)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sporadic` | `int \| None` | `None` | Controls check frequency. `None` = every call, `-1` = first call only, `N` = every Nth call |
| `report_card` | `bool` | `False` | If `True`, collect and print all issues instead of raising on the first one |

---

### `inspect()` / `__call__()`

```python
checker.inspect(vars: list) -> None
checker(vars: list) -> None          # identical — __call__ delegates to inspect()
```

Validates every item in `vars`. Items are grouped by type and checked in parallel, so mixed-type lists work naturally.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `vars` | `list` | A list of any mix of supported data objects to validate |

**Raises**

`RuntimeError` — if a bad value is found and `report_card=False`.

**Prints** a multi-line report if `report_card=True` and any issues are found.

---

## Supported Types

| Type | NaN | Inf | Null |
|---|:---:|:---:|:---:|
| `np.ndarray` | ✅ | ✅ | |
| `pd.DataFrame` | ✅ | ✅ | ✅ |
| `pd.Series` | ✅ | ✅ | ✅ |
| `pl.DataFrame` | ✅ | ✅ | ✅ |
| `pl.Series` | ✅ | ✅ | ✅ |
| `torch.Tensor` | ✅ | ✅ | |
| `tf.Tensor` | ✅ | ✅ | |

Passing any other type triggers an unsupported-type error listing the actual type encountered.

---

## Error Messages

| Issue | Message |
|---|---|
| NaN | `Encountered a {type} that contains Nans` |
| Inf | `Encountered a {type} that contains inf` |
| Null | `Encountered a {type} that contains Nulls` |
| Unsupported type | `Only (...supported types...) are supported but encountered {type}` |

---

## Recipes

### Validate a full dataset before training

```python
checker = DataChecker(report_card=True)
checker([X_train, X_test, y_train, y_test])
```

### Lightweight loop validation

```python
checker = DataChecker(sporadic=50)   # only runs every 50 steps

for step, (X_batch, y_batch) in enumerate(dataloader):
    checker([X_batch, y_batch])
    loss = model(X_batch, y_batch)
```

### Mixed framework pipeline

```python
import torch, polars as pl

checker = DataChecker()

embeddings = torch.randn(32, 128)
metadata   = pl.DataFrame({"age": [25, None, 30], "score": [0.9, 0.8, float("inf")]})

checker([embeddings, metadata])
# RuntimeError: Encountered a pl.Dataframe or pl.Series that contains Nulls
```

### One-shot startup check, then silent

```python
# Validate the data pipeline once at the start, never again
checker = DataChecker(sporadic=-1, report_card=True)

for epoch in range(200):
    for X_batch, y_batch in dataloader:
        checker([X_batch, y_batch])   # only actually runs on the very first call
        train_step(X_batch, y_batch)
```
