l2_normalize raises TypeError when given a plain Python list

`l2_normalize` is documented as accepting any array-like, but passing a list crashes:

```python
>>> from utils.data_loader import l2_normalize
>>> l2_normalize([3.0, 4.0])
Traceback (most recent call last):
  File "src/utils/data_loader.py", line 35, in l2_normalize
    norm = np.sqrt((data ** 2).sum())
TypeError: unsupported operand type(s) for ** or pow(): 'list' and 'int'
```

Passing `np.array([3.0, 4.0])` works and returns `[0.6, 0.8]`. The other helpers in this module
(`drop_missing`, `clip_outliers`, `min_max_scale`) already convert their input, so lists work there.
