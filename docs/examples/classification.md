# Classification

R NNS classification paths return class codes. PyNNS follows that convention:
predictions are numeric codes unless you map them back to labels yourself.

Run:

```bash
uv run python docs/examples/classification.py
```

The script demonstrates:

- `nns_reg(..., type="class")`
- numeric class-code predictions
- explicit class levels
