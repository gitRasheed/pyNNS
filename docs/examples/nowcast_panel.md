# Nowcast Panel

PyNNS does not fetch default live macro data behind `nns_nowcast()`. Use
`nns_nowcast_panel` for deterministic panels, or pass an explicit provider to
`nns_nowcast(fetch=True, provider_backend=...)`.

Run:

```bash
uv run python docs/examples/nowcast_panel.py
```

The script demonstrates:

- ordered mapping input
- monthly date metadata
- forecast date labels
- VAR-backed panel forecast output
