Copied from https://github.com/SLIMM-Lab/pyprnn at
`3d9cd6ca703c71b3045166d4a37cb63786c35091`:

- `prnn.py` (adapted for the generic C++ material layer and online lifecycle)
- `utils.py` (whitespace CSV parsing updated for current pandas)
- `rnn.py` (architecture examples; whitespace normalization only)
- `UPSTREAM_README.md` and `LICENSE` (GPL-3.0)

The Python J2 return mapper is deliberately replaced by `_fe2_material`;
training and online simulation use the same native material implementation.
New adapter files provide first-order autograd through current tangents and
past-history replay, plus an embedded-runtime interface. No trained weights
or upstream datasets are bundled.
