# Online PRNN material

`PRNNMaterial` implements the bias-free encoder/material/softplus-decoder
architecture from [SLIMM-Lab/pyprnn](https://github.com/SLIMM-Lab/pyprnn)
(reference revision `3d9cd6ca703c71b3045166d4a37cb63786c35091`). LibTorch runs
the neural layers on CPU in float64. The material layer uses this repository's
`J2Material`, including its analytical consistent tangent, rather than porting
the Python return mapper. No training or Python process runs inside Jive.

## Build

Extract Jem and Jive 3.0 outside the checkout. Configure and build Jem first,
then Jive, with matching compiler/standard-library ABI settings:

```sh
export JEMDIR=/path/to/jem-3.0
export JIVEDIR=/path/to/jive-3.0
(cd "$JEMDIR" && ./configure --with-threads && make -j4)
(cd "$JIVEDIR" && ./configure && make -j4)
```

Set `LIBTORCH` to a CPU LibTorch distribution with `include/` and `lib/`.
The Python CPU wheel can also supply these files; the tested version is
`torch==2.8.0` from `https://download.pytorch.org/whl/cpu`:

```sh
export LIBTORCH=$(python -c 'import torch; print(torch.__path__[0])')
make -C src -j3 HEADLESS=1 TORCH_CXX11_ABI=1
```

`HEADLESS=1` omits the OpenGL module registration when Jem/Jive were built
without OpenGL. Omit it for a GUI-capable toolkit. Omitting `LIBTORCH` keeps
the original solver build without a PRNN dependency. Different build modes
use separate object directories; clean/rebuild when switching LibTorch
versions or ABI settings. `TORCH_CXX11_ABI` must match **both** LibTorch and
Jem/Jive (`torch.compiled_with_cxx11_abi()` reports the wheel's setting).
Use `CXX_STANDARD=c++20` if a newer LibTorch distribution requires C++20.

## Export pretrained weights

Python `torch.save` checkpoints are not C++ module archives. Convert an
existing pyprnn state dictionary or `utils.Trainer.save()` checkpoint:

```sh
python tools/export_prnn.py checkpoint.pth weights.pt
# Optional: select the best validation weights rather than the saved live model.
python tools/export_prnn.py checkpoint.pth weights.pt --state-key best_state_dict
```

The exporter accepts exactly `fc1.weight` and `fc2.weight`, with no biases,
3 or 6 strain/stress components and a whole number of latent material points.
It preserves the **raw** decoder parameters; C++ applies softplus once when
loading. It exports only weights and strain preprocessing, not pyprnn's
sequence `forward`, which resets and commits history internally. The archive
is versioned and dimensions/finiteness are checked on load. Checkpoints are
read with `weights_only=True`; do not change this to load arbitrary Python
objects.

If training used `utils.Normalizer`, supply the actual training bounds as
`--strain-min a b c --strain-max d e f`. They are not stored in the Trainer
checkpoint and cannot be inferred from weights. Stress outputs remain in
the training units. Local strains/stresses are ordered point-first, then
component: `[xx, yy, xy]` in 2D (engineering shear), or the repository's
six-component Voigt order in 3D.

## Jive configuration

Replace the macro material's `FE2` block with:

```text
material =
{
  type = "PRNN";
  rank = 2;
  state = "PlaneStress";
  weights = "weights.pt";
  material =
  {
    young = 3130.;
    poisson = 0.37;
    yield = "64.80 - 33.60 * exp(x / -0.003407)";
    rmTolerance = 1.e-10;
    rmMaxIter = 100;
  };
};
```

The nested `material` block is the existing J2 constitutive law. Its properties
and the outer state must match those used during training. `rank = 2` supports
`PlaneStress` or `PlaneStrain`; `rank = 3` uses `Solid` and requires a
six-component checkpoint. The public pyprnn checkpoint is plane stress, with
the properties shown above. It is not a trained surrogate for this repo's
plane-strain `fe2_j2` microstructure.

Use `model.matrix0.symmetric = false`: a learned decoder/encoder generally
gives a nonsymmetric tangent, even when each J2 tangent is symmetric.
The implemented tangent is
`softplus(Wdecoder) * blockdiag(C_J2) * Wencoder * diag(strain_scale)`.
There is no autograd boundary across J2; finite differences verify the full
chain in elastic/plastic states and after committed unloading.

`configure` loads weights and material settings during Jive configuration.
`allocPoints` allocates independent latent histories and return-map workspaces
per macro integration point. A Newton `update` evaluates only a trial from
the previous accepted state. Jive's existing `SolidModel` dispatches `COMMIT`
once the increment is accepted and `CANCEL` on rejection. Neither operation
is performed by the network forward evaluation. Stress-only table/output
queries preserve an outstanding Newton trial; untouched points and duplicate
commits do not advance history. Reconfiguration after allocation is rejected.

Jem-managed element workers are supported; each macro point owns its J2
workspace. Keep `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` to avoid nesting large
BLAS worker pools within the element workers. Choose `threads` no larger than
the machine's available processors. A 2D network cannot reconstruct an
untrained out-of-plane stress (plane strain) or strain (plane stress);
the corresponding 3D conversion explicitly reports that unsupported request.

## Plate example

`examples/plate_incs/prnn_j2` uses the geometry, mesh generator, constraints,
loading and output of `fe2_j2`, replacing its macro FE2 material with PRNN.
It uses the public checkpoint's plane-stress material settings and a
nonsymmetric global matrix. Obtain `trained_models/prnn_gp_80_0.pth` from the
checkpoint archive linked in pyprnn's `prnn-demo.ipynb`, export it, then:

```sh
python tools/export_prnn.py /path/to/prnn_gp_80_0.pth /path/to/weights.pt
bash examples/plate_incs/prnn_j2/go.sh /path/to/weights.pt
```

The archive's current HTTPS download is
`https://surfdrive.surf.nl/index.php/s/XyIWSNUKp47jnrY/download`.
Weights are not committed to this repository. This example tests loading and
online simulation; the checkpoint's homogenized response is for its original
training composite, not the `fe2_j2` inclusion model. To model that specific
microstructure, supply its pretrained weights and corresponding J2 settings.

## Validation

```sh
python -m unittest discover -s tests -p 'test_*.py'
python tests/make_prnn_fixtures.py /tmp/prnn-fixtures
make -C tests -j3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 tests/prnn_material_test \
  /tmp/prnn-fixtures/fixture3.pt /tmp/prnn-fixtures/fixture6.pt \
  /tmp/prnn-fixtures/normalized.pt
python tests/compare_pyprnn.py /path/to/pyprnn \
  /path/to/prnn_gp_80_0.pth /path/to/weights.pt
```

The fixtures are explicitly **untrained test data**. They exercise tangents,
normalization, independent macro histories, repeated/interleaved Newton trials,
commit/cancel, duplicate commits, stress-only observations and parallel updates.
The separate upstream comparison uses published weights and checks 70 accepted
load/unload/reload/reverse increments. With CPU LibTorch 2.8 and the supplied
Jem/Jive 3.0 archives, the maximum upstream stress discrepancy was
`1.8e-5 MPa`, consistent with different return-map tolerances, and the plate
run completed all 55 increments with four element workers.
