# Apptainer image

[`apptainer.def`](../apptainer.def) is a complete build recipe for the Python
PRNN branch. It builds Jem 3.0, Jive 3.0, the standalone native material
extension and the embedded FE2 solver inside Ubuntu 24.04. It includes Python
3.12, CPU PyTorch 2.8.0, the Python training utilities, Gmsh, plotting tools and
the public pyprnn checkpoint. No host-built libraries or Python environment
are needed. All build and runtime scripts are contained in the definition.

The two supplied toolkit ZIPs are **build inputs**: the definition does not
embed their contents or depend on a toolkit download service. Keep them outside
Git. Their SHA-256 hashes are checked before extraction. Building also needs
Internet access for Ubuntu packages, GitHub, PyPI, the CPU PyTorch index and
the public checkpoint on SURFdrive.

## Build

Use Apptainer 1.3 or later, with working fakeroot support, or build as root.
For a root build, omit `--fakeroot` from the command below.
The default archive paths are `jem-3.0.zip` and `jive-3.0.zip` in the current
directory; explicit paths can be supplied:

```sh
apptainer build --fakeroot \
  --build-arg JEM_ARCHIVE=/absolute/path/jem-3.0.zip \
  --build-arg JIVE_ARCHIVE=/absolute/path/jive-3.0.zip \
  --build-arg BUILD_JOBS=4 \
  fe2.sif apptainer.def
```

The default code ref is `feat/python-prnn-bridge` from `SLIMM-Lab/fe2`.
It fetches the published branch, so uncommitted local changes are not included.
For reproducible source selection, add `--build-arg FE2_REF=FULL_COMMIT_SHA`;
tags and branch names are also supported. The resolved commit and installed
Python package versions are recorded in `/etc/fe2/source-commit` and
`/etc/fe2/python-packages.txt` inside the image. If using different toolkit
archives, override `JEM_SHA256` and `JIVE_SHA256` with their expected hashes.

The recipe builds a headless solver with threads enabled and MPI disabled.
It retains sources and build tools. Generic compiler flags avoid tuning to the
build machine; build the image on the CPU architecture where it will run.
No GPU or LibTorch C++ installation is required.

The `%test` section runs the nine Python bridge integration tests and the
55-increment pretrained plate example. It checks convergence and finite
reactions and removes its temporary output. Run those checks again with:

```sh
apptainer test --cleanenv fe2.sif
```

## Use

The runtime sets `JEMDIR`, `JIVEDIR`, `PATH` and `PYTHONPATH` for the built
toolkits, solver and Python package. CPU thread pools default to one thread;
plotting uses a noninteractive backend with its cache in writable `/tmp`.
Use `--cleanenv` to avoid inheriting an unrelated host Python environment.

Run a simulation using files in a host directory:

```sh
apptainer run --cleanenv --bind "$PWD":/work --pwd /work \
  fe2.sif macro.pro
```

The runscript forwards arguments directly to `fe2`. The solver is also
available as `apptainer exec fe2.sif fe2 ...`. Use paths visible **inside** the
container in Jive properties, including mesh, checkpoint and custom Python
module paths. A SIF is read-only: run cases in a writable bind mount rather
than under `/opt/fe2`.

The bundled example runner copies the input file, creates the mesh without a
GUI, selects the bundled checkpoint and writes all results to a new empty
directory:

```sh
apptainer exec --cleanenv --bind "$PWD":/work --pwd /work \
  fe2.sif fe2-prnn-example /work/results
```

To use another compatible checkpoint, pass its container-visible path as the
second argument:

```sh
apptainer exec --cleanenv --bind "$PWD":/work --pwd /work \
  fe2.sif fe2-prnn-example /work/results-new /work/my-checkpoint.pth
```

The bundled checkpoint is `/opt/fe2/weights/prnn_gp_80_0.pth` and uses its
plane-stress training material. Matching the
original `fe2_j2` inclusion micromodel requires corresponding weights and
material properties; see [the PRNN documentation](python-prnn.md).

Python tools use the same native material extension and package as the solver:

```sh
apptainer exec --cleanenv fe2.sif python -c \
  'from _fe2_material import MaterialFactory; print(MaterialFactory.types())'
apptainer exec --cleanenv --bind "$PWD":/work --pwd /work \
  fe2.sif python train.py
```

For architecture experiments, bind a directory containing a custom loader and
set `module`, `loader` and `pythonPath` in the Jive properties. Editing those
Python modules does not require rebuilding the C++ solver. Changes to the
native C++ materials require rebuilding the image.
