# rgevolve-core

A package providing the core tools for loading and processing Renormalization Group Evolution matrices.

`rgevolve-core` is the core runtime of the **rgevolve** ecosystem — a set of Python namespace packages for fast renormalization group evolution of Wilson coefficients in the SMEFT and the WET using the evolution matrix formalism. It loads cross-EFT matching matrices bundled with the package and discovers companion `rgevolve.<eft>.<basis>` distributions at runtime via [`importlib.metadata`](https://docs.python.org/3/library/importlib.metadata.html).

See the [rgevolve organization](https://github.com/rgevolve) for the full set of available EFT/basis packages, and the [`rgevolve` meta-package](https://github.com/rgevolve/rgevolve) for installing the core together with all companions in lockstep.

## Installation

```bash
pip install rgevolve-core
```

To install the core package together with all available EFT/basis companion packages at once, use the meta-package:

```bash
pip install rgevolve
```

## License

`rgevolve-core` is licensed under the MIT License — see [`LICENSE`](LICENSE).
