import numpy as np
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple
from importlib.metadata import distributions
from .utils import get_module, normalize
from .bases_available import bases_available
from rgevolve.matching import matching_evolution_matrices, matching_matrices

mu_wet = matching_matrices['WET'].attrs['matching scale']
mu_wet4 = matching_matrices['WET-4'].attrs['matching scale']
mu_wet3 = matching_matrices['WET-3'].attrs['matching scale']
reference_scale = {'SMEFT': mu_wet, 'WET': mu_wet, 'WET-4': mu_wet4, 'WET-3': mu_wet3}
matching_scale = {'SMEFT': mu_wet, 'WET': mu_wet4, 'WET-4': mu_wet3}
matching_basis = {'SMEFT': 'Warsaw up', 'WET': 'JMS', 'WET-4': 'JMS', 'WET-3': 'JMS'}
matching_efts = {'SMEFT': ['WET', 'WET-4', 'WET-3'], 'WET': ['WET-4', 'WET-3'], 'WET-4': ['WET-3']}
matching_sectors = {
    wet_sector: matching_matrices['WET'][wet_sector].attrs['from sector']
    for wet_sector in matching_matrices['WET']
}

efts_available = {}
for source_eft, target_efts in matching_efts.items():
    if source_eft not in efts_available:
        efts_available[source_eft] = [source_eft]
    for target_eft in target_efts:
        if target_eft not in efts_available:
            efts_available[target_eft] = [target_eft]
        efts_available[target_eft].append(source_eft)

installed_distributions = {dist.metadata['Name'] for dist in distributions()}
bases_installed = {
    eft: [
        basis for basis in bases
        if f"rgevolve.{normalize(eft)}.{normalize(basis)}" in installed_distributions
    ] for eft, bases in bases_available.items()
}

@lru_cache(maxsize=None)
def evolution_data(eft, basis):
    module = get_module(eft, basis)
    return module.evolution

@lru_cache(maxsize=None)
def translation_data(eft, basis):
    module = get_module(eft, basis)
    return module.translation

@lru_cache(maxsize=None)
def get_evolution_matrix(eft, basis, sector, scale_id, inverse=False):
    if inverse:
        return evolution_data(eft, basis)['inverse'][sector][scale_id]
    return evolution_data(eft, basis)['regular'][sector][scale_id]

@lru_cache(maxsize=None)
def get_matching_evolution_matrix(eft, sector):
    return matching_evolution_matrices[eft][sector][()]

@lru_cache(maxsize=None)
def get_matching_matrix(eft, sector):
    return matching_matrices[eft][sector][()]

@lru_cache(maxsize=None)
def get_scales(eft, basis):
    return evolution_data(eft, basis)['regular'].attrs['scales']

@lru_cache(maxsize=None)
def get_translator(eft, basis_in, basis_out, sector):
    # `[()]` reads each h5py dataset into a numpy array: callers index the
    # translator with index lists and `np.ix_`, which a Dataset does not support
    if basis_in in {'JMS', 'Warsaw up'}:
        return translation_data(eft, basis_out)[str((basis_in, basis_out))][sector][()]
    elif basis_out in {'JMS', 'Warsaw up'}:
        return translation_data(eft, basis_in)[str((basis_in, basis_out))][sector][()]
    else:
        return (
            translation_data(eft, basis_out)[str((matching_basis[eft], basis_out))][sector][()]
            @ translation_data(eft, basis_in)[str((basis_in, matching_basis[eft]))][sector][()]
        )

@lru_cache(maxsize=None)
def interpolate_from_ref(eft, basis, scale, sector, inverse=False):
    scales = get_scales(eft, basis)
    if list(scales) != sorted(scales):  # scales assumed sorted in ascending order
        raise ValueError(f'RG scales for EFT {eft} are not ordered in ascending order')
    # a scale on the grid up to float rounding gets the tabulated matrix
    on_grid = np.flatnonzero(np.isclose(scales, scale, rtol=1e-9, atol=0))
    if on_grid.size:
        return get_evolution_matrix(eft, basis, sector, int(on_grid[0]), inverse=inverse)
    elif scale < min(scales) or scale > max(scales):
        raise ValueError(f'Scale {scale} outside of available range {min(scales)}-{max(scales)}')
    else:
        scale_idx = np.searchsorted(scales, scale, side='left')
        scale_low = scales[scale_idx-1]
        scale_high = scales[scale_idx]
        mat_low = get_evolution_matrix(eft, basis, sector, scale_idx-1, inverse=inverse)
        mat_high = get_evolution_matrix(eft, basis, sector, scale_idx, inverse=inverse)
        # the same weight for the tabulated matrices and for their tabulated
        # inverses. Off the grid, the interpolated inverse is therefore not
        # exactly the inverse of the interpolated matrix; they differ by
        # about the interpolation error
        scale_factor = np.log(scale/scale_low) / np.log(scale_high/scale_low)
        return mat_low + (mat_high - mat_low) * scale_factor

@lru_cache(maxsize=None)
def run(eft, basis, scale_in, scale_out, sector):
    # run within a certain eft, basis, sector
    if scale_in == scale_out:
        return np.identity(get_evolution_matrix(eft, basis, sector, 0).shape[0])
    if scale_in == reference_scale[eft]:
        return interpolate_from_ref(eft, basis, scale_out, sector)
    if scale_out == reference_scale[eft]:
        return interpolate_from_ref(eft, basis, scale_in, sector, inverse=True)
    return interpolate_from_ref(eft, basis, scale_out, sector) @ interpolate_from_ref(eft, basis, scale_in, sector, inverse=True)

@lru_cache(maxsize=None)
def run_and_translate(eft, basis_in, basis_out, scale_in, scale_out, sector):
    # run within an eft, sector but from one basis to another
    if basis_in == basis_out:
        return run(eft, basis_in, scale_in, scale_out, sector)
    translator = get_translator(eft, basis_in, basis_out, sector)
    if scale_in == reference_scale[eft] and scale_out == reference_scale[eft]:
        return translator
    if scale_in == reference_scale[eft] and scale_out != reference_scale[eft]:
        return (
            run(eft, basis_out, reference_scale[eft], scale_out, sector)
            @ translator
        )
    if scale_in != reference_scale[eft] and scale_out == reference_scale[eft]:
        return (
            translator
            @ run(eft, basis_in, scale_in, reference_scale[eft], sector)
        )
    return (
        run(eft, basis_out, reference_scale[eft], scale_out, sector)
        @ translator
        @ run(eft, basis_in, scale_in, reference_scale[eft], sector)
    )

@lru_cache(maxsize=None)
def _run_and_match_sector(eft_in, eft_out, basis_in, basis_out, scale_in, scale_out, sector_out):
    if eft_in == eft_out:
        return run_and_translate(eft_in, basis_in, basis_out, scale_in, scale_out, sector_out)
    if eft_out not in reference_scale.keys():
        raise ValueError(f"Unknown EFT {eft_out}")
    if eft_in not in efts_available.get(eft_out, []):
        raise ValueError(f"No matching path from EFT {eft_in!r} to EFT {eft_out!r}")
    sector_in = matching_sectors[sector_out] if eft_in == 'SMEFT' else sector_out
    run_and_match_matrix = run_and_translate(eft_in, basis_in, matching_basis[eft_in], scale_in, reference_scale[eft_in], sector_in)
    if eft_in != 'SMEFT':
        # `get_matching_evolution_matrix` runs from reference scale to matching scale
        # not needed in SMEFT: matching_scale['SMEFT'] == reference_scale['SMEFT']
        run_and_match_matrix = get_matching_evolution_matrix(eft_in, sector_in) @ run_and_match_matrix
    for eft in matching_efts[eft_in]:
        run_and_match_matrix = get_matching_matrix(eft, sector_out) @ run_and_match_matrix
        if eft == eft_out:
            return (
                run_and_translate(eft, matching_basis[eft], basis_out, reference_scale[eft], scale_out, sector_out)
                @ run_and_match_matrix
            )
        else:
            run_and_match_matrix = get_matching_evolution_matrix(eft, sector_out) @ run_and_match_matrix


def _input_sector(eft_in: str, eft_out: str, sector_out: str) -> str:
    """Return the input sector implied by an output sector.

    Single-sector today; future bases may map one sector_out
    to multiple input sectors.
    """
    if eft_in == eft_out:
        return sector_out
    if eft_in == 'SMEFT':
        return matching_sectors[sector_out]
    return sector_out


def _wc_indices_in_basis(
    wcs: Sequence[Tuple[str, str]],
    wc_basis: List[Tuple[str, str]],
    label: str,
) -> List[int]:
    """Validate that ``wcs`` is a duplicate-free subset of ``wc_basis``
    and return their indices in ``wc_basis``.

    Raises ValueError on duplicate or unknown WCs.
    """
    wc_to_idx = {wc: idx for idx, wc in enumerate(wc_basis)}
    wcs_seen: set = set()
    indices: List[int] = []
    for wc in wcs:
        if wc in wcs_seen:
            raise ValueError(f"Duplicate Wilson coefficient {wc!r}")
        wcs_seen.add(wc)
        if wc not in wc_to_idx:
            raise ValueError(f"Wilson coefficient {wc!r} not in {label}")
        indices.append(wc_to_idx[wc])
    return indices


@lru_cache(maxsize=None)
def _wc_to_sector_and_idx_map(
    eft: str,
    basis: str,
) -> Dict[Tuple[str, str], Tuple[str, int]]:
    """Map each WC to (sector, block_idx) for a given (eft, basis).

    ``block_idx`` is the WC's index in ``get_wc_basis(eft, basis, sector)``.
    Sectors partition the basis, so each WC maps to exactly one sector.
    """
    m: Dict[Tuple[str, str], Tuple[str, int]] = {}
    for sector in evolution_data(eft, basis)['regular'].keys():
        for idx, wc in enumerate(get_wc_basis(eft, basis, sector)):
            m[wc] = (sector, idx)
    return m


def _group_wcs_by_sector(
    eft: str,
    basis: str,
    wcs: Sequence[Tuple[str, str]],
) -> Dict[str, List[Tuple[int, int]]]:
    """Group WCs by sector.

    Returns ``{sector: [(block_idx, M_idx), ...]}`` where ``block_idx``
    is the WC's index in ``get_wc_basis(eft, basis, sector)`` (i.e. its
    position in the per-sector block matrix) and ``M_idx`` is its
    position in the input ``wcs`` list (i.e. its position in the
    assembled result matrix).

    Raises ValueError on duplicate WCs or WCs not present in any
    sector of (eft, basis).
    """
    wc_map = _wc_to_sector_and_idx_map(eft, basis)
    grouped: Dict[str, List[Tuple[int, int]]] = {}
    wcs_seen: set = set()
    for M_idx, wc in enumerate(wcs):
        if wc in wcs_seen:
            raise ValueError(f"Duplicate Wilson coefficient {wc!r}")
        wcs_seen.add(wc)
        if wc not in wc_map:
            raise ValueError(
                f"Wilson coefficient {wc!r} not found in any sector "
                f"of basis '{basis}' of EFT '{eft}'"
            )
        sector, block_idx = wc_map[wc]
        grouped.setdefault(sector, []).append((block_idx, M_idx))
    return grouped


@lru_cache(maxsize=None)
def _run_and_match_impl(
    eft_in: str,
    eft_out: str,
    basis_in: str,
    basis_out: str,
    scale_in: float,
    scale_out: float,
    sector_out: Optional[str],
    wcs_in: Optional[Tuple[Tuple[str, str], ...]],
    wcs_out: Optional[Tuple[Tuple[str, str], ...]],
) -> np.ndarray:
    if sector_out is None and (wcs_in is None or wcs_out is None):
        raise ValueError(
            "When sector_out is None, both wcs_in and wcs_out must be provided."
        )
    if sector_out is not None:
        # Mode A — per-sector block, optional row/col subsetting
        M = _run_and_match_sector(
            eft_in, eft_out, basis_in, basis_out, scale_in, scale_out, sector_out,
        )
        if wcs_in is None and wcs_out is None:
            return M
        if wcs_out is not None:
            wc_basis_out = get_wc_basis(eft_out, basis_out, sector_out)
            row_idx = _wc_indices_in_basis(
                wcs_out, wc_basis_out,
                f"sector '{sector_out}' of basis '{basis_out}' of EFT '{eft_out}'",
            )
            M = M[row_idx, :]
        if wcs_in is not None:
            sector_in = _input_sector(eft_in, eft_out, sector_out)
            wc_basis_in = get_wc_basis(eft_in, basis_in, sector_in)
            col_idx = _wc_indices_in_basis(
                wcs_in, wc_basis_in,
                f"input sector '{sector_in}' of basis '{basis_in}' of EFT '{eft_in}'",
            )
            M = M[:, col_idx]
        return M

    # Mode B — sector_out is None, both lists given
    groups_out = _group_wcs_by_sector(eft_out, basis_out, wcs_out)
    groups_in = _group_wcs_by_sector(eft_in, basis_in, wcs_in)
    M = np.zeros((len(wcs_out), len(wcs_in)))
    for sector_out, pairs_out in groups_out.items():
        sector_in = _input_sector(eft_in, eft_out, sector_out)
        pairs_in = groups_in.get(sector_in, [])
        if not pairs_in:
            continue  # cross-sector entries stay zero
        block = _run_and_match_sector(
            eft_in, eft_out, basis_in, basis_out, scale_in, scale_out, sector_out,
        )
        block_idx_out = np.fromiter((p[0] for p in pairs_out), dtype=int)
        M_idx_out = np.fromiter((p[1] for p in pairs_out), dtype=int)
        block_idx_in = np.fromiter((p[0] for p in pairs_in), dtype=int)
        M_idx_in = np.fromiter((p[1] for p in pairs_in), dtype=int)
        M[np.ix_(M_idx_out, M_idx_in)] = block[np.ix_(block_idx_out, block_idx_in)]
    return M


_WC_FORM = "(name, 'R') or (name, 'I'), as returned by get_wc_basis"


def _is_wc_pair(wc) -> bool:
    """True for a pair ``(name, 'R' | 'I')``, given as a tuple or a list."""
    return (
        isinstance(wc, (tuple, list)) and len(wc) == 2
        and isinstance(wc[0], str)
        and isinstance(wc[1], str) and wc[1] in ('R', 'I')
    )


def _canonical_wcs(wcs, argument: str) -> Optional[Tuple[Tuple[str, str], ...]]:
    """Return ``wcs`` as a tuple of ``(name, 'R' | 'I')`` tuples, or None.

    ``wcs`` is a list or a tuple of such pairs, and each pair is a tuple
    or a list. Loads no data, so ``run_and_match`` calls it before any
    lookup (rgevolve/rgevolve-core#1).

    Raises ValueError, naming ``argument``, if ``wcs`` is not a list or a
    tuple, or at its first entry that is not such a pair.
    """
    if wcs is None:
        return None
    if isinstance(wcs, str):
        raise ValueError(
            f"{argument} is the string {wcs!r}, but it must be a list of "
            f"pairs {_WC_FORM}, e.g. [({wcs!r}, 'R')]"
        )
    if not isinstance(wcs, (list, tuple)):
        raise ValueError(
            f"{argument} is {wcs!r}, but it must be a list or a tuple of "
            f"pairs {_WC_FORM}"
        )
    canonical = []
    for position, wc in enumerate(wcs):
        if not _is_wc_pair(wc):
            if _is_wc_pair(wcs):
                hint = (
                    f"; to pass the single pair {tuple(wcs)!r}, "
                    f"write [{tuple(wcs)!r}]"
                )
            elif isinstance(wc, str):
                hint = f", e.g. ({wc!r}, 'R') for its real part"
            else:
                hint = ""
            raise ValueError(
                f"{argument}[{position}] is {wc!r}, but each entry of "
                f"{argument} must be a pair {_WC_FORM}{hint}"
            )
        canonical.append((wc[0], wc[1]))
    return tuple(canonical)


def run_and_match(
    eft_in: str,
    eft_out: str,
    basis_in: str,
    basis_out: str,
    scale_in: float,
    scale_out: float,
    sector_out: Optional[str] = None,
    wcs_in: Optional[Sequence[Tuple[str, str]]] = None,
    wcs_out: Optional[Sequence[Tuple[str, str]]] = None,
) -> np.ndarray:
    """Returns the matrix that takes a vector of input Wilson coefficients
    (at ``scale_in`` in ``eft_in`` / ``basis_in``) to a vector of output
    Wilson coefficients (at ``scale_out`` in ``eft_out`` / ``basis_out``)
    via RG evolution and, if ``eft_in != eft_out``, matching, and if ``basis_in != basis_out``, translation.

    Two operating modes:

    * ``sector_out`` given (per-sector mode). The full per-sector block
      is built; ``wcs_in`` / ``wcs_out`` are optional and, if given,
      subset and reorder the matrix's columns / rows. ``wcs_out`` must
      be a subset of ``get_wc_basis(eft_out, basis_out, sector_out)``;
      ``wcs_in`` must be a subset of the canonical basis of the input
      sector implied by ``sector_out``.

    * ``sector_out=None`` (full-EFT mode). Both ``wcs_in`` and
      ``wcs_out`` are mandatory and may span multiple sectors. The
      returned matrix has shape ``(len(wcs_out), len(wcs_in))``;
      entries connecting WCs that don't lie in matched sectors are
      exactly ``0``.

    Each entry of ``wcs_in`` / ``wcs_out`` is a pair ``(name, 'R')`` or
    ``(name, 'I')``, for the real or the imaginary part of a Wilson
    coefficient, as ``get_wc_basis`` returns them: for example
    ``[('C9_bsmumu', 'R'), ('C9_bsmumu', 'I')]``. Either list must be a
    list or a tuple, and its order is the order of the result's columns
    (``wcs_in``) or rows (``wcs_out``). Each pair may be a tuple or a
    list. Their form is checked before any data is loaded. Results are
    cached per unique call, with both lists turned into tuples of tuples.

    Raises ``ValueError`` if ``wcs_in`` or ``wcs_out`` is not a list or a
    tuple of such pairs, on duplicate WCs, WCs outside the allowed set, or
    invalid argument combinations.

    Note (future): future bases may map one ``sector_out`` to multiple
    input sectors. This release supports a single input sector per
    ``sector_out``.
    """
    wcs_in = _canonical_wcs(wcs_in, 'wcs_in')
    wcs_out = _canonical_wcs(wcs_out, 'wcs_out')
    return _run_and_match_impl(
        eft_in, eft_out, basis_in, basis_out, scale_in, scale_out,
        sector_out, wcs_in, wcs_out,
    )


# useful functions for Wilson coefficients

@lru_cache(maxsize=None)
def get_wc_basis(eft, basis, sector=None, split_re_im=True):
    """Get the list of nonredundant WCs in a given sector of an EFT and basis."""
    sectors_available = set(evolution_data(eft, basis)['regular'].keys())
    if sector:
        if sector not in sectors_available:
            raise ValueError(f"Sector {sector} not found in basis {basis} of eft {eft}")
        sectors = [sector]
    else:
        sectors = sectors_available
    basis_list = []
    for sec in sectors:
        wilson_coefficients = evolution_data(eft, basis)['regular'][sec].attrs['Wilson coefficients']
        for wc in wilson_coefficients:
            if split_re_im:
                basis_list.append((wc[0], 'R'))
                if wc[1] == 'C':
                    basis_list.append((wc[0], 'I'))
            else:
                basis_list.append(wc[0])
    return sorted(basis_list)

