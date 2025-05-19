import numpy as np
from functools import lru_cache
import scipy
from typing import List
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
    if basis_in in {'JMS', 'Warsaw up'}:
        return translation_data(eft, basis_out)[str((basis_in, basis_out))][sector]
    elif basis_out in {'JMS', 'Warsaw up'}:
        return translation_data(eft, basis_in)[str((basis_in, basis_out))][sector]
    else:
        return (
            translation_data(eft, basis_out)[str((matching_basis[eft], basis_out))][sector]
            @ translation_data(eft, basis_in)[str((basis_in, matching_basis[eft]))][sector]
        )

@lru_cache(maxsize=None)
def interpolate_from_ref(eft, basis, scale, sector, inverse=False):
    scales = get_scales(eft, basis)
    if list(scales) != sorted(scales):  # scales assumed sorted in ascending order
        raise ValueError(f'RG scales for EFT {eft} are not ordered in ascending order')
    if scale in scales:
        return get_evolution_matrix(eft, basis, sector, np.where(scales==scale)[0][0], inverse=inverse)
    elif scale < min(scales) or scale > max(scales):
        raise ValueError(f'Scale {scale} outside of available range {min(scales)}-{max(scales)}')
    else:
        scale_idx = np.searchsorted(scales, scale, side='left')
        scale_low = scales[scale_idx-1]
        scale_high = scales[scale_idx]
        mat_low = get_evolution_matrix(eft, basis, sector, scale_idx-1, inverse=inverse)
        mat_high = get_evolution_matrix(eft, basis, sector, scale_idx, inverse=inverse)
        if inverse:
            scale_factor = np.log(scale_high/scale_low) / np.log(scale/scale_low)
        else:
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
def run_and_match(eft_in, eft_out, basis_in, basis_out, scale_in, scale_out, sector_out):
    if eft_in == eft_out:
        return run_and_translate(eft_in, basis_in, basis_out, scale_in, scale_out, sector_out)
    if eft_out not in reference_scale.keys():
        raise ValueError(f"Unknown EFT {eft_out}")
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

def get_wc_mask(eft, basis, sector, wcs):
    wc_basis_sector = get_wc_basis(eft, basis, sector)
    if not all([wc in wc_basis_sector for wc in wcs]):
        raise ValueError(f"Invalid coefficients in sector {sector} of basis {basis} of EFT {eft}")
    return np.array([wc in wcs for wc in wc_basis_sector])

def get_sector_indices(eft: str, basis: str, sectors: List[str]) -> np.ndarray:
    basis_full = get_wc_basis(eft, basis)
    return np.concatenate([
        [basis_full.index(wc) for wc in get_wc_basis(eft, basis, sector)]
        for sector in sectors
    ])

####################################################################################################
## DEPRECATED
####################################################################################################
def wc_real_to_real_sector_idx(eft, basis, sector):
    basis_real_full = get_wc_basis(eft, basis, split_re_im=True)
    basis_real_sector = get_wc_basis(eft, basis, sector=sector, split_re_im=True)
    idx = np.array([basis_real_full.index(c) for c in basis_real_sector])
    return idx

def compute_observable_RGs_from_scale(sectors_eft_info, eft_in, basis_in, scale_in):
    RGs = []
    for eft_obs, basis_obs, scale_obs, sector_obs, coeffs_obs in sectors_eft_info:
        m = run_and_match(
            eft_in=eft_in, eft_out=eft_obs,
            basis_in=basis_in, basis_out=basis_obs,
            scale_in=scale_in, scale_out=scale_obs,
            sector_out=sector_obs,
        )
        coeff_mask_sector = get_wc_mask(eft_obs, basis_obs, sector_obs, coeffs_obs)
        RGs.append(m[coeff_mask_sector])
    return RGs

def compute_observable_RGs(sectors_eft_info, eft_in, basis_in):
    RG_scales = get_scales(eft_in, basis_in)
    RGs = []
    for scale in RG_scales:
        if eft_in == 'SMEFT':
            RGs.append(np.concatenate(compute_observable_RGs_from_scale(sectors_eft_info, eft_in, basis_in, scale)))
        elif eft_in == 'WET' and 'WET' in sectors_eft_info[0][0]:  # TODO: support for WET-4 and WET-3
            RGs.append(scipy.linalg.block_diag(*compute_observable_RGs_from_scale(sectors_eft_info, eft_in, basis_in, scale)))
    RGs = np.array(RGs)
    return RG_scales, RGs
####################################################################################################
