import unittest

import numpy as np


class TestPackage(unittest.TestCase):

    def test_import(self):
        try:
            import rgevolve.tools as tools
        except ImportError:
            self.fail("Importing rgevolve.tools failed.")
        else:
            self.assertIsNotNone(tools)

    def test_invalid_eft_direction_wet_to_smeft(self):
        # WET cannot run-and-match to SMEFT (matching only goes downward).
        from rgevolve.tools import run_and_match
        with self.assertRaises(ValueError):
            run_and_match('WET', 'SMEFT', 'JMS', 'Warsaw', 80.0, 1000.0, 'cbenue')

    def test_invalid_eft_direction_wet3_to_wet(self):
        # WET-3 is the most downstream EFT; matching cannot go upstream to WET.
        from rgevolve.tools import run_and_match
        with self.assertRaises(ValueError):
            run_and_match('WET-3', 'WET', 'JMS', 'JMS', 5.0, 80.0, 'cbenue')


SCALE_IN = 1000.0
SCALE_OUT = 80.0


def _has_companions():
    """The run_and_match tests below rely on the SMEFT-Warsaw and WET-JMS
    companion packages being installed."""
    try:
        from rgevolve.tools import bases_installed
    except ImportError:
        return False
    return (
        'Warsaw' in bases_installed.get('SMEFT', [])
        and 'JMS' in bases_installed.get('WET', [])
    )


_REQUIRES_COMPANIONS = (
    "Requires companion packages rgevolve.smeft.warsaw and rgevolve.wet.jms"
)


def _pick_smeft_to_wet_sectors():
    """Find two WET output sectors with distinct SMEFT input sectors,
    plus a third SMEFT sector unrelated to either."""
    from rgevolve.tools.functions import matching_sectors, evolution_data
    sA = sB = None
    for s, smeft in matching_sectors.items():
        if sA is None:
            sA = s
            continue
        if matching_sectors[sA] != smeft:
            sB = s
            break
    smeft_sectors = list(evolution_data('SMEFT', 'Warsaw')['regular'].keys())
    used = {matching_sectors[sA], matching_sectors[sB]}
    s_unrelated = next(s for s in smeft_sectors if s not in used)
    return sA, sB, s_unrelated


@unittest.skipUnless(_has_companions(), _REQUIRES_COMPANIONS)
class TestRunAndMatchModeA(unittest.TestCase):
    """sector_out given; wcs_in / wcs_out optional."""

    def test_identity_cross_eft(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import _run_and_match_sector, matching_sectors
        s = next(iter(matching_sectors))
        a = run_and_match('SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, s)
        b = _run_and_match_sector('SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, s)
        self.assertTrue(np.array_equal(a, b))

    def test_identity_same_eft(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import _run_and_match_sector, matching_sectors
        s = next(iter(matching_sectors))
        a = run_and_match('WET', 'WET', 'JMS', 'JMS', SCALE_OUT, 5.0, s)
        b = _run_and_match_sector('WET', 'WET', 'JMS', 'JMS', SCALE_OUT, 5.0, s)
        self.assertTrue(np.array_equal(a, b))

    def test_subset_and_reorder_both_axes(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import _run_and_match_sector, matching_sectors, get_wc_basis
        sector_out = next(iter(matching_sectors))
        sector_in = matching_sectors[sector_out]
        wc_basis_out = get_wc_basis('WET', 'JMS', sector_out)
        wc_basis_in = get_wc_basis('SMEFT', 'Warsaw', sector_in)
        wcs_out = list(reversed(wc_basis_out))[:3]
        wcs_in = list(reversed(wc_basis_in))[:3]
        full = _run_and_match_sector(
            'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, sector_out,
        )
        row_idx = [wc_basis_out.index(wc) for wc in wcs_out]
        col_idx = [wc_basis_in.index(wc) for wc in wcs_in]
        expected = full[np.ix_(row_idx, col_idx)]
        M = run_and_match(
            'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, sector_out,
            wcs_in=wcs_in, wcs_out=wcs_out,
        )
        self.assertEqual(M.shape, (3, 3))
        self.assertTrue(np.array_equal(M, expected))

    def test_only_wcs_out(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import _run_and_match_sector, matching_sectors, get_wc_basis
        sector_out = next(iter(matching_sectors))
        wc_basis_out = get_wc_basis('WET', 'JMS', sector_out)
        wcs_out = list(reversed(wc_basis_out))[:2]
        full = _run_and_match_sector(
            'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, sector_out,
        )
        M = run_and_match(
            'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, sector_out,
            wcs_out=wcs_out,
        )
        self.assertEqual(M.shape, (len(wcs_out), full.shape[1]))
        for i, wc in enumerate(wcs_out):
            self.assertTrue(np.array_equal(M[i, :], full[wc_basis_out.index(wc), :]))

    def test_only_wcs_in(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import _run_and_match_sector, matching_sectors, get_wc_basis
        sector_out = next(iter(matching_sectors))
        sector_in = matching_sectors[sector_out]
        wc_basis_in = get_wc_basis('SMEFT', 'Warsaw', sector_in)
        wcs_in = list(reversed(wc_basis_in))[:2]
        full = _run_and_match_sector(
            'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, sector_out,
        )
        M = run_and_match(
            'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, sector_out,
            wcs_in=wcs_in,
        )
        self.assertEqual(M.shape, (full.shape[0], len(wcs_in)))
        for j, wc in enumerate(wcs_in):
            self.assertTrue(np.array_equal(M[:, j], full[:, wc_basis_in.index(wc)]))

    def test_error_wcs_out_wrong_sector(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import matching_sectors, get_wc_basis
        sA, sB, _ = _pick_smeft_to_wet_sectors()
        # WC from sB passed as wcs_out for sector_out=sA
        wc_from_other = get_wc_basis('WET', 'JMS', sB)[0]
        with self.assertRaises(ValueError):
            run_and_match(
                'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, sA,
                wcs_out=[wc_from_other],
            )

    def test_error_wcs_in_wrong_sector(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import matching_sectors, get_wc_basis
        sA, sB, _ = _pick_smeft_to_wet_sectors()
        # WC from a different SMEFT sector
        wc_from_other = get_wc_basis('SMEFT', 'Warsaw', matching_sectors[sB])[0]
        with self.assertRaises(ValueError):
            run_and_match(
                'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, sA,
                wcs_in=[wc_from_other],
            )

    def test_error_duplicate_in_wcs_out(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import matching_sectors, get_wc_basis
        s_out = next(iter(matching_sectors))
        wc = get_wc_basis('WET', 'JMS', s_out)[0]
        with self.assertRaises(ValueError):
            run_and_match(
                'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, s_out,
                wcs_out=[wc, wc],
            )


@unittest.skipUnless(_has_companions(), _REQUIRES_COMPANIONS)
class TestRunAndMatchModeB(unittest.TestCase):
    """sector_out=None; wcs_in and wcs_out mandatory, may span sectors."""

    def test_multi_sector_matrix(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import (
            _run_and_match_sector, matching_sectors, get_wc_basis,
        )
        sA, sB, s_unrelated = _pick_smeft_to_wet_sectors()
        wc_basis_out_sA = get_wc_basis('WET', 'JMS', sA)
        wc_basis_out_sB = get_wc_basis('WET', 'JMS', sB)
        wc_basis_in_sA = get_wc_basis('SMEFT', 'Warsaw', matching_sectors[sA])
        wc_basis_in_sB = get_wc_basis('SMEFT', 'Warsaw', matching_sectors[sB])
        wc_basis_unrelated = get_wc_basis('SMEFT', 'Warsaw', s_unrelated)

        wcs_out = [wc_basis_out_sA[0], wc_basis_out_sA[1], wc_basis_out_sB[0]]
        wcs_in = [
            wc_basis_in_sA[0], wc_basis_in_sA[1],
            wc_basis_in_sB[0], wc_basis_unrelated[0],
        ]

        M = run_and_match(
            'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT,
            sector_out=None, wcs_in=wcs_in, wcs_out=wcs_out,
        )
        self.assertEqual(M.shape, (3, 4))

        block_A = _run_and_match_sector(
            'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, sA,
        )
        block_B = _run_and_match_sector(
            'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT, sB,
        )

        # rows 0, 1 (sector sA) × cols 0, 1 (sector sA's input)
        for i in (0, 1):
            for j in (0, 1):
                expected = block_A[
                    wc_basis_out_sA.index(wcs_out[i]),
                    wc_basis_in_sA.index(wcs_in[j]),
                ]
                self.assertEqual(M[i, j], expected)

        # row 2 (sector sB) × col 2 (sector sB's input)
        expected = block_B[
            wc_basis_out_sB.index(wcs_out[2]),
            wc_basis_in_sB.index(wcs_in[2]),
        ]
        self.assertEqual(M[2, 2], expected)

        # cross-sector entries: zero
        # rows 0, 1 × cols 2, 3 (sB and unrelated cols)
        for i in (0, 1):
            for j in (2, 3):
                self.assertEqual(M[i, j], 0.0)
        # row 2 × cols 0, 1, 3 (sA and unrelated cols)
        for j in (0, 1, 3):
            self.assertEqual(M[2, j], 0.0)

    def test_error_only_wcs_out_when_sector_none(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import matching_sectors, get_wc_basis
        s = next(iter(matching_sectors))
        wc = get_wc_basis('WET', 'JMS', s)[0]
        with self.assertRaises(ValueError):
            run_and_match(
                'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT,
                sector_out=None, wcs_out=[wc],
            )

    def test_error_only_wcs_in_when_sector_none(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import matching_sectors, get_wc_basis
        s = next(iter(matching_sectors))
        wc = get_wc_basis('SMEFT', 'Warsaw', matching_sectors[s])[0]
        with self.assertRaises(ValueError):
            run_and_match(
                'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT,
                sector_out=None, wcs_in=[wc],
            )

    def test_error_all_none(self):
        from rgevolve.tools import run_and_match
        with self.assertRaises(ValueError):
            run_and_match(
                'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT,
                sector_out=None, wcs_in=None, wcs_out=None,
            )

    def test_error_unknown_wc(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import matching_sectors, get_wc_basis
        s = next(iter(matching_sectors))
        good_out = get_wc_basis('WET', 'JMS', s)[0]
        good_in = get_wc_basis('SMEFT', 'Warsaw', matching_sectors[s])[0]
        with self.assertRaises(ValueError):
            run_and_match(
                'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT,
                sector_out=None,
                wcs_in=[good_in, ('NONEXISTENT', 'R')],
                wcs_out=[good_out],
            )

    def test_error_duplicate_wc(self):
        from rgevolve.tools import run_and_match
        from rgevolve.tools.functions import matching_sectors, get_wc_basis
        s = next(iter(matching_sectors))
        good_out = get_wc_basis('WET', 'JMS', s)[0]
        good_in = get_wc_basis('SMEFT', 'Warsaw', matching_sectors[s])[0]
        with self.assertRaises(ValueError):
            run_and_match(
                'SMEFT', 'WET', 'Warsaw', 'JMS', SCALE_IN, SCALE_OUT,
                sector_out=None,
                wcs_in=[good_in, good_in],
                wcs_out=[good_out],
            )


def _has_all_companions():
    """True if the companion package of every basis in bases_available
    is installed."""
    try:
        from rgevolve.tools import bases_available, bases_installed
    except ImportError:
        return False
    return all(
        set(bases) <= set(bases_installed.get(eft, []))
        for eft, bases in bases_available.items()
    )


def _basis_changes():
    """(eft, basis_in, basis_out) for every ordered pair of distinct bases
    of an EFT in bases_available."""
    from rgevolve.tools import bases_available
    return [
        (eft, basis_in, basis_out)
        for eft, bases in bases_available.items()
        for basis_in in bases
        for basis_out in bases
        if basis_in != basis_out
    ]


# One sector per EFT whose translators are neither the identity nor unchanged
# by reversing both axes, so that a result in the wrong order cannot pass. The
# SMEFT sectors without quarks translate by the identity, hence 'mue'.
_REF_SCALE_SECTORS = {
    'SMEFT': 'mue', 'WET': 'sbemu', 'WET-4': 'sdemu', 'WET-3': 'sdemu',
}


@unittest.skipUnless(
    _has_all_companions(),
    "Requires the companion packages of every basis in bases_available",
)
class TestReferenceScaleBasisChange(unittest.TestCase):
    """Issue #2: scale_in == scale_out == reference_scale[eft] with
    basis_in != basis_out. run_and_match got an h5py Dataset from
    get_translator there, which h5py cannot index with np.ix_ (Mode B)
    or with an index list out of order (Mode A)."""

    def test_get_translator_returns_array(self):
        from rgevolve.tools.functions import get_translator
        for eft, basis_in, basis_out in _basis_changes():
            with self.subTest(eft=eft, basis_in=basis_in, basis_out=basis_out):
                T = get_translator(
                    eft, basis_in, basis_out, _REF_SCALE_SECTORS[eft],
                )
                self.assertIsInstance(T, np.ndarray)

    def test_per_sector_mode(self):
        from rgevolve.tools import run_and_match, get_wc_basis, reference_scale
        for eft, basis_in, basis_out in _basis_changes():
            with self.subTest(eft=eft, basis_in=basis_in, basis_out=basis_out):
                mu = reference_scale[eft]
                sector = _REF_SCALE_SECTORS[eft]
                wcs_in = get_wc_basis(eft, basis_in, sector)
                wcs_out = get_wc_basis(eft, basis_out, sector)
                args = (eft, eft, basis_in, basis_out, mu, mu, sector)
                in_order = run_and_match(*args, wcs_in=wcs_in, wcs_out=wcs_out)
                self.assertEqual(in_order.shape, (len(wcs_out), len(wcs_in)))
                M = run_and_match(*args, wcs_out=wcs_out[::-1])
                self.assertTrue(np.array_equal(M, in_order[::-1, :]))
                M = run_and_match(*args, wcs_in=wcs_in[::-1])
                self.assertTrue(np.array_equal(M, in_order[:, ::-1]))
                M = run_and_match(*args)
                self.assertIsInstance(M, np.ndarray)
                self.assertTrue(np.array_equal(M, in_order))

    def test_full_eft_mode(self):
        from rgevolve.tools import run_and_match, get_wc_basis, reference_scale
        for eft, basis_in, basis_out in _basis_changes():
            with self.subTest(eft=eft, basis_in=basis_in, basis_out=basis_out):
                mu = reference_scale[eft]
                sector = _REF_SCALE_SECTORS[eft]
                wcs_in = get_wc_basis(eft, basis_in, sector)
                wcs_out = get_wc_basis(eft, basis_out, sector)
                args = (eft, eft, basis_in, basis_out, mu, mu)
                in_order = run_and_match(
                    *args, sector, wcs_in=wcs_in, wcs_out=wcs_out,
                )
                M = run_and_match(
                    *args, sector_out=None, wcs_in=wcs_in, wcs_out=wcs_out,
                )
                self.assertTrue(np.array_equal(M, in_order))
                M = run_and_match(
                    *args, sector_out=None,
                    wcs_in=wcs_in[::-1], wcs_out=wcs_out[::-1],
                )
                self.assertTrue(np.array_equal(M, in_order[::-1, ::-1]))


class TestGetTranslatorBranches(unittest.TestCase):
    """Issue #2: get_translator returns an array in each of its three
    branches. The translation data are in memory, so this needs no
    companion package. No installed basis pair reaches the third branch,
    which translates via the matching basis."""

    def test_every_branch_returns_array(self):
        import io
        from unittest import mock
        import h5py
        from rgevolve.tools import functions
        jms = functions.matching_basis['WET']
        rng = np.random.default_rng(0)
        x_to_jms = rng.random((4, 3))
        jms_to_y = rng.random((2, 4))
        data = {}
        for basis, key, matrix in (
            ('X', str(('X', jms)), x_to_jms),
            ('Y', str((jms, 'Y')), jms_to_y),
        ):
            f = h5py.File(io.BytesIO(), 'w')
            self.addCleanup(f.close)
            f.create_dataset(f'{key}/s', data=matrix)
            data[basis] = f
        get_translator = functions.get_translator.__wrapped__  # bypass the cache
        with mock.patch.object(
            functions, 'translation_data', lambda eft, basis: data[basis],
        ):
            for basis_in, basis_out, expected in (
                (jms, 'Y', jms_to_y),
                ('X', jms, x_to_jms),
                ('X', 'Y', jms_to_y @ x_to_jms),
            ):
                with self.subTest(basis_in=basis_in, basis_out=basis_out):
                    T = get_translator('WET', basis_in, basis_out, 's')
                    self.assertIsInstance(T, np.ndarray)
                    self.assertTrue(np.allclose(T, expected))


class TestInterpolateFromRef(unittest.TestCase):
    """peterstangl/rgevolve#1 on a synthetic grid held in memory, so this
    needs no companion package. Off the grid, interpolate_from_ref weights
    the tabulated matrices, or with inverse=True their tabulated inverses,
    with t = log(mu/mu_lo) / log(mu_hi/mu_lo). A scale on the grid up to
    float rounding gets the tabulated matrix."""

    EFT = 'WET-3'  # its reference scale, 1.3 GeV, is on the grid below

    def setUp(self):
        from unittest import mock
        from rgevolve.tools import functions
        self.functions = functions
        self.assertEqual(functions.reference_scale[self.EFT], 1.3)
        self.scales = np.array([1.0, 1.3, 2.0, 5.0])
        rng = np.random.default_rng(0)
        b, c = 0.2 * rng.standard_normal((2, 3, 3))
        # U(1.3 GeV -> mu), the identity at 1.3 GeV
        self.regular = np.array([
            np.eye(3) + b * x + c * x**2 for x in np.log(self.scales / 1.3)
        ])
        self.inverse = np.linalg.inv(self.regular)
        tabulated = {False: self.regular, True: self.inverse}
        for name, new in (
            ('get_scales', lambda eft, basis: self.scales),
            ('get_evolution_matrix',
             lambda eft, basis, sector, scale_id, inverse=False:
                 tabulated[inverse][scale_id]),
            ('interpolate_from_ref', functions.interpolate_from_ref.__wrapped__),  # no cache
        ):
            patcher = mock.patch.object(functions, name, new)
            patcher.start()
            self.addCleanup(patcher.stop)

    def interpolate(self, scale, inverse=False):
        return self.functions.interpolate_from_ref(
            self.EFT, 'B', scale, 's', inverse=inverse,
        )

    def test_weight_in_both_branches(self):
        for mu in (1.1, 1.6, 3.0):
            with self.subTest(mu=mu):
                hi = np.searchsorted(self.scales, mu)
                lo = hi - 1
                t = np.log(mu / self.scales[lo]) / np.log(self.scales[hi] / self.scales[lo])
                for inverse, tabulated in ((False, self.regular), (True, self.inverse)):
                    expected = tabulated[lo] + t * (tabulated[hi] - tabulated[lo])
                    self.assertTrue(np.allclose(
                        self.interpolate(mu, inverse=inverse), expected,
                        rtol=1e-12, atol=1e-14,
                    ))

    def test_grid_point_up_to_rounding(self):
        for i, s in enumerate(self.scales):
            for scale in (np.nextafter(s, 0), s, np.nextafter(s, np.inf)):
                for inverse, tabulated in ((False, self.regular), (True, self.inverse)):
                    with self.subTest(scale=float(scale), inverse=inverse):
                        M = self.interpolate(float(scale), inverse=inverse)
                        self.assertTrue(np.array_equal(M, tabulated[i]))
        with self.assertRaises(ValueError):
            self.interpolate(float(self.scales[0] * (1 - 1e-6)))


def _has_wet3_flavio():
    """True if the companion package rgevolve.wet_3.flavio is installed."""
    try:
        from rgevolve.tools import bases_installed
    except ImportError:
        return False
    return 'flavio' in bases_installed.get('WET-3', [])


# CVLL_sddd in the WET-3, flavio basis, run from mu to 1.3 GeV by wilson with
# qcd_order=1, qed_order=1: the table in peterstangl/rgevolve#1. wilson 2.5.2
# reproduces it to the five digits given there.
_WILSON_CVLL_SDDD_TO_1P3 = {
    1.31: 0.99866, 1.40: 0.98746, 1.53: 0.97362,
    1.70: 0.95859, 1.90: 0.94412, 2.00: 0.93787,
}


@unittest.skipUnless(
    _has_wet3_flavio(), "Requires the companion package rgevolve.wet_3.flavio",
)
class TestOffGridRun(unittest.TestCase):
    """peterstangl/rgevolve#1 on the data of the sector 'sd' of the WET-3,
    flavio basis, whose grid points next to the reference scale 1.3 GeV are
    1.15 and 2.0 GeV. Off the grid, the run to the reference scale weighted
    the tabulated inverses with 1/t: from 1.31 GeV, CVLL_sddd came back as
    -2.49."""

    OFF_GRID = (1.31, 1.40, 1.53, 1.70)

    def setUp(self):
        from rgevolve.tools import reference_scale
        self.assertEqual(reference_scale['WET-3'], 1.3)

    def run_wet3(self, scale_in, scale_out, wcs=None):
        from rgevolve.tools import run_and_match
        return run_and_match(
            'WET-3', 'WET-3', 'flavio', 'flavio', scale_in, scale_out,
            sector_out='sd', wcs_in=wcs, wcs_out=wcs,
        )

    def test_round_trip_cvll_sddd(self):
        # the issue's reproduction. A single element leaves out the mixing
        # with the rest of the sector, so the product is 1 only approximately
        wcs = [('CVLL_sddd', 'R')]
        for mu in self.OFF_GRID:
            with self.subTest(mu=mu):
                down = self.run_wet3(mu, 1.3, wcs)[0, 0]
                up = self.run_wet3(1.3, mu, wcs)[0, 0]
                self.assertAlmostEqual(down * up, 1, delta=5e-3)

    def test_round_trip_sector(self):
        # off the grid, the interpolated inverse is not exactly the inverse of
        # the interpolated matrix, so a round trip is the identity only up to
        # about the interpolation error: at most 2.3e-2 on the diagonal here.
        # Off the diagonal it is larger where the entries are large, e.g. 0.6
        # for C7_sd <- CTRR_sdss
        pairs = [(mu, 1.3) for mu in self.OFF_GRID]
        pairs += [(1.40, 1.70), (1.10, 1.70), (0.90, 1.20)]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                product = self.run_wet3(a, b) @ self.run_wet3(b, a)
                self.assertLess(np.abs(np.diag(product) - 1).max(), 5e-2)

    def test_agrees_with_wilson(self):
        wcs = [('CVLL_sddd', 'R')]
        for mu, expected in _WILSON_CVLL_SDDD_TO_1P3.items():
            with self.subTest(mu=mu):
                value = self.run_wet3(mu, 1.3, wcs)[0, 0]
                self.assertLess(abs(value / expected - 1), 5e-3)
