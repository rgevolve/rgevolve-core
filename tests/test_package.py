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
