"""
Tests for the experiment configuration.

These verify that the config dictionaries have the expected structure and
consistent values - e.g. that experimental phase run ranges don't overlap,
that all mouse IDs appear in the expected groups, and that speed definitions
match the conditions used in exp_cats.
"""
import numpy as np
from helpers.config import expstuff, exp_cats, micestuff, structural_stuff


def test_run_ranges_non_overlapping():
    """
    Within each experiment type (Repeats/Extended), the run index ranges for
    each phase should not overlap. Overlapping ranges would mean the same
    trial gets assigned to two different phases.
    """
    for exp_type, phases in expstuff['condition_exp_runs']['APAChar'].items():
        # Only check the basic non-overlapping phases (skip sub-phases like APA1/APA2)
        basic_phases = expstuff['condition_exp_runs_basic']['APAChar'][exp_type]
        all_runs = []
        for phase_name, runs in basic_phases.items():
            run_set = set(runs)
            for prev_runs in all_runs:
                overlap = run_set & prev_runs
                assert len(overlap) == 0, (
                    f"{exp_type}: phase '{phase_name}' overlaps with another phase at runs {overlap}"
                )
            all_runs.append(run_set)


def test_run_ranges_contiguous():
    """
    The basic phase ranges should cover a contiguous block of run indices
    with no gaps (Baseline -> APA -> Washout).
    """
    for exp_type, phases in expstuff['condition_exp_runs_basic']['APAChar'].items():
        all_runs = np.concatenate(list(phases.values()))
        all_runs_sorted = np.sort(all_runs)
        expected = np.arange(all_runs_sorted[0], all_runs_sorted[-1] + 1)
        assert np.array_equal(all_runs_sorted, expected), (
            f"{exp_type}: run ranges have gaps"
        )


def test_mouse_ids_consistent():
    """All mice in group A and B should be unique with no overlap."""
    group_a = set(micestuff['mice_IDs']['A'])
    group_b = set(micestuff['mice_IDs']['B'])
    assert len(group_a & group_b) == 0, "Mouse IDs overlap between groups A and B"
    assert len(group_a) == len(micestuff['mice_IDs']['A']), "Duplicate IDs in group A"
    assert len(group_b) == len(micestuff['mice_IDs']['B']), "Duplicate IDs in group B"


def test_speed_definitions_cover_conditions():
    """Every speed referenced in exp_cats should be defined in expstuff['speeds']."""
    defined_speeds = set(expstuff['speeds'].keys())
    for condition_name in exp_cats.keys():
        if 'VMT' in condition_name or 'Perception' in condition_name:
            continue
        # Condition names like 'APAChar_LowHigh' contain two speed names
        parts = condition_name.split('_')
        if len(parts) >= 2:
            speed_str = parts[1]  # e.g. 'LowHigh'
            for speed_name in defined_speeds:
                # Each component (e.g. 'Low', 'High') should be a defined speed
                if speed_name in speed_str:
                    speed_str = speed_str.replace(speed_name, '', 1)
            assert speed_str == '', (
                f"Condition '{condition_name}' references undefined speed component '{speed_str}'"
            )


def test_belt_transition_consistent():
    """The transition point in expstuff should match the belt WCS coordinates."""
    assert expstuff['setup']['transition_mm'] == 470
    assert structural_stuff['belt_width'] == 53.5
