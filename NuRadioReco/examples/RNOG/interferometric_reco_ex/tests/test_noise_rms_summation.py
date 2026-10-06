"""The `noise_rms_summation` option: the form of the split-trace noise RMS behind the channel SNRs.

The noise RMS is the mean of the two smallest standard deviations of four consecutive
segments of a trace. `sequential` (the default) is `trace_utilities.get_split_trace_noise_RMS`,
which for a trace length that is a multiple of four adds the samples of each segment one
after the other; `pairwise` keeps the segments as float64 rows, which numpy adds pairwise.
Both are pinned here to their stated formula bit for bit; they agree to about 1e-15
relative. The pair weights, the validation columns and the driver's helper check follow
the key, a search on pair series refuses a config whose value differs from the one the
series were made with, and the driver writes the value to the results file.
"""

import math

import numpy as np
import pytest

from conftest import STATION
from interferometric_reco_3d_advanced import objective_attrs
from reco_validation import compute_channel_snrs
from synthetic import VPOL_CHANNELS, cylindrical_to_enu, make_event, same_value
from NuRadioReco.modules.interferometricDirectionReconstruction3D import InterferometricReco3D
from NuRadioReco.modules.reco3d.pair_weights import _split_trace_noise_rms_pairwise
from NuRadioReco.utilities.trace_utilities import get_signal_to_noise_ratio, get_split_trace_noise_RMS

# trace lengths in use: 2,048 samples at 3.2 GHz, 6,400 after upsampling to 10 GHz, 5,120 at 8 GHz
DIVISIBLE_LENGTHS = (2048, 5120, 6400)
SOURCE = (80.0, 120.0, -40.0)


def _traces(n_samples, count, seed):
    """Gaussian noise traces with standard deviations between 0.5 and 30."""
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, rng.uniform(0.5, 30.0), n_samples) for _ in range(count)]


def _sequential_formula(trace):
    """Mean of the two smallest segment standard deviations, every sum taken sample by sample."""
    stds = []
    for segment in np.array_split(trace, 4):
        values = [float(v) for v in segment]
        mean = sum(values[1:], values[0]) / len(values)
        squares = [(v - mean) * (v - mean) for v in values]
        stds.append(math.sqrt(sum(squares[1:], squares[0]) / len(values)))
    stds.sort()
    return (stds[0] + stds[1]) / 2


def _pairwise_formula(trace):
    """Mean of the two smallest segment standard deviations, each numpy's standard deviation of a float64 segment."""
    stds = sorted(np.std(segment) for segment in np.array_split(trace, 4))
    return (stds[0] + stds[1]) / 2


def _snrs(volt_arrays, noise_rms):
    """Three-sample SNR of every trace with the given noise RMS function."""
    return [get_signal_to_noise_ratio(v, noise_rms(v), 3) for v in volt_arrays]


@pytest.mark.parametrize('n_samples', DIVISIBLE_LENGTHS)
def test_each_form_equals_its_formula_bit_for_bit(n_samples):
    """Lengths that are multiples of four: both forms equal their formula exactly and differ from each other by rounding."""
    n_differ, largest = 0, 0.0
    for trace in _traces(n_samples, 40, n_samples):
        sequential, pairwise = get_split_trace_noise_RMS(trace), _split_trace_noise_rms_pairwise(trace)
        assert sequential == _sequential_formula(trace)
        assert pairwise == _pairwise_formula(trace) and type(pairwise) is float
        n_differ += sequential != pairwise
        largest = max(largest, abs(sequential - pairwise) / pairwise)
    assert n_differ >= 10 and largest < 1e-14


def test_forms_are_equal_for_other_lengths():
    """A length that is not a multiple of four: both forms are numpy's standard deviation of float64 segments."""
    for trace in _traces(2047, 40, 3):
        assert get_split_trace_noise_RMS(trace) == _split_trace_noise_rms_pairwise(trace) == _pairwise_formula(trace)


def test_small_trace_by_hand():
    """Segments with standard deviations sqrt(1.25), 0, 2 and sqrt(0.1875): the mean of the two smallest."""
    trace = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 4.0, 4.0, 1.0, 1.0, 1.0, 2.0])
    assert _split_trace_noise_rms_pairwise(trace) == math.sqrt(0.1875) / 2
    assert get_split_trace_noise_RMS(trace) == math.sqrt(0.1875) / 2
    assert _split_trace_noise_rms_pairwise(np.arange(12.0), segments=3, lowest=1) == math.sqrt(1.25)


def test_channel_snrs_follow_the_key_and_the_default_is_unchanged():
    """Without the argument the SNRs are those of trace_utilities; `pairwise` uses the float64 rows."""
    volt_arrays = _traces(6400, 15, 7)
    default = _snrs(volt_arrays, get_split_trace_noise_RMS)
    pairwise = _snrs(volt_arrays, _split_trace_noise_rms_pairwise)
    assert InterferometricReco3D._channel_snrs(volt_arrays, 3) == default
    assert InterferometricReco3D._channel_snrs(volt_arrays, 3, 'sequential') == default
    assert InterferometricReco3D._channel_snrs(volt_arrays, 3, 'pairwise') == pairwise
    assert default != pairwise and np.allclose(default, pairwise, rtol=1e-13, atol=0.0)
    channels = list(range(15))
    assert compute_channel_snrs(volt_arrays, channels) == dict(zip(channels, default))
    assert compute_channel_snrs(volt_arrays, channels, 'pairwise') == dict(zip(channels, pairwise))
    weights, snr = InterferometricReco3D._compute_snr_pair_weights(volt_arrays, channels, 'pairwise')
    assert snr == dict(zip(channels, pairwise))
    assert weights == InterferometricReco3D._record_pair_weights(pairwise)


@pytest.mark.slow
def test_reconstruction_follows_the_key(reco, det, base_config, tables, pa):
    """The validation SNR columns come from the chosen form, and the position of a clear source does not depend on it."""
    results = {}
    for name, extra in (('absent', {}), ('sequential', {'noise_rms_summation': 'sequential'}),
                        ('pairwise', {'noise_rms_summation': 'pairwise'})):
        evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*SOURCE, pa), VPOL_CHANNELS, tables, snr=20.0, seed=71)
        results[name] = reco.run(evt, stn, det, dict(base_config, validation=True, **extra))
    volt_arrays = [stn.get_channel(ch).get_trace() for ch in VPOL_CHANNELS]
    for name, noise_rms in (('absent', get_split_trace_noise_RMS), ('pairwise', _split_trace_noise_rms_pairwise)):
        expected = dict(zip(VPOL_CHANNELS, _snrs(volt_arrays, noise_rms)))
        assert all(results[name][f'ch{ch}_snr'] == expected[ch] for ch in VPOL_CHANNELS), name
    for key in (k for k in results['absent'] if not k.endswith('_time')):
        assert same_value(results['sequential'][key], results['absent'][key]), key
    assert any(results['pairwise'][f'ch{ch}_snr'] != results['absent'][f'ch{ch}_snr'] for ch in VPOL_CHANNELS)
    for key in ('rho', 'phi', 'z'):
        assert abs(results['pairwise'][key] - results['absent'][key]) < 1e-3, key
    with pytest.raises(ValueError, match='noise_rms_summation'):
        reco._validate_config(dict(base_config, noise_rms_summation='record'))


@pytest.mark.slow
def test_pair_series_keep_the_form_they_were_made_with(reco, det, base_config, tables, pa):
    """A search on pair series equals `run` under the same form and refuses a config of the other form."""
    pairwise = dict(base_config, validation=True, noise_rms_summation='pairwise')
    evt, stn, _ = make_event(det, STATION, cylindrical_to_enu(*SOURCE, pa), VPOL_CHANNELS, tables, snr=20.0, seed=72)
    pairs = reco.compute_pairs(stn, pairwise)
    assert pairs.settings['noise_rms_summation'] == 'pairwise'
    from_pairs, run = reco.reconstruct_from_pairs(pairs, pairwise, station=stn), reco.run(evt, stn, det, pairwise)
    for key in (k for k in run if not k.endswith('_time')):
        assert same_value(from_pairs[key], run[key]), key
    with pytest.raises(ValueError, match='noise_rms_summation'):
        reco.reconstruct_from_pairs(pairs, dict(base_config, validation=True))
    with pytest.raises(ValueError, match='noise_rms_summation'):
        reco.reconstruct_from_pairs(reco.compute_pairs(stn, base_config), pairwise)


def test_driver_writes_the_value_to_the_results_attributes(base_config):
    """The results file names the form, `sequential` when the config has no key."""
    assert objective_attrs(base_config)['noise_rms_summation'] == 'sequential'
    assert objective_attrs(dict(base_config, noise_rms_summation='pairwise'))['noise_rms_summation'] == 'pairwise'
    assert objective_attrs(base_config)['objective_version'] == 'record'
