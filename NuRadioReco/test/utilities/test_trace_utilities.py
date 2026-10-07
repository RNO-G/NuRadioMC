"""
Tests of the per-trace variables in `NuRadioReco.utilities.trace_utilities` that the RNO-G
feature extraction uses, on synthetic traces.
"""
import numpy as np
import scipy.signal
import scipy.stats

from NuRadioReco.utilities import trace_utilities, units

SAMPLING_RATE = 3.2 * units.GHz


def _traces(n_traces=40, seed=1):
    """
    Noise traces of three lengths; every second one carries a short pulse, some are rounded to whole numbers.

    Rounding gives many samples of equal envelope value at equal distance from the maximum.
    """
    rng = np.random.default_rng(seed)
    traces = []
    for n_samples in (2048, 2047, 512):
        for i in range(n_traces):
            trace = rng.normal(0, 1, n_samples)
            if i % 2:
                start = rng.integers(50, n_samples - 50)
                trace[start:start + 20] += rng.normal(0, 8, 20)
            if i % 5 == 0:
                trace = np.round(trace)
            traces.append(trace)
    return traces


def _reference_impulsivity(trace):
    """The impulsivity as `get_impulsivity` computed it before it took an envelope and returned diagnostics."""
    envelope = np.abs(scipy.signal.hilbert(trace))
    closeness = list(np.abs(np.arange(len(envelope)) - np.argmax(envelope)))
    sorted_envelope = np.array([x for _, x in sorted(zip(closeness, envelope))])
    cdf = np.cumsum(sorted_envelope**2)
    cdf = cdf / cdf[-1]
    return max((np.mean(np.asarray([cdf])) * 2.0) - 1.0, 0.0)


def test_impulsivity_default_is_unchanged():
    for trace in _traces():
        expected = _reference_impulsivity(trace)
        assert trace_utilities.get_impulsivity(trace) == expected
        envelope = trace_utilities.get_hilbert_envelope(trace)
        assert trace_utilities.get_impulsivity(trace, envelope=envelope) == expected
        assert trace_utilities.get_impulsivity(trace, return_diagnostics=True)["impulsivity"] == expected


def test_impulsivity_diagnostics_match_scipy():
    for trace in _traces(n_traces=6):
        diagnostics = trace_utilities.get_impulsivity(trace, return_diagnostics=True)
        envelope = trace_utilities.get_hilbert_envelope(trace)
        closeness = np.abs(np.arange(len(envelope)) - np.argmax(envelope))
        order = np.lexsort((envelope, closeness))
        cdf = np.cumsum(envelope[order]**2)
        cdf /= cdf[-1]
        x = closeness[order].astype(float)
        fit = scipy.stats.linregress(x, cdf)
        line = np.clip(fit.slope * x + fit.intercept, 0.0, 1.0)
        np.testing.assert_allclose(diagnostics["impulsivity_slope"], fit.slope, rtol=1e-9)
        np.testing.assert_allclose(diagnostics["impulsivity_intercept"], fit.intercept, rtol=1e-9)
        np.testing.assert_allclose(diagnostics["impulsivity_r_squared"], fit.rvalue**2, rtol=1e-9)
        # values that agree to rounding can fall on either side of each other: one step of 1/n
        assert abs(diagnostics["impulsivity_ks_statistic"]
                   - scipy.stats.ks_2samp(cdf, line).statistic) <= 1.0 / len(trace) + 1e-12


def test_impulsivity_separates_pulse_from_noise():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, 2048)
    pulse = 0.05 * noise
    pulse[1000:1010] += 10 * np.hanning(10)
    assert trace_utilities.get_impulsivity(pulse) > 0.9
    assert trace_utilities.get_impulsivity(noise) < 0.2
    flat = trace_utilities.get_impulsivity(noise, return_diagnostics=True)
    peaked = trace_utilities.get_impulsivity(pulse, return_diagnostics=True)
    assert flat["impulsivity_r_squared"] > 0.95 > peaked["impulsivity_r_squared"]
    assert flat["impulsivity_ks_statistic"] < peaked["impulsivity_ks_statistic"]


def test_spectral_slope_matches_linear_regression():
    for trace in _traces(n_traces=6):
        fmin, fmax = 0.08 * units.GHz, 0.6 * units.GHz
        features = trace_utilities.get_spectral_features(trace, SAMPLING_RATE, fmin, fmax)
        freqs = np.fft.rfftfreq(len(trace), 1 / SAMPLING_RATE)
        power = np.abs(np.fft.rfft(trace))**2
        mask = (freqs >= fmin) & (freqs <= fmax)
        expected = scipy.stats.linregress(freqs[mask], np.log10(np.clip(power[mask], 1e-30, None))).slope
        np.testing.assert_allclose(features["spectral_slope"], expected, rtol=1e-9)


def test_impulse_template_correlations_match_direct_correlation():
    assert set(trace_utilities.get_impulse_template_correlations(np.ones(64), SAMPLING_RATE).values()) == {0.0}
    rng = np.random.default_rng(3)
    for n_samples in (256, 255):
        trace = rng.normal(0, 1, n_samples)
        center = n_samples // 2
        delta = np.zeros(n_samples)
        delta[center] = 1.0
        sigma = int(5 * units.ns * SAMPLING_RATE)
        gaussian = np.exp(-(np.arange(n_samples) - center)**2 / (2.0 * sigma**2))
        correlations = trace_utilities.get_impulse_template_correlations(trace, SAMPLING_RATE)
        assert list(correlations) == ["delta", "bipolar", "gaussian", "bipolar_wide", "sinc"]
        for name, template in (("delta", delta), ("gaussian", gaussian)):
            direct = scipy.signal.correlate((trace - trace.mean()) / trace.std(),
                                            (template - template.mean()) / template.std(), mode="same") / n_samples
            np.testing.assert_allclose(correlations[name], np.max(np.abs(direct)), rtol=1e-9)
        template_like = (gaussian - gaussian.mean()) * 3.0 + 1.0
        np.testing.assert_allclose(
            trace_utilities.get_impulse_template_correlations(template_like, SAMPLING_RATE)["gaussian"], 1.0, rtol=1e-9)


def test_band_features_of_a_tone():
    n_samples = 2048
    times = np.arange(n_samples) / SAMPLING_RATE
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 1, n_samples)
    tone = noise + 5 * np.sin(2 * np.pi * 0.25 * units.GHz * times)
    in_band = trace_utilities.get_band_features(tone, SAMPLING_RATE, 0.2 * units.GHz, 0.3 * units.GHz)
    assert set(in_band) == {"band_power", "band_snr", "band_power_ratio", "band_slope", "peak_frequency"}
    np.testing.assert_allclose(in_band["peak_frequency"], 0.25 * units.GHz, atol=SAMPLING_RATE / n_samples)
    assert in_band["band_power_ratio"] > 0.9
    out_of_band = trace_utilities.get_band_features(tone, SAMPLING_RATE, 0.4 * units.GHz, 0.5 * units.GHz)
    assert out_of_band["band_power_ratio"] < 0.05
    # the noise reference is taken in the same band, so a gain factor cancels
    scaled = trace_utilities.get_band_features(7.0 * tone, SAMPLING_RATE, 0.2 * units.GHz, 0.3 * units.GHz)
    np.testing.assert_allclose(scaled["band_snr"], in_band["band_snr"], rtol=1e-9)
    np.testing.assert_allclose(scaled["band_power"], 49.0 * in_band["band_power"], rtol=1e-9)
    assert np.isnan(trace_utilities.get_band_features(np.zeros(64), SAMPLING_RATE, 0.2, 0.3)["band_snr"])


def test_normalized_cross_correlation_finds_the_shift():
    rng = np.random.default_rng(5)
    trace = rng.normal(0, 1, 512)
    shifted = 3.0 * np.roll(trace, 7)
    corr, lag = trace_utilities.get_normalized_cross_correlation(trace, shifted)
    assert lag == -7 and corr > 0.95
    corr, lag = trace_utilities.get_normalized_cross_correlation(trace, trace, max_lag=5)
    assert lag == 0
    np.testing.assert_allclose(corr, 1.0, rtol=1e-12)
    assert abs(trace_utilities.get_normalized_cross_correlation(trace, shifted, max_lag=3)[1]) <= 3
    corr, lag = trace_utilities.get_normalized_cross_correlation(trace, np.zeros(512))
    assert np.isnan(corr) and lag == 0


if __name__ == "__main__":
    test_impulsivity_default_is_unchanged()
    test_impulsivity_diagnostics_match_scipy()
    test_impulsivity_separates_pulse_from_noise()
    test_spectral_slope_matches_linear_regression()
    test_impulse_template_correlations_match_direct_correlation()
    test_band_features_of_a_tone()
    test_normalized_cross_correlation_finds_the_shift()
    print("test_trace_utilities.py: all tests passed")
