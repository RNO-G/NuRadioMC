# RNO-G trigger simulation with FLOWER trigger model

General-purpose RNO-G simulation with a FLOWER trigger model and two noise modes (thermal and measured forced-trigger).

## Table of contents

- [Overview](#overview)
- [Files](#files)
- [Noise modes](#noise-modes)
- [FLOWER trigger model](#flower-trigger-model)
- [ADC pedestal and asymmetric saturation](#adc-pedestal-and-asymmetric-saturation)
- [Usage](#usage)
- [CLI arguments](#cli-arguments)
- [Output](#output)
- [Known limitations](#known-limitations)
- [Framework changes on this branch](#framework-changes-on-this-branch)

## Overview

`simulate.py` wraps the NuRadioMC simulation framework with three features:

1. **Measured noise injection.** Real forced-trigger (FT) waveforms replace synthetic thermal noise via `--ft_noise_dir`. See [noise modes](#noise-modes).

2. **Asymmetric ADC saturation.** Models the off-center pedestal bias of the RADIANT ADC via `--pedestal_voltage`. See [ADC pedestal](#adc-pedestal-and-asymmetric-saturation) and [`pedestal_extraction/`](pedestal_extraction/).

3. **FLOWER trigger model.** `triggerBoardResponse` + `highLowThreshold`. First-pass approximation; see [known limitations](#known-limitations).

All three are optional. Without `--ft_noise_dir`, thermal noise is used. Without `--pedestal_voltage`, the ADC range is symmetric. The FLOWER trigger is always active.

## Files

| File | Description |
|------|-------------|
| `simulate.py` | Simulation script supporting thermal and FT noise modes |
| `RNO_config.yaml` | Default NuRadioMC config (`noise: False` for FT mode) |
| [`noise_analysis/`](noise_analysis/) | FT noise cleaning (clean mask) and trigger-path Vrms extraction |
| [`pedestal_extraction/`](pedestal_extraction/) | ADC pedestal extraction from `pedestal.root` files, produces per-channel clip thresholds |

## Noise modes

### Thermal (default)

Without `--ft_noise_dir`, the framework generates noise from a temperature model through the signal chain response. The default config uses 300 K, configurable via `trigger.noise_temperature` in the YAML (or `"detector"` for per-channel values from the detector description).

### Measured FT noise (`--ft_noise_dir`)

FT waveforms are recorded through the readout signal chain (RADIANT, 3.2 GHz). The trigger path uses a different signal chain after the 3 dB splitter (arXiv:2411.12922, Sec. 3.2). So FT noise must be injected differently for each path. This is the v9 production method (matching `08_RNO_G_trigger_simulation_testing/rollover_demonstration/trace_length_study/simulate_fixed_response_v9.py`), implemented in-script:

1. **Trigger path** (at 5 GHz internal sim rate): the internal trigger trace is much longer than one FT event (padded to ~2 us for linear convolution), so several FT events are each upsampled from 3.2 to 5 GHz and stitched into one continuous noise trace with a Hann overlap-add crossfade (`tile_noise_overlap_add`, `TILE_OVERLAP` samples of overlap). The stitched noise is multiplied by a transfer function (`trigger_response / readout_response`, from the detector description) to convert from readout to trigger domain, then added to the trigger channel copies (ch 0-3). This is the v9 headline fix: earlier versions evaluated the trigger on a signal-only copy.

2. **Readout path** (at 3.2 GHz): a separate FT event is added directly to the readout channels at native rate, after the readout-window cut and resample. No transform is needed since the noise was recorded through the readout chain.

The two paths draw independent FT realizations from the same streaming pool (the trigger tiles and the readout event are different draws). The config must set `noise: False` to prevent the framework from also adding thermal noise on top of the injected FT noise.

**Measured tiling artifact.** The Hann overlap-add attenuates the first tile's leading edge to ~0.42 of the full noise RMS (no preceding tile to fill it) and dips the RMS to ~0.8 (near the sqrt(0.5) prediction for a 0.5/0.5 crossfade of independent noise) at each 3000-sample seam. A 3-arm study on real station-23 FT data and measured production geometry (`deep_cr_search/results/ft_injection_study/`, 10000 trials/arm) measured this to cost a small but statistically significant few percent (~2-10%, z up to -9) of near-threshold noise-only trigger probability versus an ideal seamless fill. The same study confirms this tiling gives full coverage and tracks the ideal to within that few percent, whereas the earlier single-event injection (first 3200 samples only) leaves 49-73% of the trigger copy noiseless and under-triggers by 40-71%, with an effective threshold inflated to ~5.3-7.2 sigma. The tiling is kept verbatim here for provenance; a proposed equal-power sqrt-Hann crossfade fix (no taper on the outermost edges) is written up as an unimplemented follow-up in that study's README.

The readout window is cut with a zero-padded cutter (`zero_padded_readout_window_cutter`) that replaces the framework's cyclic roll: when the readout window extends past the internal trace edge, the overflow is filled with zeros rather than wrapped. In FT mode the zeros are then covered by the readout FT injection, which spans the full 2048-sample readout trace.

Point `--ft_noise_dir` at a directory of `station{id}_run*.root` ROOT files. To exclude non-thermal FT events, pass `--ft_clean_mask` with an NPZ mask file (`runNum`/`eventNum`/`is_clean`) from [`noise_analysis/ft_cleaning/`](noise_analysis/ft_cleaning/). A pool smaller than `--n_events` simply reuses realizations (the file list is cycled and reshuffled).

## FLOWER trigger model

`triggerBoardResponse` (VGA gain + 8-bit ADC) followed by `highLowThreshold`:

- Threshold: ~3.76 sigma at 1 Hz rate
- Coincidence: 2-fold across PA channels 0-3
- High-low window: 6 samples at FLOWER rate (~472 MSa/s)
- Coincidence window: 20 samples at FLOWER rate

In FT mode, the trigger-path Vrms is loaded from a YAML file (`--trigger_vrms`). In thermal mode, it is computed from the noise temperature and the trigger signal chain response. See [`noise_analysis/trigger_vrms/`](noise_analysis/trigger_vrms/) for extraction and limitations.

## ADC pedestal and asymmetric saturation

The RADIANT ADC digitizes a 0-2.5V range. The pedestal bias sits at ~1.5V, off-center from the 1.25V midpoint, making the effective clip range asymmetric in pedestal-subtracted coordinates: [-1500, +1000] mV for a 1.5V pedestal.

`--pedestal_voltage` accepts a single value for all channels. For per-channel precision, use `analogToDigitalConverter.set_pedestal_voltage(dict)` programmatically. See [`pedestal_extraction/`](pedestal_extraction/) for measured per-channel values.

## Usage

### FT noise mode

```bash
python simulate.py \
    --config /path/to/config.yaml \
    --station_id 23 \
    --energy 1e18 \
    --n_events 1000 \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    --trigger_vrms /path/to/trigger_vrms.yaml \
    --ft_clean_mask /path/to/clean_mask_station23.npz \
    --ft_seed 12345 \
    --pedestal_voltage 1.5 \
    --output_file output.hdf5 \
    --data_dir /path/to/output
```

### Thermal noise mode

```bash
python simulate.py \
    --station_id 23 \
    --energy 1e18 \
    --n_events 1000 \
    --output_file output.hdf5 \
    --data_dir /path/to/output
```

### Parallel production via SLURM

```bash
#SBATCH --array=0-99
python simulate.py --station_id 23 --energy 1e18 --n_events 100 \
    --index $SLURM_ARRAY_TASK_ID \
    --output_file "chunk_${SLURM_ARRAY_TASK_ID}.hdf5" \
    --data_dir /path/to/output ...
```

## CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `RNO_config.yaml` | NuRadioMC YAML config (can include a `fiducial_volume` section) |
| `--station_id` | required | Station ID |
| `--energy` | 1e18 | Neutrino energy in eV |
| `--n_events` | 1000 | Number of events to simulate |
| `--detector_file` | None (MongoDB) | Fallback detector file when MongoDB is unavailable |
| `--ft_noise_dir` | None | FT data directory (enables measured noise mode) |
| `--ft_seed` | None | Reproducibility seed for FT noise selection |
| `--ft_clean_mask` | None | NPZ clean mask to exclude non-thermal FT events |
| `--trigger_vrms` | None | YAML with per-channel trigger-path Vrms (required for FT mode) |
| `--pedestal_voltage` | 1.5 | ADC pedestal in V for asymmetric clipping |
| `--noise_temperatures` | None | JSON with per-channel noise temperatures (K), overrides DB values |
| `--fiducial_rmax` | from config | Max fiducial radius in m (overrides `fiducial_volume.rmax` in config) |
| `--min_zenith` | from config (0) | Min zenith in degrees (overrides `fiducial_volume.min_zenith` in config) |
| `--max_zenith` | from config (60) | Max zenith in degrees (overrides `fiducial_volume.max_zenith` in config) |
| `--nur_output` | False | Also write NUR files |
| `--index` | 0 | Chunk index for parallel runs |

## Output

- **HDF5**: standard NuRadioMC output with triggered event data
- **NUR** (optional): NuRadioReco event files
- **Ledger CSV**: one row per input event with `event_group_id`, `zenith_deg`, `azimuth_deg`, `energy_eV`, `flavor`, `status` (`triggered` / `trigger_failed` / `efield_cut`), `max_amp_ch{0-3}_mV`

## Known limitations

**FT noise mode:**

- **Trigger Vrms must be pre-extracted.** `--trigger_vrms` requires a YAML file. Extract it using `noise_analysis/trigger_vrms/extract_trigger_vrms.py` before running. See [`noise_analysis/trigger_vrms/`](noise_analysis/trigger_vrms/).

- **VGA gain mismatch.** The simulated VGA gain selection does not match the real FLOWER hardware. Under investigation.

**Pedestal handling (applies when using `--pedestal_voltage`):**

- **Single value for all channels.** `--pedestal_voltage` applies one value. Real per-channel pedestals vary. For per-channel values, call `analogToDigitalConverter.set_pedestal_voltage()` with a dict in your own script. See [`pedestal_extraction/`](pedestal_extraction/).

- **No pedestal in the detector database.** The RNO-G MongoDB doesn't store pedestal voltages yet, so they must be passed via `--pedestal_voltage`.

## FT noise injection lives in the script

The measured-FT-noise machinery is implemented in `simulate.py` itself (the v9 method), not in a framework module:

- `FTNoisePool`: streaming reader/cycler over `station{id}_run*.root` FORCE events, with clean-mask and corrupt-file handling.
- `upsample_trace` + `tile_noise_overlap_add`: build the trigger-copy noise (Hann overlap-add of upsampled FT tiles) spanning the full internal trace.
- `_get_readout_to_trigger_transfer`: readout->trigger domain conversion for the trigger copies.
- `zero_padded_readout_window_cutter`: monkey-patched over the framework cutter (FT mode only).
- `resampler_with_noise_and_clip`: monkey-patched over `channelResampler` to add the readout FT realization and apply the ADC clip.

## Framework changes on this branch

This branch (`ft_noise_trigger_sim`) also carries these NuRadioMC modifications:

1. **`analogToDigitalConverter`**: pedestal voltage support (`set_pedestal_voltage()`)
2. **`readRNOGDataMattak`**: `ValueError` catch for corrupt ROOT files
3. **`efieldToVoltageConverterPerEfield`**: pre/post pulse zero-padding for linear convolution
4. **`rnog_detector`**: response_chain dict-to-list format conversion for exported detector files
5. **`highLowThreshold`**: channel ID included in trace_start_time warning
6. **`noiseImporter`**: trigger copy injection and two-stage mode. Retained on the branch but no longer used by this example, which injects FT noise in-script (see above).
