# Interferometric Direction Reconstruction

3D interferometric direction reconstruction for in-ice radio events. Searches the full cylindrical (rho, phi, z) source volume using per-channel-pair cross-correlations evaluated against pre-computed travel-time tables, with a hierarchical coarse scan + refine stages and an L-BFGS-B optimizer.

Pipeline module: `NuRadioReco/modules/interferometricDirectionReconstruction3D.py`.

Compute kernels (Numba + CuPy RawKernels): `NuRadioReco/utilities/reco3d_kernels.py`.

Driver: `interferometric_reco_3d_advanced.py` (preprocessing, pass 1 and the optional pass 2). Configurations: `configs/reco3d_cr.yaml` (cosmic rays, record) and `configs/reco3d_cr_candidate.yaml` (cosmic rays, recommended), `configs/reco3d_neutrino_gzk.yaml` and `configs/reco3d_pulser_sim.yaml` (with solution-ordered tables, recommended for neutrinos and pulsers), `configs/reco3d_pulser_sim_fast.yaml` (pulser, hw mode).

This README explains how to reproduce the reference results on the GZK neutrino simulation and the simulated pulser calibration dataset.

## Prerequisites

1. **NuRadioMC/NuRadioReco** installed (this repo, `reco3d_release` branch)
2. **Python 3.11+** with numpy, scipy, h5py, pyyaml, numba (optional but recommended for speed)
3. **Multiray travel time tables** for your station. Two table schemes are supported:
   - **Ray-type tables** (default): 44 NPZ files per station (11 Vpol channels x 4 tables: direct, refracted, reflected, combined). File pattern: `st{ID}_ch{CH}_rz_table_{ray_type}.npz`.
   - **Solution-ordered tables** (recommended, faster): 22 NPZ files per station (11 Vpol channels x 2 tables: solution_0 = fastest arrival, solution_1 = slowest). File pattern: `st{ID}_ch{CH}_rz_table_solution_{0,1}.npz`. Set `table_scheme: "solution_ordered"` in the config to use these. See [Solution-ordered tables](#solution-ordered-tables) below.

   Tables are not included in the repo. On the Chicago cluster, pre-generated tables (both ray-type and solution-ordered) for stations 11, 12, 13, 21, 22, 23, and 24 are at `/data/reconstruction/validation_sets/test_tables/multiray_tables/`. For other stations, generate them using the scripts in `tables/` (see below).
4. **Detector description.** Needed for both sim and real data reconstruction. The default is a live query to the RNO-G MongoDB at `radio.zeuthen.desy.de:27017`. If your cluster can't reach that host, set `detector_file` in the config to a local detector export (JSON.xz snapshot) as a fallback.

## Setup

1. Edit `time_delay_tables` in the config file to point to the parent directory of `station23/`:

```yaml
time_delay_tables: "/path/to/multiray_tables"
```

The code appends `station{ID}/` internally, so it will look for files at `<time_delay_tables>/station23/st23_ch{N}_rz_table_{ray_type}.npz`.

2. Verify the table files are present:

```bash
ls /path/to/multiray_tables/station23/st23_ch0_rz_table_direct.npz
```

3. The shipped configs read their paths from environment variables (`NURADIO_TABLE_DIR`, `NURADIO_DETECTOR_DIR`, and `NURADIO_DELAY_CORRECTIONS_DIR` for `configs/reco3d_cr_candidate.yaml`, the directory holding `delay_corrections_2022_v2.yaml`). The driver expands variables in the top-level string values and in the `preprocessor:` block. With the corrections turned on, it loads the delay-corrections file at start-up for ROOT input, so the variable must be set (or the path edited) for real data. Simulated `.nur` input is reconstructed without the corrections: the driver then does not read the file, logs a warning and leaves `delay_corrections_hash` empty, so the same config runs on simulations without the variable.

## Running reconstruction

### Single file (interactive)

```bash
cd NuRadioReco/examples/RNOG/interferometric_reco_ex/

# Neutrino, hw mode (no antenna dedispersion, ~2 s/event)
python interferometric_reco_3d_advanced.py \
    --config configs/reco3d_neutrino_gzk.yaml \
    --mode hw \
    -i /path/to/nu_e_ccnc_1e18_1e20eV_GZK-2_IceCube-nu-2022_000000.nur \
    -o results/test_neutrino_hw.h5

# Pulser, rxtx mode (Rx + Tx antenna dedispersion, ~10 s/event)
python interferometric_reco_3d_advanced.py \
    --config configs/reco3d_pulser_sim.yaml \
    --mode rxtx \
    -i /path/to/output_r50.0_zen90.0_az0.0.nur \
    -o results/test_pulser_rxtx.h5

# Pulser, hw mode (no antenna dedispersion, ~5 s/event)
python interferometric_reco_3d_advanced.py \
    --config configs/reco3d_pulser_sim.yaml \
    --mode hw \
    -i /path/to/output_r50.0_zen90.0_az0.0.nur \
    -o results/test_pulser_hw.h5
```

### Batch processing

The driver takes any number of input files (`-i a.nur b.nur ...`), so a dataset is split into chunks with one driver call per chunk on any scheduler; the resource estimates below give the chunk counts, walltime and memory of the published timings. A merge of the per-chunk results files must carry the file attributes and the `coherent_waveforms` group.

### Evaluating results

Use `evaluate_reco_results.py` to compute angular separations against simulation truth and compare to the validated results:

```bash
# Neutrino GZK (truth from paired HDF5 files)
python evaluate_reco_results.py \
    --reco-file /path/to/output/neutrino_hw/merged_reco_results.h5 \
    --dataset neutrino \
    --sim-dir /path/to/gzk_hdf5_files/

# Pulser sim (truth parsed from NUR filenames)
python evaluate_reco_results.py \
    --reco-file /path/to/output/pulser_rxtx/merged_reco_results.h5 \
    --dataset pulser
```

The script prints median angular separation, percentiles, and the fraction of events below 1 and 2 degrees. It also prints reference values from the validated results below for comparison. These reference values are specific to the shipped validation datasets and station 23. If you are running on a different simulation set or station, your numbers will differ; treat them as a ballpark sanity check, not an exact target.

Real calibration-pulser runs are scored with `evaluate_pulser_runs.py` against the pulser device positions of the detector description in the absolute frame (`reco_validation.pulser_truth`: the exports store device z relative to the station, the reconstruction uses absolute z, so the station z is added). The device that fired comes from the run's `aux/comment.txt` (`fiber0` is the helper-C pulser, `fiber1` the helper-B pulser) and is checked against the helper SNR pattern; it is never the device closest to a reconstruction. Signal events are selected by PA SNR and host-string SNR (`--min-pa-snr`, `--min-host-snr`; the SNR is the module's per-channel value in the validation columns). `pulser_pair_residuals.py` reports, per channel pair, the correlation at the delay the tables predict for the pulser and the lag of the largest correlation around it, which is how per-string timing offsets are measured. The shared arithmetic (PA reference, angular separation, truth join by `(source_file, run_number)`, bounded-blend Xmax point, signal selection) lives in `reco_scoring.py` and is covered by `tests/test_reco_scoring.py`.

## Mode reference

| Mode | What it does | When to use | Runtime |
|------|-------------|-------------|---------|
| `hw` | Pass 1 only: cable delay + HW phase removal + grid search | Neutrinos (unknown source), fast pulser baseline | ~2 s/event |
| `rx` | Pass 1 + Pass 2: Rx antenna dedispersion at estimated arrival angles, local re-search | Neutrinos when runtime is acceptable | ~15 s/event |
| `rxtx` | Pass 1 + Pass 2: Rx + Tx antenna dedispersion (requires known emitter position in filename) | Pulser simulations only | ~10 s/event |

All modes run pass 1: a hierarchical 3D grid search (coarse log-spaced rho scan, peak extraction, linear refine grid, L-BFGS-B optimization) using the multiray tables.

### Coarse z grid and the ice surface

The coarse z vector of the hierarchical search covers `coarse_limits[4]` to `coarse_limits[5]`. A volume that reaches above the surface needs air-ice tables (`time_delay_tables` pointing at a set with z rows above 0) and `allow_above_surface: true`.

| Config key | Default | Description |
|------------|---------|-------------|
| `coarse_n_z` | `0` | Number of coarse z points; `0` steps by `coarse_step_sizes[2]` |
| `z_spacing` | `linear` | `linear` or `log`; log spacing concentrates points at the surface, on both sides when the volume crosses it |
| `z_surface_offset` | `0.1` | Smallest absolute z of a log grid (m) |
| `allow_above_surface` | `false` | Permit `z_max > 0` (checked by the flat search; the hierarchical search needs air-ice tables either way) |
| `z_grid_below` | absent | In-ice block of the split z grid: `{n: 100, spacing: linear}` (`offset` defaults to `z_surface_offset`, used by log spacing only); given together with `z_grid_above` |
| `z_grid_above` | absent | Air block of the split z grid: `{n: 60, spacing: log, offset: 1.0, refine_spacing: linear}`; the block starts at `max(offset, z_min)` (m), `offset` being the first tabulated air row; `refine_spacing` (`linear` or `log`, default `linear`) places the air part of refine and polish windows |

With `z_grid_below` and `z_grid_above` the coarse z vector is the in-ice block, built exactly as an in-ice volume ending at z = 0 builds it (the record in-ice grid node for node), followed by the air block from `max(offset, z_min)` to `z_max`. The two keys replace `coarse_n_z`, `n_z` and `z_spacing`: a config that carries any of them together with the split keys is rejected at `begin()`, so a converted air config cannot keep an old `coarse_n_z: 160` silently. After the coarse map the two blocks are searched separately: each takes its own `coarse_n_peaks` peaks, refine levels, optimizer seeds and, in candidate mode, candidate polish, within the search limits clamped at z = 0 (z_max at most 0 for the in-ice block, z_min at least 0 for the air block), and the results are merged by correlation. Refine and polish windows of the in-ice block are the plain linear windows of an in-ice volume; those of the air block start at `max(offset, lower window edge)` and place `refine_spacing` points at the count the linear step gives. In candidate mode every chain's coarse map, the shared fused stack included, is computed once on the whole grid and sliced per block; each block's candidates are polished within that block's limits, and the polished lists are merged by their ranking value and deduplicated before the tie band, its noise ceiling, the filled saved peaks and the candidate pool, so `candidate_raw_chain_corr` is the raw chain's best optimizer output over both blocks, `candidate_n_basins` is summed over the blocks and the pre-polish maxima are taken over them. The tie band (0.0054) and its ceiling (0.035) were measured on the forced-trigger null with the in-ice grid and checked above the surface on the 160-point two-sided log grid, never with the split keys, so the forced-trigger null has to be re-run with the split configuration before the two are used together. The coarse delay and travel-time caches of the hierarchical search are keyed by the two blocks, and those of the flat search (`hierarchical: false`, which also accepts the split keys) by the z vector itself. With the two keys absent nothing changes.

On the same tables the in-ice block therefore repeats the in-ice search exactly, and the two-sided result differs from the in-ice result only when an air maximum has the higher correlation. On the station-23 in-ice emitter grid (370 events, measured on the split-grid branch before its integration) under the air-ice tables, the two-sided split configuration with the 60-point air block equals the in-ice configuration on 368 events (record chain and candidate search alike, within 1e-6); the other two are rho 30 m sources at z = -10 m with peak correlations below 0.07 that both configurations misplace and where the air maximum is higher by 0.0006 and 0.0007. The medians are those of the record configuration, 0.749 deg (record chain) and 0.477 deg (candidate search). Against the in-ice configuration on the record tables 4 (record) and 10 (candidate) further events differ, because the z = 0 row is dead on the record tables (the batch lookup excludes the top table edge) and tabulated on the air-ice tables, and the (-1, 0) m band interpolates toward different z = 0 rows; that is a table difference, not a grid one. Searching the blocks separately doubles the refine, optimizer and polish work after the coarse map.

On the above-surface emitter grid (393 events, same branch measurement) the record chain with `n: 60` has no height bin with a larger median absolute z residual than the two-sided 160-point log grid (+1, +5, +20, +50, +100, +200 m: 1.39, 1.76, 6.20, 18.55, 23.48, 86.2 m before, 1.08, 1.37, 5.39, 12.25, 21.94, 70.7 m after); 40, 86 and 120 points and log refine placement each regress at least one bin. In candidate mode the same block gives 0.82 to 1.00 m at +1 m and 24.7 to 25.7 m at +100 m, both inside the 68 percent paired bootstrap interval, and better or equal values elsewhere. By default only pass 1 sees the split grid: the pass 2 of the `rx` and `rxtx` driver modes re-searches an in-ice template volume without the split keys or `allow_above_surface`, unless `pass2_volume: pass1` is set (below).

In `rx` and `rxtx` modes, pass 2 re-reads the event and removes antenna phase dispersion before a local re-search around the pass 1 result. Antenna dispersion introduces frequency-dependent phase shifts that broaden the cross-correlation peak. Removing them sharpens the peak and improves localization.

- **Rx dedispersion** removes the receiving antenna's phase response at the arrival angles estimated from pass 1. Since the arrival direction is unknown beforehand, it requires the pass 1 result.
- **Tx dedispersion** (rxtx only) also removes the transmitting antenna's phase response at the launch angles computed from the known emitter position. The emitter position is parsed from the NUR filename (pattern `output_r{R}_zen{ZEN}_az{AZ}.nur`), so this mode only applies to pulser simulations where the source location is known.

Three driver keys (read by the driver in `reco_pass2.py`; each keeps the previous behaviour when absent) make pass 2 usable for sources above the surface and for mixed antenna types:

| Config key | Default | Description |
|------------|---------|-------------|
| `pass2_volume` | `template` | `template`: pass 2 re-searches the in-ice template volume (flat grid at `pass2_step_sizes`, no air rows) over `pass2_window` around the pass-1 answer, clamped to the pass-1 `limits` (rho never below 1 m) and at z = 0 (a pass-1 answer above the surface raises; use `pass1`). `pass1`: pass 2 re-searches the pass-1 configuration itself (its tables, `allow_above_surface`, `z_grid_below`/`z_grid_above` and search keys, candidate search included) with `limits` and `coarse_limits` set to `pass2_window` around the pass-1 answer, clamped to the pass-1 `limits` in rho (never below 1 m) and z; phi is not clamped |
| `rx_arrival_mode` | `direct` | `direct`: the direct-ray receive direction of the `air_ice` propagator from the pass-1 position placed around the station origin. `first_arrival`: the position is placed around the phased-array axis, as the reconstruction defines it, and each channel takes the receive direction of the solution with the smallest travel time (air leg plus refraction at the surface for a position above the ice); a surface LPDA takes the straight line from the antenna to a position above the surface |
| `cross_type_sign_mode` | `signed` | `abs`: every pair of two antenna types (VPol, HPol, LPDA, from the antenna model name) is scored by the absolute value of its raw correlation, in both passes (it sets `pair_signs`; not combinable with `polarization_groups`) |

After each pass-2 search the driver empties the travel-time and delay caches keyed by the search grid (`clear_grid_caches`); the window moves with every event, so these caches otherwise grow by one entry per event (the size of the cached stack is about 130 MB per event for the flat template at 1 m, 0.3 deg, 1 m steps with 11 channels). Results are unchanged by this.

## Input format and preprocessing

The driver script (`interferometric_reco_3d_advanced.py`) expects standard NuRadioMC simulation output: NUR files containing voltage traces (not electric fields). No manual preprocessing is needed. The driver applies all necessary waveform processing (cable delays, hardware response removal, upsampling, optional bandpass/CW filtering) internally before calling the reconstruction module. You do not need to apply voltage calibration, antenna deconvolution, bandpass filtering, or any other signal processing yourself.

Which preprocessing steps are applied is controlled by the config file (see table below). The shipped configs use validated defaults, so for a first pass you only need to update the paths.

If you are integrating the reconstruction module into your own processing chain rather than using the driver script, the module expects waveforms that have at minimum had cable delays applied. See the config options below for the full set of preprocessing the driver applies.

The two set-up steps of the driver are public functions of `NuRadioReco.modules.RNO_G.dataProviderSetup`, for any script that reads RNO-G data or simulations. `init_detector(config)` returns the detector description for `station_id`, read from `detector_file` when the config has one and from the RNO-G database otherwise, updated to `detector_date` (default 2022-10-01). `select_data_provider(input_file, det, reader_kwargs=None, preprocessor_config=None)` returns the provider of an input with `begin` called: `dataProviderNuRadio` for a `.nur` file, `dataProviderRNOG` for a run folder or ROOT file, with `reader_kwargs` merged over the mattak defaults `read_daq_status: False` and `backend: uproot`. `init_detector` also stays importable from the driver.

The driver reads two preprocessing keys at the top level of the config, `apply_upsampling` and `apply_dedispersion`. Every other step is configured in the `preprocessor:` block, which is passed to `channelPreprocessor` (inside `dataProviderRNOG` or `dataProviderNuRadio`); a `channelPreprocessor` key written at the top level, such as `apply_bandpass: true`, would have no effect, so the driver stops with an error naming it (`misplaced_preprocessor_keys` in `reco_config.py`). The defaults are those of `channelPreprocessor`:

| Config key | Default | Description |
|------------|---------|-------------|
| `apply_upsampling` | `true` | Resample to 10 GHz (`channelResampler`); top level, driver-owned |
| `apply_dedispersion` | `false` | Antenna phase dedispersion at broadside (`channelAntennaDedispersion`); top level, driver-owned |
| `preprocessor.apply_block_offset_removal` | `true` | Fit and subtract the LAB4D block offsets (`channelBlockOffsets`) |
| `preprocessor.apply_cable_delay` | `true` | Subtract cable delays (`channelAddCableDelay`) |
| `preprocessor.apply_hw_phase_removal` | `false` | Remove hardware phase response (`hardwareResponseIncorporator`, phase-only) |
| `preprocessor.apply_cw_removal` | `false` | CW sinewave subtraction (`channelSinewaveSubtraction`) |
| `preprocessor.apply_notch` | `false` | Set the spectrum bins of every channel inside each band of `notch_bands` to zero (pairs of lower and upper edge in GHz, edges included; default `[[0.399, 0.407]]`, the radiosonde telemetry near 403 MHz), after the CW subtraction and before the bandpass. A fixed band rejection for a known narrow-band transmitter that the adaptive CW subtraction does not always catch; it costs a broadband pulse about 1% of its amplitude |
| `preprocessor.apply_bandpass` | `false` | Bandpass filter (`channelBandPassFilter`): `bandpass_band` (GHz, default `[0.1, 0.6]`), `bandpass_order` (10), `bandpass_filter_type` (`butter`) |
| `preprocessor.apply_delay_corrections` | `false` | Add per-channel corrections to the database cable delays from `delay_corrections_file` (`channelPreprocessor`, ROOT input only: `dataProviderNuRadio` drops the key because simulations share the tables' detector description, so real data stored as `.nur` gets no corrections and the results file records an empty `delay_corrections_hash`) |
| `preprocessor.delay_corrections_file` | none | YAML with `corrections: {station_id: {channel_id: delta_ns}}`, `uncertainty_ns` per channel and the provenance keys `derived_from`, `method`, `date`, `valid_from`, `valid_to`; `delta_ns` is added to the database delay (the step shifts the trace start time by minus delta); an event of a corrected station outside its validity window raises, the window being the file's `valid_from` to `valid_to`, narrowed to the station's own `valid_from` and `valid_to` when its `derived_from` entry carries them (stations without an entry in `corrections` are untouched at any date); the file's SHA-256 and path are written to the results file |

The CR configs (`reco3d_cr.yaml`, `reco3d_cr_candidate.yaml`) carry the record chain in the `preprocessor:` block: cable delays, hardware phase, 0.1 to 0.7 GHz Butterworth order 10 and CW subtraction. The neutrino and pulser configs carry cable delays and hardware-phase removal without bandpass or CW subtraction. Until the 2026-10-03 fix they wrote these keys at the top level, so earlier driver runs used block offsets and cable delays only (no hardware-phase removal); on regenerated GZK neutrinos that cost 0.50 deg (station 23) and 0.54 deg (station 13) on the median vertex direction and about 140 m on the median 3D vertex error.

Unknown keys under `preprocessor:` raise, so a misspelt step name cannot be dropped silently.

The reconstruction module handles additional correlation-level options internally: `hilbert_envelope_mode`, `apply_hann_window`, `correlation_normalization`, and `snr_pair_weighting`. These are also set in the config file.

`channels` under the nested `preprocessor:` block (default `null` = every channel in the event) restricts the per-channel steps of `channelPreprocessor` (block offsets, glitch detection, cable delay, hardware phase removal, CW subtraction, notch, bandpass) to the listed channel ids; the other channels stay in the event with their raw traces. When the key is absent the driver sets it to the union of `channels` and every `polarization_groups` entry (plus the helper and power-string channels when `plane_wave_fallback` is on). The reconstruction never reads the other channels, so its output is unchanged and the preprocessing cost of the unused channels (9 of 24 in the record configuration) is saved. Set `channels: null` explicitly to preprocess every channel, for example when the event is written out with `--save-nur` for another consumer.

For simulation data, bandpass, CW removal, and dedispersion are typically unnecessary (the permutation study confirmed bandpass is negligible for neutrinos). For real data with CW contamination, enable `apply_cw_removal: true` in the `preprocessor:` block.

## Output format

Each HDF5 output file contains a `results` group with these datasets:

| Dataset | Shape | Description |
|---------|-------|-------------|
| `rho` | (N,) | Reconstructed radial distance (m) |
| `phi` | (N,) | Reconstructed azimuth (deg) |
| `z` | (N,) | Reconstructed depth (m) |
| `max_corr` | (N,) | Peak correlation value |
| `run_number` | (N,) | Event group ID from NUR file |
| `event_number` | (N,) | Sub-event index |
| `source_file` | (N,) | Source NUR filename |
| `pass1_rho` | (N,) | Pass 1 rho (rx/rxtx mode only) |
| `pass1_phi` | (N,) | Pass 1 phi (rx/rxtx mode only) |
| `pass1_z` | (N,) | Pass 1 z (rx/rxtx mode only) |
| `pass1_corr` | (N,) | Pass 1 correlation (rx/rxtx mode only) |

For neutrino truth comparison, the paired HDF5 files contain `xx`, `yy`, `zz` vertex coordinates in the simulation frame. Convert to cylindrical relative to the phased array center for angular separation calculations.

The dataset names, dtypes and definitions above and in the multi-peak and validation sections are the output contract (`reco_output.py`, checked by `tests/test_output_contract.py`): consumers rely on them, so a change is a versioned change and new per-event quantities carry a `_v<N>` suffix. The file attributes record the provenance of a run:

| Attribute | Description |
|-----------|-------------|
| `mode`, `n_events`, `validation` | Driver mode, number of events, whether validation datasets were written |
| `reco_version` | Version of the reconstruction module (`RECO_VERSION`) |
| `detector_epoch` | Detector time at which the cable delays below were read (the first event's station time) |
| `detector_delay_hash` | First 16 hex characters of the SHA-256 of the per-channel cable delays at that epoch; a changed database delay changes it |
| `detector_delay_channels`, `detector_delay_ns` | The cable-delay vector itself |
| `delay_corrections_hash`, `delay_corrections_file` | SHA-256 and path of the applied corrections file, empty when none was applied |
| `ray_tracer_backend` | `cpp` or `python`: the analytic ray tracer of the rx/rxtx arrival and launch angles. NuRadioMC falls back to the python tracer without raising when the C++ extension does not import; the two agree only to numerical precision, so compare rx/rxtx files only within one backend (also logged once per run) |

At start-up and again at the first event's epoch the driver runs `check_detector_consistency`: every channel whose table records `antenna_z_abs` (the air-ice tables do; the in-ice tables of record carry no metadata) must sit within 0.05 m of the table's antenna depth or the run stops, horizontal positions are compared with the table's detector snapshot when that file is found next to `detector_file` or under `NURADIO_DETECTOR_DIR`, and a cable-delay change between `detector_date` and the event epoch is logged. `InterferometricReco3D.begin` runs the same check and keeps the report as `detector_report`.

### Multi-peak output

Set `n_peaks_save: 3` in the config to retain the top N peaks from the correlation map. Each peak gets its own fields: `peak_0_rho`, `peak_0_phi`, `peak_0_z`, `peak_0_corr`, `peak_0_map_snr` (and similarly for peaks 1, 2). The primary result (`rho`, `phi`, `z`, `max_corr`) always matches peak 0.

The columns of a results file are the union over its events: an event whose search filled fewer peak slots than another event of the file, or that lacks any other field, has NaN there. With `polarization_groups` the coarse peaks of each group are written as `coarse_peaks_<group>` of shape (N, n_peaks, 4) holding (rho, phi, z, corr), n_peaks being the largest number in the file and the rows of an event beyond its own number NaN.

Set `save_coarse_map: true` to get the coarse correlation map of the hierarchical search in the result dict of `run` and `reconstruct_from_pairs`: `coarse_map_v1` of shape (n_rho, n_phi, n_z) with its axes `coarse_map_rho_v1` (m), `coarse_map_phi_v1` (deg) and `coarse_map_z_v1` (m). It is the map the map SNR of the saved peaks is read on: the map of the search chain, in candidate mode that of the raw chain or, without one, of the first chain. With `polarization_groups` every group has its map under the group's suffix (`coarse_map_v1_hpol`). An event without a coarse peak has no map, and the driver's results file does not store the maps. Without the key the result is unchanged.

### Search geometry options

Three keys change how the refine grids are placed and how the tables are read at their top row. All default to the behaviour without them.

| Config key | Default | Description |
|------------|---------|-------------|
| `refine_window_mode` | `fixed` | `fixed` uses `refine_window` as given; `adaptive` widens the first refine level's half-window in rho and z to at least 0.6 times the wider of the two coarse cells adjacent to the peak bin (so the grid covers at least half of either neighbouring cell on both sides), keeps the phi half-window and the steps, so a coarse peak beyond about 117 m on the record grid (rho cells 24 to 43 m wide) has the whole cell inside its refine grid. The multi-level adaptive machinery (`n_refinements`, `refinement_factor`, `refinement_window_bins`) remains available and is unchanged |
| `subbin_coarse_seeds` | `false` | Centre the refine grids of the envelope chains on a sub-bin estimate of each coarse peak from a parabola through the peak bin and its two neighbours on each axis (log-linear in rho, linear in z, wrapped in phi when the grid covers the full circle); an axis keeps the bin centre at a grid edge, next to a non-finite neighbour, or when the three values do not form a maximum. The raw coarse map and the raw polish grids are never shifted (the raw lobes are narrower than a bin, so a parabola interpolates between lobes): the key changes nothing in a raw-only configuration, and in the default search it acts only when `hilbert_envelope_mode` is set. `coarse_peaks` still reports the bin centres |
| `tolerant_table_edge` | `false` | Read at `begin` (a change clears the travel-time and delay caches): every grid table lookup (the fused singleray coarse stack, refine and polish grids, the delay-matrix and multiray lookups and the fused multiray refine kernel) goes through one masked bilinear lookup, which with this key accepts a query on the last table row or column as the scalar optimizer kernel already does, so a grid point at z = 0 exactly uses the extrapolated top row instead of being invalid. The golden reference and the forced-trigger `surf_corr_z` distribution change with it |

### Candidate search

Set `candidate_search` to run the coarse, refine and optimizer stages once per search objective on the same coarse grid and rank every optimizer output by the raw (non-envelope) correlation after a local polish. The saved peaks are the ranked candidates with their raw correlations. With the key absent the reconstruction behaves exactly as before.

| Config key | Default | Description |
|------------|---------|-------------|
| `candidate_search` | absent | List of search chains: `raw`, `envelope:traces`, `envelope:correlation`, or plain `envelope` for the mode named by `candidate_envelope_mode`; `[envelope:traces, envelope:correlation, raw]` runs both envelope chains and lets the raw ranking choose; absent or empty disables the mode |
| `candidate_envelope_mode` | `traces` | Hilbert envelope used by the `envelope` chain: `traces` (envelope of each trace) or `correlation` (envelope of each correlation) |
| `candidate_polish_window` | `[3.0, 1.0, 3.0]` | Half-widths (m, deg, m) of the raw-correlation grid evaluated around each candidate, clamped to `limits` in rho and z; a list of triples runs one grid per level, each centred on the best point so far |
| `candidate_polish_steps` | `[0.5, 0.1, 0.5]` | Steps (m, deg, m) of that grid, one triple per level; the final maximum seeds the optimizer with the raw objective (and, with `polish_objective: two_arrival_consistent` and a `two_arrival_margin`, a second grid and optimizer pass with the two-arrival objective) |
| `candidate_include_refined` | `false` | Also polish the refined peaks of each chain that were not optimizer seeds; the candidate pool then holds the ranked candidates alone |
| `candidate_fill_saved_peaks` | `false` | Fill `peak_1` to `peak_{n_peaks_save-1}` from the candidate pool (the ranked candidates plus the refined peaks no chain optimized, graded with the raw correlation where they stand, in raw order after the primary) so `n_peaks_save` distinct positions are saved whenever the pool has them; `peak_0` stays the ranked primary. Also fills the default search from its own unseeded refined peaks |
| `candidate_diagnostics` | `false` | Add the miss-versus-misrank diagnostics (best raw correlation at any candidate position before polishing, overall and per chain, number of distinct basins before the optimizer, pre-polish raw correlation of each saved peak, the first eight pool entries) |
| `candidate_tie_band` | absent | Raw-correlation margin below which the raw chain's own answer is kept: when the ranked best candidate beats the raw chain's best optimizer output (before the polish) by less than this value, the primary result and saved peak 0 are the raw chain's answer, which equals the default search's for the same parameters with raw correlation (`hilbert_envelope_mode` null) and no `refinement_envelope_mode`; requires `raw` in `candidate_search`; absent keeps the ranked best always. 0.0054 is the 99 percent quantile of `candidate_gain` on the forced-trigger null |
| `candidate_tie_band_max_raw_corr` | absent | Noise ceiling for the tie band: the fallback applies only when `candidate_raw_chain_corr` is also below this value, so events whose raw chain already found a source above the noise level keep the ranked best; requires `candidate_tie_band`; absent applies the band to every event. 0.035 sits just above the forced-trigger maximum of `candidate_raw_chain_corr` (0.031) |
| `polish_objective` | `raw` | Objective family of the polish stage: `raw` (the raw correlation) or `two_arrival_consistent` (experimental: the consistent two-arrival correlation on the solution-ordered tables is evaluated at every polished position and, with `two_arrival_margin`, polishes and ranks the candidates, see below); the coarse search, refine levels and chains stay single-arrival; rejected with `multi_ray_types: true` or `objective_normalisation: valid` |
| `two_arrival_weight_mode` | `mask` | Weight of the solution_1 term of a pair: `mask` multiplies `two_arrival_second_weight` by the product of the two channels' critical-angle masks, `fixed` uses `two_arrival_second_weight` everywhere |
| `two_arrival_second_weight` | `1.0` | Weight of the solution_1 term (non-negative) |
| `two_arrival_margin` | absent | Absent: the candidates are polished and ranked by the raw correlation and the two-arrival value is saved as a diagnostic. A non-negative number: the candidates are also polished by the two-arrival objective and every position from either polish is ranked by its two-arrival value where that exceeds its raw correlation by more than the margin, otherwise by its raw correlation. The value must be measured on the forced-trigger null before any production use |
| `max_corr_source` | `raw` | With the two-arrival objective, which value `max_corr` and `peak_{i}_corr` carry: `raw` (the single-arrival raw correlation at the polished position) or `two_arrival` (the two-arrival value) |

`configs/reco3d_cr_candidate.yaml` is the recommended CR configuration: the record CR configuration `configs/reco3d_cr.yaml` plus `candidate_search: [envelope:traces, envelope:correlation, raw]`, `candidate_tie_band: 0.0054`, `candidate_tie_band_max_raw_corr: 0.035`, `candidate_fill_saved_peaks: true` (fills every saved peak and moves no primary) and, for real data, the 2022 delay corrections `delay_corrections_2022_v2.yaml` in its `preprocessor:` block. Its search keys and preprocessing chain (cable delays, hardware phase, 0.1 to 0.7 GHz Butterworth order 10, CW subtraction) are the ones the thresholds were measured with. Both thresholds come from the 2022 station-13 and station-23 forced-trigger runs alone, on the eleven VPol channels with the delay corrections off. Measured with the record search parameters on the station-23 VPol channels and paired against the record (95 percent cluster-bootstrap intervals), the median angular error changes by -0.271 deg [-0.513, -0.059] on the in-ice emitter grid, -0.376 deg [-0.756, -0.108] on the above-surface grid and -1.104 deg [-1.561, -0.149] on the CR simulation tier (median 4.688 deg for the record, 3.583 deg here), and the fraction beyond 3 deg by -0.176, -0.069 and -0.033; no CR bin is worse at 95 percent, and the lgE 16.0 and rho 150-200 m bins stay at the record's level. On the forced-trigger null 990 of 1000 station-13 events and 991 of 1000 station-23 events fall back to the record answer and every geometry fraction stays within 0.003 of the record's. A different noise environment or channel set needs both thresholds re-measured on its own forced triggers. Three parts of the configuration lie outside that measurement. The HPol group (the `_hpol` fields) runs the same candidate search with the same thresholds, which were never calibrated on it. The delay corrections were off; on the first 200 events of station-23 run3400 the v2 station-23 correction (channel 5, -2.83 ns) moves no `max_corr` percentile of the record chain or the candidate search by more than 0.0014. The split z grid keys (see Coarse z grid and the ice surface) take `candidate_raw_chain_corr` and the gain over both blocks and have not been tested with the thresholds. The v2 station-21 channel-5 constant worsens the station-21 run476 pulser pointing (median 0.83 deg with v1, 2.25 deg with v2), so leave the corrections off for station-21 data until that is resolved.

Output fields added in this mode: `n_candidates` (distinct candidates after deduplication), `candidate_n_pool` (size of the candidate pool: the ranked candidates plus, unless `candidate_include_refined` already made them candidates, the refined peaks no chain optimized, graded with the raw correlation where they stand and deduplicated against the ranked candidates; ordered primary first, then by raw value, so unpolished entries and polished candidates interleave), `candidate_origin_{i}` (chain of saved peak i: 0 raw, 1 traces envelope, 2 correlation envelope), `candidate_map_snr_chain` (chain whose coarse map gives `peak_{i}_map_snr` and the legacy quality metrics: the raw chain when searched, else the first chain), `candidate_search_time` and `candidate_polish_time`, `candidate_raw_chain_corr` (the raw chain's best optimizer output before the polish, NaN without a raw chain result) and `candidate_gain` (ranked best raw correlation minus `candidate_raw_chain_corr`; with the raw ranking it is non-negative up to float round-off of about 1e-8, so a zero band can send such an event to the fallback; with `two_arrival_margin` the ranked best is chosen by its ranking value while the gain uses its raw correlation, so the gain can be negative, -0.0044 on a synthetic in-ice event and -0.0027 on a noise event), plus `candidate_fallback` (1 when the tie band kept the raw chain's answer, else 0) when `candidate_tie_band` is set. With a fallback the saved peaks after peak 0 are the ranked candidates outside its deduplication box, so their correlations can exceed peak 0's. A band has no effect on an event whose raw chain gives no optimizer output (`candidate_gain` NaN). `coarse_peaks`, `n_coarse_peaks`, `n_refined_peaks` and the validation metrics come from that same chain; `coarse_time`, `refine_time` and `opt_time` are summed over the chains. `hilbert_envelope_mode` and `refinement_envelope_mode` are not used in this mode. With `polarization_groups` each group runs its own chains and the fields carry the group suffix.

Quality metrics whose meaning would change in this mode are versioned:

| Field | Mode | Definition |
|-------|------|------------|
| `peak_isolation_ratio` | both | Top coarse peak over the mean of the top five coarse peaks of the coarse map (validation only). In candidate mode the coarse map and peaks are the raw chain's when `raw` is searched, so the value is what the default search reports on the same event |
| `peak_{i}_map_snr` | both | Coarse-map value at the bin nearest saved peak i over the standard deviation of that map outside a 3-bin box; same map as above. In candidate mode the peaks are that chain's own optimizer outputs (the default search's saved peaks when `raw` is searched), so the value is what the default search reports on the same event; NaN for a saved peak beyond that chain's outputs |
| `peak_{i}_map_snr_v2` | candidate | The same map-SNR definition at saved peak i of the candidate result (the ranked, filled or fallback position the result reports) |
| `peak_isolation_ratio_v2` | candidate | Raw correlation of the pool primary (`candidate_pool_0_corr`, equal to `max_corr` unless a `post_optimizer_mode` moved the primary after the candidate stage) over the mean of the five highest raw correlations in the candidate pool; NaN with fewer than two pool entries |
| `map_snr_v2` | candidate | Raw correlation of the pool primary over the standard deviation of the raw coarse map; NaN unless `raw` is searched |
| `n_filled_peaks` | both, with `candidate_fill_saved_peaks` | Number of saved peaks that are unpolished pool entries (graded where they stand), wherever they rank among the saved peaks; `peak_0` is never one. With `candidate_diagnostics` such a peak has `candidate_prepolish_corr_{i}` equal to `peak_{i}_corr` |
| `candidate_prepolish_max_corr`, `candidate_prepolish_max_corr_{raw,traces,correlation}`, `candidate_n_basins`, `candidate_prepolish_corr_{i}`, `candidate_pool_{i}_{rho,phi,z,corr,origin}` | candidate, with `candidate_diagnostics` | Best raw correlation at any candidate position before polishing, overall and per chain (NaN for a chain not searched); distinct positions after the grading deduplication and before the optimizer; raw correlation of saved peak i before polishing; the first eight pool entries (primary first, then by raw correlation; NaN and origin -1 beyond the pool). A failure is a miss when no pool entry lies near the truth and a misrank when one does but is not the primary |

### Kernels, optimizer and objective normalisation

The singleray CPU search runs on fused numba kernels: the coarse grid on a per-channel travel-time stack cached once per process (about 36 MB for the record grid instead of two 158 MB delay sets), refine and polish grids on an inline table lookup with no per-pair delay matrices, and one packed correlation set per event shared by every stage. The delay-matrix path remains behind `use_fused_correlator: false` and on the GPU path. The keys below are all optional; with none of them set the results are those of the fused kernels with the strict table edge, the record objective and the L-BFGS-B optimizer. With the optimizer on, every position, correlation and saved-peak column of the record chain is bit-identical to the delay-matrix code on the synthetic suite, both emitter grids, the CR tier and pulser run1000; the coarse-map diagnostics (`peak_{i}_map_snr`, `peak_isolation_ratio`, `surf_corr_z`, `surf_corr_zen`) agree within about 1e-11 relative (largest measured 3.4e-12, `peak_2_map_snr_hpol` with the 15-channel polarization-group configuration), from the floating-point evaluation order of the fused coarse map. With `skip_optimizer` the reported correlations (`max_corr`, `peak_{i}_corr`) are refine-grid values from the fused kernel and agree within about 1e-13 relative; the positions stay exact. In candidate mode two sources of tiny differences can reorder near-degenerate candidates: the correlation envelope comes from the padded pair spectrum (up to 4.6e-9 from the Hilbert envelope of the cropped correlation) and the fused polish grids agree with the delay-matrix grids to 1e-9. Against the delay-matrix code 12 of 370 in-ice, 28 of 393 air-ice and 86 of 2650 CR events moved by more than 0.05 deg, with every tier metric inside its cluster-bootstrap 95 percent interval (median differences -0.004, -0.003 and +0.05 deg).

| Config key | Default | Description |
|------------|---------|-------------|
| `use_fused_correlator` | `true` | `false` selects the per-pair delay-matrix path of the record code (debugging and reference only) |
| `optimizer_method` | `L-BFGS-B` | `L-BFGS-B` or `Nelder-Mead` (scipy, per seed) or `compass`: a numba bounded search run on all seeds of a stage in one call (seeds in parallel). Each iteration takes the forward-difference gradient, scales it by the current steps and scans along it at 1/4 to 64 steps taking the best point (so the dips between correlation lobes are crossed, as the L-BFGS-B line search does; the scan stops at 2 steps once the steps are below 1e-3 of the initial steps), falls back to six axis and eight diagonal moves at the current steps with the same scan along the accepted move, halves the steps when nothing improves, wraps phi and clamps rho and z to `limits`. The objective is piecewise linear in position, so its maximum is a vertex and the stopping steps set the final precision; singleray only (multiray falls back to L-BFGS-B) |
| `optimizer_gradient` | `finite_difference` | Gradient L-BFGS-B receives with the objective, point source and far field. `finite_difference`: the forward differences scipy would take itself (absolute step 1e-8, steps adjusted to the bounds), computed with the objective in one compiled call per point (three far-field points in one `plane_wave_times` call), so the iterates and every result equal scipy's own differencing bit for bit. `exact`: the exact gradient of the objective (chain rule through the linear series interpolation and the bilinear tables, or through the plane-wave delays); one objective evaluation per point instead of four, different optimizer paths. Singleray only; multiray, custom polish objectives and Nelder-Mead keep scipy's differencing |
| `compass_step` | `[1.0, 0.2, 1.0]` | Initial compass steps (m, deg, m) |
| `compass_step_min` | `[1e-8, 2e-9, 1e-8]` | The search stops once every step is below these (m, deg, m); 1e-8 m is about 5e-10 in correlation at the typical slope, below L-BFGS-B's 1e-6 reproducibility between starting points (1e-10 m changes nothing measurable) |
| `compass_max_evals` | `1500` | Objective evaluations per seed (27 halvings from 1 m to 1e-8 m plus the ascent) |
| `compass_phi_scan` | `false` | Precede each compass search with an azimuth line scan of +/- 1.5 deg at 0.02 deg at the seed's rho and z |
| `objective_normalisation` | `total` | `total`: the weighted pair sum is divided by the total pair weight (record objective; pairs without a table solution or with a lag outside the correlation contribute 0 but keep their weight). `valid`: divided by the weight of the pairs that contributed, times a coverage factor equal to 1 when that weight is at least `valid_weight_floor` of the total and falling linearly to 0 at half the floor; singleray fused path only |
| `valid_weight_floor` | `0.6` | Coverage floor of the `valid` normalisation |

`use_fused_correlator`, `tolerant_table_edge`, `objective_normalisation` and `valid_weight_floor` are read by `begin()`, so changing them needs a new `begin()`; `optimizer_method` and the `compass_*` keys are read on every `run()` call (the flat `run()` path always uses L-BFGS-B). `begin()` rejects an unknown `optimizer_method`, compass steps that are not three positive numbers and a non-positive `compass_max_evals`.

Measured against L-BFGS-B on 189 synthetic events (station 23, VPol, SNR 50, 20 and 8): the compass cuts the time per event (record chain mean 0.166 to 0.132 s, median 0.161 to 0.094 s; candidate mode 0.301 to 0.177 s) with similar summary accuracy on that set (candidate mode median 0.0091 vs 0.0090 deg, p68 0.0150 vs 0.0152; record chain median 0.899 vs 0.902 deg, p95 61.9 vs 66.3 deg), but it does not dominate L-BFGS-B event by event. On the record chain it ends below L-BFGS-B on 139 of the 189 events (median deficit 3.8e-5 in correlation, 16 events by more than 1e-2, largest 0.40) and above it on 50 (11 by more than 1e-2, largest 0.31). The small differences are the two optimizers stopping at different vertices of the piecewise-linear objective, below L-BFGS-B's own 1e-6 reproducibility between starting points; the large ones are lobe paths, mostly at SNR 50 where the 1 m refine grid rarely lands inside the 0.25 m main lobe and one optimizer climbs through successive correlation lobes to a peak the other does not reach. On the emitter grids and the CR tier in candidate mode (`[envelope:traces, envelope:correlation, raw]`, paired per event against L-BFGS-B with cluster-bootstrap 95 percent intervals) the compass is worse in ice (median 0.474 to 0.540 deg, difference +0.066 [+0.009, +0.143], 23 events worse by more than 0.5 deg against 8 better) and not distinguishable from L-BFGS-B in air (median 1.221 to 1.322 deg, +0.10 [-0.04, +0.31]) or on the CR tier (3.68 to 3.84 deg, +0.16 [-0.25, +0.78]). The method is therefore optional and off by default; the test suite records the per-event dominance rule as strict expected failures.

Output field `objective_version` (0 for `total`, 1 for `valid`) records which normalisation produced the result and is written as a per-event dataset; the file attribute `objective_version` names the whole objective, including this normalisation (see the pair weights below). `max_corr`, `peak_{i}_corr`, `surf_corr_*` and `peak_isolation_ratio` change scale near the table edges under `valid`. Measured on the synthetic families (station 23, VPol, floors 0.5, 0.6 and 0.8; 50 synthetic pure-noise events stand in for the forced-trigger null gate, which was not run on recorded forced triggers): the `valid` normalisation moves pure-noise reconstructions toward the table edge (median reconstructed rho 45 m to 154 m at the default floor, fraction beyond 200 m 0.02 to 0.32) and loses far sources the record objective recovers, because a mean over fewer valid pairs has a larger noise spread and the coverage factor rewards exactly those regions. Candidate mode at the default floor on the emitter grids and the CR tier shows the same: median 0.474 to 0.821 deg in ice, 1.221 to 2.119 deg in air and 3.68 to 6.16 deg on the CR tier, with the fraction beyond 3 deg up by 0.18, 0.16 and 0.14. It therefore fails its acceptance gate and stays off; the test suite records the two gate failures as strict expected failures so that a change to the mechanism that passes them is noticed.

#### Two-arrival polish objective

A shallow source whose ray reaches the surface steeper than the critical angle arcsin(1/n(z_src)) from the vertical (48.6 degrees at z = -5 m) is totally reflected, so beyond a horizontal distance of about 90 m the deep antennas receive a full-amplitude surface-reflected pulse 10 to 40 ns after the direct one at z = -5 m (5 to 185 ns at z = -20 m). The single-arrival objective treats the reflected pulses as clutter. With `polish_objective: two_arrival_consistent` the polish stage evaluates every channel pair at the same solution index in both channels on the solution-ordered tables (`solution_0` and `solution_1`, loaded in `begin` from `multiray_table_name_pattern` in addition to the single-arrival tables; combos 00 and 11 only, no cross terms, because the reflection phase rotates from 0 at the critical angle to 180 degrees at grazing incidence and a fixed-sign cross term would be wrong) and sums the two contributions, the second weighted by a critical-angle mask: 1 where the solution_1 ray leaves the source steeper than the critical angle, 0 where the reflection is sub-critical (Fresnel amplitude about 0.12 near normal incidence). Where `solution_1` is undefined a pair contributes its solution_0 value only, so with the mask at 0 everywhere the objective equals the raw correlation.

The mask is computed from the table geometry at polish time: the horizontal slowness of the solution_1 ray at the source is the table gradient p = dT1/dR (central difference over 1 m, one sided at a table boundary), the launch zenith follows from sin(theta) = c p / n(z_src) with n(z) the `greenland_simple` exponential profile, and the ray is totally reflected when theta exceeds arcsin(1/n(z_src)), which is the condition c p > 1. A refracted solution_1 (turning point below the surface, c p >= n(0)) gets mask 1 as well, matching its full amplitude. The approximation is the bilinear table gradient, accurate to the table's interpolation error.

Status: experimental, not for production configs. The two-arrival value is on the raw scale plus the solution_1 contribution, so it credits any position with two valid solutions and mask 1 with up to twice the raw scale, and a pair's solution_1 delay can coincide with a direct-pulse alignment elsewhere in the volume. Ranked by that sum alone (the build measured on the benchmark battery) the mode is a net regression against the same three-chain configuration with the raw polish: on the cosmic-ray tier the fraction within 3 degrees fell from 0.478 to 0.377 (median 3.63 to 4.90 degrees; 1.000 to 0.919 on the high-quality cut), on the in-ice emitter grid from 0.968 to 0.927 (p95 1.95 to 5.20 degrees) and on the above-surface grid from 0.718 to 0.667 (p95 10.0 to 18.1 degrees); it wins only on the synthetic doublet family and on the in-ice sources beyond 150 m. The designed mitigation is `two_arrival_margin`: without it the ranking stays raw and the two-arrival value is a diagnostic; with it a position is ranked by the two-arrival value only where that exceeds its raw correlation by more than the margin, whose value must come from the inflation measured on the forced-trigger null. `two_arrival_margin: 0` is the closest setting to the measured build and carries its regressions. Do not set the margin in a production config until that measurement exists. The tie band (`candidate_tie_band`) compares raw correlations only, so with a margin it replaces a primary ranked by its two-arrival value with the raw chain's answer on every event whose `candidate_raw_chain_corr` is below the noise ceiling (or on every event without a ceiling), unless that primary's own raw correlation also beats the raw chain's by the band. The two-arrival value is always divided by the total pair weight and reads the tables with the tolerant edge rule, whatever `tolerant_table_edge` says; `objective_normalisation: valid` is rejected in this mode.

Output fields: `corr_two_arrival` (two-arrival value of the primary), `raw_corr_single` (raw single-arrival correlation at the primary position), and per saved peak `peak_{i}_corr_two_arrival` and `peak_{i}_raw_corr_single`. `max_corr` and `peak_{i}_corr` keep the raw single-arrival definition (`max_corr_source: raw`), so downstream consumers see the same quantity as before at the polished position; `max_corr_source: two_arrival` makes them carry the two-arrival value instead. With a margin the saved peaks are ordered by the margin rule, so neither `peak_{i}_corr` nor `peak_{i}_corr_two_arrival` is necessarily sorted; `post_optimizer_mode` is skipped in this mode. Memory: two more tables per channel loaded through the table loader (about 40 MB per channel in the record format) plus a packed copy of both per channel set for the kernels (another 40 MB per channel), about 450 MB loaded plus 450 MB packed for an 11-channel group.

### Per-polarization reconstruction

Set `polarization_groups` in the config to run independent reconstructions per polarization:

```yaml
channels: [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23, 4, 8, 11, 21]
polarization_groups:
  vpol: [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
  hpol: [4, 8, 11, 21]
```

VPOL is the primary result. HPOL results are stored with `_hpol` suffix (`rho_hpol`, `phi_hpol`, etc.). No cross-polarization pairs are formed.

### Validation metrics

Pass `--validation` to the driver to record per-channel SNR and quality metrics:

| Field | Description |
|-------|-------------|
| `ch{N}_snr` | Per-channel SNR (max|V|/std) |
| `pa_max_snr`, `pa_avg_snr` | Phased array SNR summary |
| `helper_b_max_snr`, `helper_c_max_snr` | Helper string max SNR |
| `n_helpers_above`, `n_channels_above` | Channels above SNR threshold |
| `has_helper_signal` | Boolean: any helper above threshold |
| `peak_isolation_ratio` | Top coarse peak over the mean of the top five coarse peaks. Higher = more confident. See the candidate-search table for the versioned `_v2` metrics |
| `surf_corr_z`, `surf_corr_zen` | Surface correlation quality metrics |

The record SNR is the peak-to-peak amplitude within a 3-sample window over twice the split-trace noise RMS, measured on the traces the reconstruction sees (10 GHz after upsampling, so the window is 0.3 ns and the value reads about a third of the pulse amplitude for the 100-700 MHz band and about 1 on pure noise). `snr_pair_weighting` uses these values. Set `snr_window_ns` (default null) to also measure every channel over a physical time window, `round(snr_window_ns x sampling rate)` samples (1.5 ns is 15 samples at 10 GHz and 5 at 3.2 GHz, and reads the pulse amplitude itself and about 3 on pure noise); the values are written beside the record columns under the window's suffix (`ch{N}_snr_w15` and `pa_avg_snr_w15`, `pa_max_snr_w15`, `helper_b_max_snr_w15`, `helper_b_min_snr_w15`, `helper_c_max_snr_w15`, `helper_c_min_snr_w15`, `n_helpers_above_w15`, `n_channels_above_w15`, `has_helper_signal_w15` for 1.5 ns), the counts gated by `helper_snr_threshold_windowed` (default 19.5: on the cosmic-ray simulation tier the windowed estimator reads 3.9 times the record value on lit helper channels and 19.5 reproduces the `n_helpers_above` distribution of `helper_snr_threshold: 5` with a histogram distance of 0.03 and the same helper count on 92 percent of events), and the driver writes the window and the windowed threshold to the HDF5 attributes `snr_window_ns` and `helper_snr_threshold_windowed`. The record columns, weights and results are unchanged by the key.

### Pair weights

With `snr_pair_weighting: true` every pair enters the objective with a weight. `pair_weight_mode: record` (the default) is the geometric mean of the two record SNRs, normalised to a maximum of 1. `pair_weight_mode: information` (needs `snr_window_ns` and `snr_pair_weighting: true`; `begin()` rejects it otherwise) weights each pair by its inverse timing variance, `1 / (floor^2 + (k / SNR_i)^2 + (k / SNR_j)^2)` normalised to a maximum of 1, with the windowed SNR, `pair_weight_k_ns` (default 12.65, the bandwidth-limited timing error at unit SNR) and `pair_weight_floor_ns` (default 2.0, the pulse-shape mismatch that does not fall with SNR): a pair with a noise channel keeps a small weight without a hard gate and the loudest channels saturate above SNR about k / floor. Under the information weights `max_corr` and every peak correlation are weighted means on a different scale from the record's; the driver writes the HDF5 attribute `objective_version` with the value of `InterferometricReco3D.objective_version(config)` (`record` for the record objective; otherwise the pair weight mode, the HPol sign mode and the valid-weight normalisation that changed it) and, when it is not `record`, the attributes `pair_weight_mode`, `hpol_sign_mode` and `objective_normalisation`.

### HPol polarity

The horizontal Askaryan field flips sign across the vertical plane through the shower axis, so the HPol channels of strings on opposite sides of that plane see pulses of opposite polarity and the signed correlation of such a cross-string pair scores the true delay at minus its peak. With `polarization_groups` containing a group named `hpol`, `hpol_sign_mode` chooses how that group's raw correlations are scored: `signed` (default, the record behaviour), `abs_cross_string` (absolute value of the correlation for pairs of different strings, signed for pairs within one string, the strings being the power string 0-8, helper B 9-11 and helper C 21-23; for the HPol group 4, 8, 11, 21 these are the five cross-string pairs 4-11, 4-21, 8-11, 8-21 and 11-21, while 4-8 stays signed) or `joint_sign` (the group is reconstructed once per relative sign assignment of its strings, 4 runs for 3 strings, and the run with the largest correlation is kept, which is the signed objective maximised over the assignments). The VPol group is never touched. Envelope chains of the candidate search ignore the signs (an envelope carries no polarity); the raw chain and the raw polish use them. With a mode other than `signed` the HPol fields (`rho_hpol`, `max_corr_hpol`, `peak_{i}_*_hpol`) are on the mode's objective and the result carries `sign_mode_hpol` (1 abs_cross_string, 2 joint_sign); `joint_sign` adds `sign_assignment_hpol` (index of the winning assignment, `itertools.product((1, -1))` over the strings after the first in channel order, so 0 is signed, 1 flips helper C, 2 flips helper B, 3 flips both for the record group) and `sign_corr_{k}_hpol` (every assignment's correlation). The driver writes `objective_version` and `hpol_sign_mode` as HDF5 attributes. `pair_signs` (a list with one entry per pair in `itertools.combinations(channels, 2)` order: 1, -1 or `abs`) is the per-pair mechanism the modes use and can be set directly for a single-group configuration; `begin()` rejects it together with `polarization_groups` and when its length is not the number of channel pairs.

### Coherent waveforms

Set `save_coherent_waveforms: true` and `n_coherent_waveforms: 3` to save the beam-formed waveform at each peak's reconstructed position. Only available in singleray mode (`multi_ray_types: false`). Stored in a separate `coherent_waveforms` HDF5 group. Useful for CNN-based peak selection or signal quality assessment.

The group holds `times` (ns, the time axis of the first event of the file that has a waveform) and `peak_<i>` of shape (N, n_samples), with zeros for an event without that waveform. With `polarization_groups` it also holds each group's waveforms and time axis as `peak_<i>_<group>` and `times_<group>`; the datasets without a suffix are those of the primary group. `--save-nur PATH` writes the waveforms of the primary result to a NUR file as channels 100 + i, sampled at the rate of `times` (10 GHz with the default upsampling); the trace start time is not stored there, so take the time axis from `times`. An event without a stored waveform is not written; the file itself always is, so with `save_coherent_waveforms` off it holds no event (`eventReader` opens it and yields nothing).

To form a summed waveform at any other position, `InterferometricReco3D.travel_times(rho, phi_deg, z, channels)` returns the table travel time in ns from a position to each listed channel, after `begin` with single-ray tables. The position is in the frame of the results: rho (m) and azimuth (deg) about the vertical axis through the phased-array centre, the mean of the positions of channels 1 and 2, and z (m) with the surface at 0. A value is NaN where the channel's table holds no ray solution and minus infinity outside the table's range (above the surface with the in-ice tables).

### Pair store and the search on stored pairs

The hierarchical reconstruction is two stages. `InterferometricReco3D.compute_pairs(station, config)` reads the preprocessed traces and returns a `PairSet`: the cross-correlation series of every channel pair for each envelope mode the search reads (raw; `traces`, the correlation of the trace envelopes; `correlation`, the envelope of the raw correlation, both taken before the Hann taper), their lag axes (lag = t_a - t_b of pair (a, b)) and the channel SNRs. `reconstruct_from_pairs(pairs, config, channel_mask=None, pair_weights=None, channel_delay_shift=None, channel_polarity=None)` runs the whole search on them (coarse grid, candidate chains, raw ranking, tie band, noise ceiling, polish, refine, optimizer, per polarization group and HPol sign mode). `run` of a hierarchical configuration is the second on the first, so the two agree bit for bit. `channel_mask` removes channels, `pair_weights` maps (ch_a, ch_b) to a weight replacing the SNR weights, `channel_delay_shift` maps a channel to ns added to its cable delay (the convention of the delay-corrections files; the lag axis of pair (a, b) moves by shift_b - shift_a) and `channel_polarity` maps a channel to +1 or -1 (the raw series of a pair is multiplied by the product of its polarities, which equals negating the trace bit for bit; envelopes carry no polarity).

With `save_pair_store: true` in the config, or `--pair-store PATH`, the driver writes the pass-1 series of every event to an HDF5 pair store (`pair_store.py`, default path `<output stem>_pairs.h5`; layout in the module docstring): every pair of the configured channels, all three envelope modes, the record and windowed SNRs, the event keys and time, the pair weights of the configured search, the config and its hash and the code commit. Each series is kept only over the lags the loaded tables can reach anywhere in the configured volume (`pair_lag_windows`, a bound from the table cell corners and the antennas' horizontal separation) plus `pair_store_margin_ns` (default 20 ns) on each side for delay-shift trials, in `pair_store_dtype` (default float32). Pass 1 then runs on the cut and rounded series, so `reco_from_pairs.py --pairs <store> -o <results>` reproduces the pass-1 results of that run exactly, and with options (`--config`, `--mask`, `--polarity 6:-1`, `--delay-shift 22:9.0`, `--calibration <yaml>`) runs the same search under a variation. `reconstruct_from_pairs` refuses series made with another Hann or normalisation setting, a configuration that needs an envelope mode the store lacks, and a volume or shift whose delays leave the stored windows. Pass 2 (rx and rxtx dedispersion) changes the waveforms and is not kept; `save_coherent_waveforms`, the driver's `plane_wave_fallback` decision and `tdoa_mode` need the traces; the Nelder-Mead optimizer is not bounded by the volume and is not covered by the windows.

### Region hypotheses

With `region_hypotheses: true` every result (each polarization group) also reports the best position of the same search below and above the ice surface: `below_*_v1` and `above_*_v1` with `rho`, `phi`, `z`, `corr_raw`, `corr_env_traces`, `corr_env_correlation` (the three objectives at that position), `map_snr` (the record map SNR on the coarse map of the saved peaks) and `origin` (chain code, -1 when the region holds no position). The candidates are every optimizer and polish output and every graded refined peak; the best of a region is the one with the highest search value. Under the split z grid the region is the z block that produced a position (so z = 0 belongs to its block), otherwise the sign of z (z > 0 above). The primary result and every other field are unchanged by the key.

### Far-field hypothesis

With `far_field_hypothesis: true` every result also reports the best plane-wave direction over the sky from the same pair series and weights: `far_zen_v1` (sky zenith of the arrival direction, deg), `far_az_v1` (azimuth toward the source, deg, the convention of `phi`), `far_corr_raw_v1`, `far_corr_env_traces_v1`, `far_corr_env_correlation_v1`, `far_map_snr_v1` and `far_origin_v1`. The arrival times (`plane_wave_times`) keep the horizontal slowness sin(zen) / c through the stratified medium: -sin(zen) (x cos(az) + y sin(az)) / c plus, below the surface, the integral of sqrt(n(z)^2 - sin(zen)^2) / c from the antenna depth to the surface with the tables' exponential profile, and above it -z cos(zen) / c. The search maps the raw and both envelope objectives on a 1 degree sky grid, refines the three best maxima of each on the raw correlation (0.1 degree steps) and finishes with L-BFGS-B. The point-source tables end at z = +300 m and rho = 1600 m, where a plane wave still differs from the nearest point source by the wavefront curvature (about 0.6 ns rms over the pairs at 1600 m); the far-field hypothesis has no such limit. Pair-store windows include the far-field delays.

The plane wave's n(z) is that of the tables' ice model: the `ice_model` the tables record (the air-ice generator writes it; all tables must agree), else the config key `ice_model` (a NuRadioMC ice model name, e.g. `greenland_3exp_layered`, the three-layer exponential of RNO-G antenna positioning; it is trusted for tables without metadata, such as the in-ice record tables), else `greenland_simple`, the profile of the record tables. A config key that differs from the model the tables record raises. Single-exponential models (`IceModelSimple`) keep the closed exponential form; layered models (`IceModelExpLayers`, `IceModelContinuousExpLayers`) are integrated layer by layer with the same Gauss-Legendre rule, which is exact to float rounding on each smooth layer.

Narrow-band sources (solar bursts, with a carrier period of about 8.6 ns) correlate in lobes one carrier period apart, and the raw sky maximum can put some pairs on a neighbouring lobe. `far_field_lobe_guard_ns: <ns>` (off by default) chooses the far-field direction on the envelope of the correlation, which has no lobes, and lets the raw correlation refine it only among the directions where every weighted pair delay stays within that many ns of its delay at the envelope peak (+/- 5 degrees searched at 0.1 degree, then L-BFGS-B); half the carrier period is the natural value. `far_origin_v1` is then 2 (envelope of the correlation).

### Unreadable travel-time cells and tables of differing extent

A channel's travel time is read only where the query lies inside its table and the four corners of its cell are finite, with a positive value; elsewhere every pair with that channel adds nothing at that point while its weight stays in the divisor (`objective_normalisation: total`; with `valid` the divisor is the weight read times the coverage factor). This holds for the coarse map, the refine and polish grids, the optimizer and the candidate grading. Tables of one spacing may cover different r and z ranges (for example LPDA tables from -100 m next to deep tables from -1600 m); a query outside a table is unreadable like a NaN cell, and the lag windows handle both.

### Channel polarity, delay and position calibration

A delay-corrections file (`apply_delay_corrections` with `delay_corrections_file` in the `preprocessor` block, ROOT input only) may carry an optional `polarity` block and an optional `position_shift` block beside `corrections`:

```yaml
derived_from: {23: {runs: [999, 1000]}}
method: deep pulser timing fit
date: 2026-10-03
valid_from: 2022-01-01
valid_to: 2022-12-31
corrections:        # ns added to the database cable delay
  23: {5: -2.83}
uncertainty_ns:
  23: {5: 0.05}
polarity:           # +1 or -1 per channel; -1 negates the trace
  23: {6: -1, 7: -1}
position_shift:     # [dx, dy] in m added to the database position
  23: {22: [1.3, -0.7], 23: [1.3, -0.7]}
```

The preprocessor shifts each corrected channel's trace start time by minus its correction and negates the traces of the channels marked -1, within the same validity windows. On a pair store, `pair_store.calibration_trial(store.applied_calibration(), load_delay_corrections(new), station_id)` gives the delay shifts and polarities that turn the stored series into those of another file, which `reco_from_pairs.py --calibration` applies after checking each event's date against the new file's window.

Position shifts are not a trace operation: the driver passes the station's `position_shift` block to the reconstruction as `channel_position_shift` (a dict channel -> (dx, dy) in m; it may also be given in the config, but not in both places), and `reconstruct_from_pairs(..., channel_position_shift=...)` takes the same dict. The shift moves the channel's horizontal position wherever the search reads positions: the grid, refine and optimizer kernels, the lag windows (cut series whose stored windows cannot hold the shifted delays are refused, as for delay shifts), the region hypotheses and the far field's plane-wave times. Each channel's table is axially symmetric about its antenna, so no table changes; vertical shifts would need new tables and are refused. The reconstruction frame (the phased-array centre of the database) does not move, and the pass-2 template carries the shift. `reco_from_pairs.py --calibration` uses the new file's `position_shift` block (none: no shift) in place of the store configuration's, since positions do not enter the stored series. A zero shift is bit-identical to no key, and a shift equals editing the channel's position in the detector description to float rounding (`tests/test_position_shift.py`).

## Real data (ROOT files)

The driver auto-detects ROOT vs NUR input. For ROOT files, it uses `readRNOGData` with `read_daq_status=False` to avoid requiring the `combined` tree (not present in all data versions). No other changes needed. With that setting a run folder needs `waveforms.root` and `headers.root` only; `daqstatus.root`, which hand-carried runs can lack, is required only by a caller that reads it.

A `reader_kwargs` block in the config is passed to `readRNOGData.begin` on top of these defaults, for example `reader_kwargs: {select_triggers: FORCE}` to reconstruct only the forced triggers of a run. Its `mattak_kwargs` entry is merged key by key into the driver's (`read_daq_status: false`, `backend: uproot`), so a config changes only the keys it names (`select_data_provider`, below). NUR input does not read the block.

```bash
python interferometric_reco_3d_advanced.py \
    --config configs/reco3d_neutrino_gzk.yaml \
    --mode hw \
    -i /path/to/station21/run1234/waveforms.root \
    -o results/run1234.h5
```

## Multiray travel time tables

Standard interferometric reconstruction assumes a single ray path between source and receiver. In a medium with a depth-dependent refractive index profile, signals can propagate via multiple paths: direct, reflected (off the surface), and refracted (bent by the index gradient). At many source geometries, two or more of these paths arrive with comparable amplitude, so the correct ray type varies by channel and source position.

The 3D module uses per-channel, per-ray-type travel time tables. In `grouped` mode (`multiray_combo_mode: "grouped"` in the config), it evaluates all physically valid ray-type combinations and selects the one that maximizes the summed correlation. Channels at similar depths are grouped together (they see the same ray type), reducing the combinatorial cost. The coarse grid scores every pair with its own best combination for speed; the refine grids and the optimizer use the grouped combinations, with or without `use_fused_correlator`. `per_pair` mode takes the per-pair maximum at every grid stage and refines on the fused kernel.

Each table is a 2D (R, Z) grid of travel times for one channel and one ray type, stored as an NPZ file. For station 23's Vpols (11 channels) and 4 table types (direct, reflected, refracted, plus combined), this is 44 files. The 3D configs in this directory use the per-ray-type tables (direct, refracted, reflected). The combined tables (no suffix, min travel time across ray types) are used when `multi_ray_types: false`.

### Generating tables

The table generators are in `tables/`:

```bash
cd tables/

# Single channel, multiray only (default)
python rz_lookup_table_creator_inice.py \
    --station 23 --channel 0 --num_threads 8 \
    --output-dir /path/to/multiray_tables/station23

# Multiray + combined tables in one pass
python rz_lookup_table_creator_inice.py \
    --station 23 --channel 0 --mode all --num_threads 8 \
    --output-dir /path/to/multiray_tables/station23
```

The `--mode` flag controls output: `multiray` (3 per-ray-type files), `combined` (1 min-time file), `solution_ordered` (2 solution-ordered files), or `all` (all of the above). The `--det-date` argument sets the detector description date used for antenna positions (default `2022-10-01`). Use `--detector-file` to read a local detector export instead of querying MongoDB. Tables are computed using NuRadioMC analytic raytracing with the `greenland_simple` exponential ice model. Each channel takes roughly 6 minutes on 9 cores and uses about 5 GB of memory.

### Solution-ordered tables

With the `greenland_simple` ice model, each (R,Z) geometry has 0 or 2 ray solutions (direct + refracted). The reflected solution is rare. Solution-ordered tables reorder by travel time: solution_0 = fastest arrival, solution_1 = slowest. This reduces the grouped correlator combinations from 3^N_groups (81 for 4 depth groups) to 2^N_groups (16), giving a 1.2-1.6x speedup with slightly better accuracy in the optimization tail.

To generate solution-ordered tables:

```bash
python rz_lookup_table_creator_inice.py \
    --station 23 --channel 0 --mode solution_ordered --num_threads 8 \
    --output-dir /path/to/multiray_tables/station23
```

To use them, set `table_scheme: "solution_ordered"` in the config (commented line in `reco3d_neutrino_gzk.yaml` and `reco3d_pulser_sim.yaml`).

## Using a different station

The shipped configs are for station 23. To run on a different station, copy a config and change `station_id` to your station number. Make sure travel time tables for your station exist at the path specified by `time_delay_tables`. The code looks for files at `<time_delay_tables>/station{ID}/st{ID}_ch{N}_rz_table_{ray_type}.npz`. All other config parameters (channels, grid limits, preprocessing) are the same across stations.

## Optional features summary

All optional features are off by default. Enable via config YAML or CLI flags.

| Feature | Config key | CLI flag | Default |
|---------|-----------|----------|---------|
| Multi-peak retention | `n_peaks_save: 3` | -- | 1 (single peak) |
| Coarse map in the result dict | `save_coarse_map: true` | -- | false |
| Per-polarization | `polarization_groups: {vpol: [...], hpol: [...]}` | -- | None (all channels together) |
| Coherent waveforms | `save_coherent_waveforms: true`, `n_coherent_waveforms: 3` | -- | false |
| Validation metrics | `validation: true` | `--validation` | false |
| Time-window SNR columns | `snr_window_ns: 1.5`, `helper_snr_threshold_windowed: 19.5` | -- | null (record 3-sample SNR only) |
| Information pair weights | `pair_weight_mode: information`, `pair_weight_k_ns: 12.65`, `pair_weight_floor_ns: 2.0` (with `snr_window_ns`) | -- | record (geometric-mean SNR weights) |
| HPol polarity | `hpol_sign_mode: abs_cross_string` or `joint_sign` (with a `polarization_groups` entry `hpol`), `pair_signs` | -- | signed |
| Plane wave fallback | `plane_wave_fallback: true`, `plane_wave_snr_threshold: 5.0` | -- | false |
| Bandpass filter | `preprocessor: {apply_bandpass: true, bandpass_band: [0.1, 0.7]}` | -- | false |
| Notch | `preprocessor: {apply_notch: true, notch_bands: [[0.399, 0.407]]}` | -- | false |
| Search geometry | `refine_window_mode: adaptive`, `subbin_coarse_seeds: true`, `tolerant_table_edge: true` | -- | fixed windows, bin-centred seeds, strict table edge |
| Candidate search | `candidate_search: [envelope, raw]` or `[envelope:traces, envelope:correlation, raw]`, `candidate_envelope_mode`, `candidate_polish_window`, `candidate_polish_steps`, `candidate_include_refined`, `candidate_fill_saved_peaks`, `candidate_diagnostics`, `candidate_tie_band`, `candidate_tie_band_max_raw_corr` | -- | absent (single search chain) |
| Split z grid at the surface | `z_grid_below: {n: 100, spacing: linear}`, `z_grid_above: {n: 60, spacing: log, offset: 1.0, refine_spacing: linear}` | -- | absent (one z vector from `coarse_n_z` and `z_spacing`) |
| Two-arrival polish (experimental) | `polish_objective: two_arrival_consistent`, `two_arrival_weight_mode`, `two_arrival_second_weight`, `two_arrival_margin`, `max_corr_source` (needs `candidate_search`, the solution-ordered tables, `multi_ray_types: false` and `objective_normalisation: total`) | -- | `raw` (polish by the raw correlation) |
| Compass optimizer | `optimizer_method: compass`, `compass_step`, `compass_step_min`, `compass_max_evals`, `compass_phi_scan` | -- | L-BFGS-B |
| Valid-weight normalisation | `objective_normalisation: valid`, `valid_weight_floor: 0.6` | -- | total |
| Delay corrections and channel polarity (real data) | `preprocessor: {apply_delay_corrections: true, delay_corrections_file: <yaml>}` (optional `polarity` block in the file) | -- | false |
| Pair store | `save_pair_store: true`, `pair_store_margin_ns: 20`, `pair_store_dtype: float32` | `--pair-store PATH` | false |
| Region hypotheses | `region_hypotheses: true` | -- | false |
| Far-field hypothesis | `far_field_hypothesis: true` | -- | false |
| Far-field lobe guard (ns) | `far_field_lobe_guard_ns: 4.3` | -- | null |
| Ice model of the tables (far-field n(z)) | `ice_model: greenland_3exp_layered` | -- | the tables' recorded model, else greenland_simple |

## Validation on reference sets

### Datasets

Reference validation sets for station 23 live on the Chicago cluster. Each directory has its own `README.md` with full specs (event count, energy/geometry coverage, filename conventions, generation notes).

- **GZK neutrino simulation** (27,667 triggered events, GZK-weighted 10^18-10^20 eV, 300 K thermal noise): `/data/reconstruction/validation_sets/sim_neutrinos/sim_output_gzk/`
- **Simulated pulser calibration** (18,879 triggered events over a 3D emitter grid around the station): `/data/reconstruction/validation_sets/sim_cal_pulsers/test_set/`

### Resource estimates

Estimates below use the solution-ordered (2-table) scheme. Ray-type (3-table) runtimes are 1.2-1.6x longer.

#### Neutrino GZK (27,667 events, hw mode)

| Resource | Estimate |
|----------|----------|
| Time per event | ~1.4 s |
| Total CPU time | ~11 CPU-hours |
| Recommended chunks | 200 |
| Walltime per chunk | 10 min |
| Memory per chunk | 4 GB |

#### Pulser sim (18,879 events, rxtx mode)

| Resource | Estimate |
|----------|----------|
| Time per event | ~7.8 s |
| Total CPU time | ~41 CPU-hours |
| Recommended chunks | 200 |
| Walltime per chunk | 25 min |
| Memory per chunk | 4 GB |

#### Pulser sim (18,879 events, hw mode)

| Resource | Estimate |
|----------|----------|
| Time per event | ~3.5 s |
| Total CPU time | ~18 CPU-hours |
| Recommended chunks | 100 |
| Walltime per chunk | 15 min |
| Memory per chunk | 4 GB |

### Validated results

All results below are on station 23 with the shipped configs and validation datasets. Numbers are angular separation between the reconstructed vertex direction and the true vertex direction.

#### Neutrino GZK (hw mode, 27,667 events, 300K noise)

| Cut | N | Median | p68 | < 1 deg | < 3 deg |
|-----|------|--------|------|---------|---------|
| All events | 27,667 | 1.48 deg | 3.30 deg | 39% | 66% |
| corr >= 0.3 | 14,915 | 1.15 deg | 2.18 deg | 46% | 75% |
| corr >= 0.3, reco z < -200m | 12,863 | 0.94 deg | 1.65 deg | 52% | 82% |
| corr >= 0.5 | 9,346 | 1.04 deg | 1.93 deg | 49% | 78% |
| corr >= 0.7 | 4,314 | 0.82 deg | 1.39 deg | 58% | 89% |

Runtime: ~2.2 s/event (ray-type tables), ~1.4 s/event (solution-ordered tables). Both schemes give identical accuracy.

#### Pulser sim (hw mode, 18,879 events)

| Distance | N | Median | p68 |
|----------|------|--------|------|
| 10-30m | 3,149 | 3.57 deg | 4.54 deg |
| 30-50m | 2,782 | 1.49 deg | 2.59 deg |
| 50-100m | 5,681 | 1.00 deg | 1.55 deg |
| 100-150m | 4,354 | 0.77 deg | 1.41 deg |
| 150-200m | 2,663 | 0.52 deg | 0.92 deg |
| All | 18,879 | 1.42 deg | 2.38 deg |

With rxtx mode (antenna dedispersion): 0.42 deg median on the full set.

Configs: `reco3d_neutrino_gzk.yaml` (neutrino), `reco3d_pulser_sim.yaml` (pulser). Your results should match when using the same configs, tables, and datasets.
