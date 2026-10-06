# RNO-G feature extraction

Writes one table row per event with the variables of `NuRadioReco/modules/RNO_G/stationFeatureExtractor.py`.

## Contents

- [Files](#files)
- [Usage](#usage)
- [Config](#config)
- [Output](#output)
- [Tests](#tests)
- [Known limitations](#known-limitations)

## Files

| File | Purpose |
|---|---|
| `feature_extraction.py` | Driver: reads the inputs through a data provider, runs the module on every selected event, writes the table |
| `tests/test_station_feature_extractor.py` | The module's rows on synthetic events: columns, values against direct calls of the per-trace functions, missing channels, config, hit filter counts |
| `tests/test_feature_table.py` | The driver: event selection, output path, the written table read back |
| `tests/synthetic.py` | Builds the synthetic events |

The variables themselves are computed by the module (channel groups, group means, summed traces, H/V pairs, hit filter counts) with the per-trace functions of `NuRadioReco/utilities/trace_utilities.py`. The class description of the module lists the columns.

## Usage

```bash
python feature_extraction.py --config features.yaml \
    --input /path/to/station13/run1000 /path/to/station13/run1001 \
    --station_id 13 --year 2022 --experiment_id my_run --run_chunk 0
```

Inputs are RNO-G run folders or `.root` files, or `.nur` files; all of one kind. `--events` restricts the events: a list of event numbers, or a JSON file `{run: [events]}` or `{file name: [[run, event], ...]}` (`NuRadioReco.utilities.io_utilities.parse_event_ids`). `--station_id`, `--year` and `--experiment_id` replace the config keys of the same name.

## Config

A YAML file. The driver reads these keys; any other key at the top level raises:

| Key | Meaning |
|---|---|
| `station_id`, `year`, `experiment_id` | Name the output path; `station_id` also selects the detector description |
| `detector_file`, `detector_date` | Detector description, as for the reconstruction driver in `../interferometric_reco_ex/` |
| `preprocessor` | Passed to `channelPreprocessor` through the data provider. The module does no preprocessing of its own |
| `reader_kwargs` | Passed to the reader of the data provider |
| `output_root_dir` | Root of the output path. Default: the environment variable `FEATURE_OUTPUT_ROOT`, then `./feature_extraction` |
| `features` | Config of the module, see `stationFeatureExtractor.begin`. An unknown key raises |

Example with the module's defaults written out:

```yaml
station_id: 13
year: 2022
experiment_id: my_run
preprocessor:
  apply_cw_removal: true
  apply_bandpass: true
  bandpass_band: [0.1, 0.7]
features:
  feature_groups: [snr, rpr, max_amplitude, impulsivity, kurtosis_entropy, spectral, band, impulse_correlations]
  build_coherent_sums: true
  hit_filter: false
  spectral_fmin: 0.08          # GHz
  spectral_fmax: 0.6
  spectral_low_band_boundary: 0.1
  band_lo: 0.1
  band_hi: 0.3
  channel_groups:
    pa: [0, 1, 2, 3]
    vpol: [0, 1, 2, 3, 5, 6, 7, 9, 10, 22, 23]
    hpol: [4, 8, 11, 21]
    deep: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23]
  hv_pairs:
    b: {hpol: 11, vpol: [9, 10]}
    c: {hpol: 21, vpol: [22, 23]}
```

## Output

`<root>/results/<sim_data|real_data>/<experiment_id>/station<id>/<year>/station<id>_features_df_chunk<run_chunk>_<experiment_id>.h5`, a pandas table under the key `data` sorted by run and event number, with the config as attributes of the group `config`. Besides the module's columns each row has `run_number`, `event_number`, `source_file`, `trigger_time` (unix time of the station) and, for `.nur` files named `..lgE_<value>..`, `log10_energy`.

With the default config a row has 31 variables for each of the 15 channels, 25 group means and 22 summed-trace variables for each of the 4 groups, and 7 polarisation variables: 660 columns, 665 with `hit_filter: true`.

To use the module without the driver:

```python
from NuRadioReco.modules.RNO_G.stationFeatureExtractor import stationFeatureExtractor

extractor = stationFeatureExtractor()
extractor.begin({"feature_groups": ["band"], "band_lo": 0.15, "band_hi": 0.6})
row = extractor.run(event, station, det)
```

## Tests

Run from `tests/` with the checkout importable:

```bash
python -m pytest -q
```

`NuRadioReco/test/utilities/test_trace_utilities.py` covers the per-trace functions.

## Known limitations

- The module expects preprocessed traces of equal length within a channel group; it does not resample.
- `hpol_vpol_band_snr_ratio` is written only if groups named `hpol` and `vpol` are configured.
- The hit filter counts follow the fixed channel layout of `stationHitFilter` (phased array and three antenna pairs on the strings), not `channel_groups`.
- A summed trace is aligned by cross correlation to the first available channel of its group, with `trace_utilities.get_coherent_sum`. For a sum aligned at a reconstructed position, align the traces yourself and call `stationFeatureExtractor.get_summed_trace_features`.
- Samples at the same distance from the envelope maximum enter the impulsivity in order of their envelope value. Tables made on the branch `rnog-analysis`, which left such ties unsorted, differ in the impulsivity columns: by up to 3.6e-4 in the impulsivity itself on 900 random traces, by more in its linearity variables.
