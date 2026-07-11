# Pedestal extraction: per-channel ADC clip thresholds

The RADIANT 12-bit ADC digitizes over a 0-2.5 V range (4096 counts). The pedestal (DC baseline) sits at approximately 1.5 V, not the 1.25 V midpoint. This makes the effective dynamic range asymmetric: signals can swing further negative (~1.5 V headroom) than positive (~1.0 V headroom) before clipping.

`simulate.py` applies this clip in the readout resampler. Per-channel bounds come from
`--clip_thresholds <yaml>` (`{ch: [lo_mV, hi_mV]}`); with no file it falls back to a single
uniform range built from the scalar `--pedestal_voltage`, an approximation.

## Shipped production values

The number is the per-channel pedestal position in the 0-2500 mV RADIANT range, turned into
asymmetric saturation bounds (`clip- = -pedestal`, `clip+ = 2500 - pedestal`). The shipped
`clip_thresholds_station{11,12,13,21,22,23,24}.yaml` carry the per-station v9 production
`CLIP_THRESHOLDS_MV` (from the `simulate_fixed_response_v9_st{NN}.py` production script copies;
base = station 23), so `--clip_thresholds` reproduces production. Pedestal source per station is
recorded in each file's metadata (station 23 = 6849 satellite runs; 21/22 = handcarry;
11/12/13/24 = satellite). Station 23 also records the never-adopted extracted values
(`clip_thresholds_station23_2022.yaml`, `pedestal_analysis.py` rerun) as `later_measurement`.
Re-derive for a new station or epoch with `pedestal_analysis.py`; it will not reproduce the
shipped production dicts to the digit (pedestals drift), so the shipped YAMLs pin what production
used.

## Files

| File | Description |
|------|-------------|
| `clip_thresholds_station{NN}.yaml` | Shipped per-station v9 production clip thresholds (used by `simulate.py --clip_thresholds`) |
| `pedestal_analysis.py` | Re-derives per-channel pedestal voltages / clip thresholds from pedestal.root files |
| `clip_thresholds_station23_2022.yaml` | Extracted station-23 clip thresholds (superseded generation; embedded as `later_measurement` in `clip_thresholds_station23.yaml`) |
| `pedestal_analysis_results.npz` | Raw per-run, per-channel pedestal voltages for further analysis |

## Method

Each RNO-G run includes a pedestal measurement that records the ADC pedestal distribution (4096 bins) for all 24 channels. The script:

1. Crawls a data directory for `run*/pedestal.root` files
2. Extracts the mean pedestal (in ADC counts, converted to mV) per channel per run
3. Optionally filters by year using the UTC timestamp in the ROOT file
4. Computes the median pedestal across all qualifying runs for each channel
5. Derives asymmetric clip thresholds: `clip_negative = -median`, `clip_positive = 2500 - median` (in mV)

## 2022 clip thresholds

Derived from 1,124 runs for station 23 between 2022-06-27 and 2022-10-01. Sample values:

| Channel | Pedestal (mV) | Clip- (mV) | Clip+ (mV) |
|---------|--------------|------------|------------|
| ch0 | 1416 | -1416 | +1084 |
| ch1 | 1467 | -1467 | +1033 |
| ch3 | 1593 | -1593 | +907 |
| ch9 | 1560 | -1560 | +940 |

The full set is in `clip_thresholds_station23_2022.yaml`.

## Usage

```bash
# Extract thresholds for 2022 (requires pedestal data and SLURM)
python pedestal_analysis.py \
    --data_dir /path/to/station23/ \
    --station_id 23 \
    --year 2022 \
    --outdir .

# Use in the CR proxy simulation
python ../simulate.py \
    ... \
    --pedestal_voltage 1.5
```

The script uses `joblib` for parallel processing of ROOT files. Run with `--cpus-per-task=20` on SLURM for the full 6,849-file dataset.

## Known limitations

- **Station 23, 2022 only.** The included `clip_thresholds_station23_2022.yaml` was derived from station 23 runs between 2022-06-27 and 2022-10-01. Other stations require separate extraction.

- **Single median per channel.** Pedestals drift over time, so the script uses `--year` to restrict to a single year. However, even within a year some channels show run-to-run variation that a single median doesn't capture.

- **No database integration.** The RNO-G MongoDB does not currently store `adc_pedestal_voltage`. Pedestals must be set at runtime via `--pedestal_voltage` (single value) or `set_pedestal_voltage(dict)` (per-channel). Adding this field to the DB would allow automatic pedestal loading.