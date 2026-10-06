# Pedestal extraction: per-channel ADC clip thresholds

The RADIANT 12-bit ADC digitizes a range of 0 to 2.5 V. The pedestal (DC baseline) sits near 1.5 V and not at the 1.25 V midpoint, so a pedestal-subtracted trace has about 1.5 V of headroom on the negative side and about 1.0 V on the positive side before it saturates.

`../simulate.py` clips the readout traces at these bounds. `--clip_thresholds <yaml>` gives them per channel. Without the file a single range from `--pedestal_voltage` is used for all channels.

## Files

| File | Description |
|------|-------------|
| `pedestal_analysis.py` | Extracts the per-channel pedestals from `pedestal.root` files and writes the clip thresholds |
| `clip_thresholds_station{12,13,21,22,23,24}.yaml` | Clip thresholds the 2022 simulations were run with |

## Method

Each RNO-G run contains a pedestal measurement for all 24 channels (4096 values per channel). The script

1. collects the `run*/pedestal.root` files below a data directory,
2. takes the mean pedestal per channel and run in ADC counts and converts it to mV,
3. optionally keeps only the runs of one year, using the UTC time stamp in the file,
4. takes the median over the runs for each channel,
5. writes the bounds `low = -median` and `high = 2500 - median` in mV.

```bash
python pedestal_analysis.py \
    --data_dir /path/to/station23/ \
    --station_id 23 \
    --year 2022 \
    --outdir .
```

The output is `clip_thresholds_station{id}_{year}.yaml`, which can be passed to `simulate.py --clip_thresholds`, and `pedestal_analysis_results.npz` with the pedestal of every run and channel. The script needs `uproot` and `joblib` and uses 20 worker processes unless `SLURM_CPUS_PER_TASK` is set.

## Shipped values

The YAML files hold the bounds that the 2022 simulations were run with. The `metadata` block of each file gives the number and the kind of pedestal runs behind it.

## Known limitations

- One median per channel. Pedestals drift, and some channels vary from run to run within a year. A second extraction for station 23 from 1,124 runs between 2022-06-27 and 2022-10-01 differs from the shipped bounds by up to 108 mV.
- The detector database does not store the pedestal voltages, so the bounds have to be passed to the simulation as a file.
