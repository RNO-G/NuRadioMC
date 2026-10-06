# Forced-trigger noise cleaning

Forced-trigger (FT) events are the noise pool of `../../simulate.py --ft_noise_dir`. Some FT events contain continuous-wave or transient signals and are not thermal noise. The scripts in this directory find those events and write a clean mask for `simulate.py --ft_clean_mask`.

No masks are shipped: a mask lists events of one station and one data-taking period, and the repository does not track binary data files. All numbers below are for station 23 and the FT data of 2022 and will differ for other stations and years.

## Pipeline

```
extract_ft_rms.py       (slow: reads the ROOT files, saves the per-channel RMS as NPZ)
        |
        v
generate_clean_mask.py  (fast: applies the cut to the RMS NPZ, saves the mask NPZ)
validate_threshold.py   (fast: sweeps the cut value on the RMS NPZ, plots the convergence)
```

| File | Description |
|------|-------------|
| `extract_ft_rms.py` | Reads the `FORCE` triggers with `readRNOGDataMattak` (voltage calibration, median baseline correction) and stores the RMS of every channel and event |
| `generate_clean_mask.py` | Flags events with a high RMS on any channel and writes the mask |
| `validate_threshold.py` | Sweeps the cut and compares the RMS distributions after the cut with a thermal-noise reference |

## Usage

```bash
# 1. per-channel RMS of all FT events (run once per station)
python extract_ft_rms.py \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    --station_id 23

# 2. clean mask
python generate_clean_mask.py --rms_npz ft_rms_station23.npz

# optional: check the choice of the cut
python validate_threshold.py \
    --rms_npz ft_rms_station23.npz \
    --sim_nur /path/to/simulated_noise.nur \
    --output_dir figures/
```

## Method

For each of the 15 deep channels (0 to 11 and 21 to 23) the median and the median absolute deviation (MAD) of the RMS over all FT events are calculated, and the MAD is converted to a Gaussian-equivalent width, `sigma = 1.4826 * MAD`. An event is flagged if `(RMS - median) / sigma > 4` on any channel.

The mask is an NPZ file with one entry per FT event:

| Field | Type | Description |
|-------|------|-------------|
| `runNum` | int32 | Run number |
| `eventNum` | int32 | Event number |
| `is_clean` | int8 | 1 = clean (thermal), 0 = flagged |
| `station_id` | int32 | Station ID (scalar) |

## Choice of the cut

`validate_threshold.py` sweeps the cut from 2.5 to 8 sigma and calculates the excess kurtosis and the skewness of the RMS distributions of the events that pass. The cut is chosen where these match a thermal-noise reference. The reference values built into the script (kurtosis 0.28, skewness 0.22) come from 1000 simulated thermal-noise events for station 23, generated with per-channel effective temperatures and signal-chain responses of the 2023 season. That noise file is not part of the repository. `--sim_nur` calculates the reference from a NUR file of simulated noise instead.

Result for station 23, 2022 (718,979 FT events from runs 1 to 1135). The kurtosis is the largest value among the helper-string channels (9 to 11 and 21 to 23), which are the most contaminated ones:

| Cut | Events removed | Largest helper kurtosis | Matches the thermal reference |
|-----|----------------|-------------------------|-------------------------------|
| 2.5 sigma | 11.6% | 0.11 | no, over-cut (negative skewness) |
| 3.0 sigma | 3.7% | 0.17 | no, over-cut |
| 3.5 sigma | 1.4% | 0.23 | marginal |
| 4.0 sigma | 0.84% | 0.28 | yes |
| 4.5 sigma | 0.72% | 0.32 | no, residual tails |
| 5.0 sigma | 0.69% | 0.33 | no, residual tails |
| 8.0 sigma | 0.63% | 0.89 | no |
| no cut | 0% | 463 | no |

At 4 sigma, 6,066 of the 718,979 events (0.84%) are flagged. The channels with the most flagged events are 21 (4,469), 22 (4,366), 0 (3,640) and 1 (2,204). One event can be flagged on several channels.

## Known limitations

- The reference for the cut is simulated noise of the 2023 season, while the FT data is from 2022. If the noise changed between the seasons, the cut may need to be chosen again with a matching reference.
- The cut only uses the per-channel RMS. An event with non-thermal content that leaves the RMS unchanged is not flagged.
