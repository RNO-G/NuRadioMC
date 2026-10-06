# RNO-G trigger simulation with measured forced-trigger noise

`simulate.py` runs a NuRadioMC simulation of one RNO-G station with the FLOWER trigger model. It extends `../RNO_G_trigger_simulation/simulate.py` with three things:

1. Measured noise. Recorded forced-trigger (FT) waveforms can replace generated thermal noise (`--ft_noise_dir`).
2. Asymmetric ADC saturation. The readout traces are clipped at the bounds set by the off-center pedestal of the RADIANT ADC (`--clip_thresholds` or `--pedestal_voltage`).
3. A per-event ledger. Every thrown event gets one row in a CSV file, whether it triggered or not.

## Requirements

The detector description comes from the RNO-G MongoDB at `--event_time`, or from a file given with `--detector_file` or the `RNOG_DETECTOR_FILE` environment variable. The ledger is written with `pandas`, which is not a NuRadioMC dependency.

The measured-noise mode also needs:

- `mattak`, to read RNO-G ROOT files.
- A directory of full-waveform run files named `station{id}_run*.root` for the simulated station. The script selects the `FORCE` triggers itself.
- A YAML file with the trigger-path noise Vrms of the four phased-array channels (`--trigger_vrms`).
- Optionally an NPZ mask of FT events to skip (`--ft_clean_mask`) and a YAML file with per-channel ADC clip bounds (`--clip_thresholds`).

The formats are described under [input files](#input-files).

## Noise modes

### Thermal noise (default)

Without `--ft_noise_dir`, noise is generated with `channelGenericNoiseAdder` from `trigger.noise_temperature` in the config, and only if the config has `noise: True`. The shipped `RNO_config.yaml` has `noise: False`, so pass your own config to get thermal noise. `--noise_temperatures` replaces the noise temperature of the detector description per channel.

### Measured FT noise (`--ft_noise_dir`)

FT waveforms are recorded through the readout signal chain (RADIANT, 3.2 GHz). The trigger path (FLOWER) has a different signal chain after the 3 dB splitter (arXiv:2411.12922, Sec. 3.2), so the noise is injected separately for the two paths:

1. Trigger path, at the 5 GHz internal sampling rate. The internal trace is longer than one FT event, so several FT events are upsampled from 3.2 to 5 GHz and stitched into one noise trace (`tile_noise_overlap_add`). Neighbouring events overlap by `TILE_OVERLAP` samples with an equal-power crossfade: the weights are `sqrt(ramp)` and `sqrt(1 - ramp)` with a Hann ramp, so the variance of independent noise stays constant through the seam. The stitched trace is multiplied by `trigger_response / readout_response` from the detector description and added to the trigger copies of channels 0 to 3.
2. Readout path, at 3.2 GHz. A separate FT event is added to the readout channels after the readout-window cut and the resampling to the detector sampling rate. No transfer function is needed here.

The two paths draw different FT events from the same pool. The pool reads one run file at a time, skips events flagged in the clean mask and files that cannot be read, and cycles through the file list, so a pool smaller than the number of thrown events reuses noise events.

In this mode the readout window is cut with `zero_padded_readout_window_cutter` instead of the framework cutter: where the window extends past the internal trace, the missing samples are zeros instead of a cyclic wrap of the trace. The readout FT event then covers the full 2048-sample trace.

The script sets `noise: False` in this mode so that no thermal noise is added on top.

## Trigger model

`triggerBoardResponse` (VGA gain and 8-bit ADC) followed by `highLowThreshold`:

- threshold of 3.76 times the trigger-path noise Vrms, the value for a 1 Hz noise trigger rate (`RNO_G_HighLow_Thresh`)
- 2 of 4 coincidence between the phased-array channels 0 to 3
- high-low window of 6 samples and coincidence window of 20 samples at the FLOWER sampling rate

In the measured-noise mode the Vrms comes from the `--trigger_vrms` file. In the thermal mode it is calculated from the noise temperature and the trigger signal chain.

`--ch0_trigger_model` changes the treatment of channel 0 for station 13, whose trigger path for this channel is suppressed. `measured_8x` scales the digitized channel 0 trace by 1/8 and uses an absolute threshold of 4 ADC counts. `measured_dead` removes channel 0 from the coincidence. `normal` (default) treats it like the other channels.

## ADC saturation

The RADIANT ADC covers 0 to 2.5 V and the pedestal sits near 1.5 V, so a pedestal-subtracted trace saturates at about -1500 mV and +1000 mV. `--clip_thresholds` gives the bounds per channel. Without it, `--pedestal_voltage` (default 1.5 V) sets the same bounds for all channels.

The clip and the readout FT noise are applied when the traces are resampled for the NUR file, so they need `--nur_output`. Quantities stored in the HDF5 file are calculated before that step.

## Input files

| Argument | Format |
|----------|--------|
| `--trigger_vrms` | YAML with `trigger_vrms_V: {0: v0, 1: v1, 2: v2, 3: v3}` in volts and optionally `metadata: {station_id: ...}` |
| `--ft_clean_mask` | NPZ with arrays `runNum`, `eventNum` and `is_clean` (1 keeps the event, 0 skips it) |
| `--clip_thresholds` | YAML with `clip_thresholds_mV: {channel: [low, high]}` in mV, relative to the pedestal |
| `--noise_temperatures` | JSON with `{channel: temperature}` in kelvin |

The tools that produce these files are in this folder:

- [`noise_analysis/ft_cleaning/`](noise_analysis/ft_cleaning/): clean mask of the FT pool.
- [`noise_analysis/trigger_vrms/`](noise_analysis/trigger_vrms/): trigger-path noise Vrms, with the values of the 2022 season.
- [`pedestal_extraction/`](pedestal_extraction/): ADC clip thresholds from pedestal runs, with the values of the 2022 season.

The shipped values belong to the detector description of 2022-10-01 and the FT data of 2022. Other years need a new measurement.

## Usage

Measured FT noise:

```bash
python simulate.py \
    --station_id 23 \
    --energy 1e18 \
    --n_events 1000 \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    --trigger_vrms noise_analysis/trigger_vrms/trigger_vrms_station23.yaml \
    --ft_clean_mask /path/to/clean_mask_station23.npz \
    --clip_thresholds pedestal_extraction/clip_thresholds_station23.yaml \
    --ft_seed 12345 \
    --nur_output \
    --output_file output.hdf5 \
    --data_dir /path/to/output
```

Thermal noise:

```bash
python simulate.py \
    --config /path/to/config_with_noise.yaml \
    --station_id 23 \
    --energy 1e18 \
    --n_events 1000 \
    --output_file output.hdf5 \
    --data_dir /path/to/output
```

For parallel runs give every job its own `--index` and `--output_file`. The index offsets the event ids.

[`production/`](production/) holds a Snakemake workflow that runs the measured-noise mode on a SLURM cluster, one chunk of events per job, until every energy bin has a target number of triggered events.

`python simulate.py --help` lists all arguments.

## Fiducial volume

The volume and the zenith range come from an optional `fiducial_volume` section of the config (see `RNO_config.yaml`) and can be overridden with `--fiducial_rmax`, `--min_zenith` and `--max_zenith`. The default zenith range is 0 to 60 degrees.

- With `rmax` and `zmin` set, the cylinder from the config is used.
- With only a maximum radius (for example `--fiducial_rmax 200` with the shipped config), vertices are thrown in a disc of that radius in the top metre of the ice. This is the volume of the cosmic-ray proxy simulations.
- With neither, the energy-dependent neutrino volume of `../RNO_G_trigger_simulation/simulate.py` is used.

## Output

- HDF5: the standard NuRadioMC output for the triggered events.
- NUR (with `--nur_output`): the events with readout traces.
- Ledger CSV (`<output>_ledger.csv`): one row per thrown event with `event_group_id`, `zenith_deg`, `azimuth_deg`, `energy_eV`, `flavor`, `status` and the maximum amplitude of channels 0 to 3 in mV. The status is `triggered`, `trigger_failed` (reached the trigger stage) or `efield_cut` (dropped before it).

## Known limitations

- The internal sampling rate of 5 GHz and the FT event length of 2048 samples at 3.2 GHz are written into the script.
- The seed of the event generator is drawn at random in every run. `--ft_seed` only fixes the order of the FT files.
- The VGA gain that `triggerBoardResponse` selects does not match the gain codes of the real FLOWER board for the same input Vrms, so the ADC-count threshold does not correspond to exactly the same voltage as in the hardware.
- The detector database does not store pedestal voltages, so the ADC bounds have to be passed on the command line.
