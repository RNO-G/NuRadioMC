# Trigger-path noise Vrms

Forced-trigger (FT) noise is recorded through the readout signal chain (RADIANT). The trigger path (FLOWER) has a different signal chain after the 3 dB splitter (arXiv:2411.12922, Sec. 3.2), so the noise Vrms seen by the trigger differs from the Vrms of the recorded traces. `../../simulate.py` needs the trigger-path Vrms in the measured-noise mode (`--trigger_vrms`): `triggerBoardResponse` uses it to select the VGA gain and digitize the trace, and the trigger threshold is 3.76 times this Vrms.

## Files

| File | Description |
|------|-------------|
| `measure_trigger_vrms_full.py` | Measures the trigger-path Vrms over the full FT pool of a station |
| `trigger_vrms_station{12,13,21,22,23,24}.yaml` | Values for the 2022 season with the signal-chain responses of the detector database |
| `trigger_vrms_station{13,23}_calibrated.yaml` | Values for the 2022 season with the calibrated readout response |

## Measurement

For every FT event and each phased-array channel (0 to 3), `measure_trigger_vrms_full.py`

1. takes the recorded trace (3.2 GHz, median baseline corrected),
2. upsamples it to the 5 GHz internal sampling rate of the simulation,
3. multiplies its spectrum by `trigger_response / readout_response` from the detector description, the same transfer the simulation applies to the injected noise,
4. takes the standard deviation.

The value per channel is the median over all events. The script prints it as a `trigger_vrms_V` block, the part of the YAML file that `simulate.py` reads, and writes the per-event values and the per-run means to `trigger_vrms_station{id}_{mode}.npz`.

```bash
python measure_trigger_vrms_full.py \
    --station 23 \
    --ft_noise_dir /path/to/forced_triggers/station23 \
    --clean_mask /path/to/clean_mask_station23.npz \
    --n_jobs 20
```

Without `--detector_file` the detector description is read from the MongoDB at `--event_time` (default 2022-10-01). `--mode burn --burn_root <dir>` processes all triggers of a directory of run folders instead of the forced triggers of an FT pool. The script needs `mattak` and `joblib`.

## Shipped values

The YAML files hold the values that the 2022 simulations were run with. Each file records in its `metadata` block how the values were obtained.

- Stations 12, 13, 21, 22 and 24: full-pool measurement with the database responses.
- Station 23: a sampled measurement of 200 tiled noise realizations, made with an earlier script. The full-pool measurement gives 4.30 / 5.07 / 4.16 / 2.98 mV for channels 0 to 3, 5 to 13 percent above the shipped values. The simulations kept the earlier values.
- `*_calibrated.yaml` (stations 13 and 23): full-pool measurement with the calibrated season-2022 readout response in the transfer and the database trigger response. Use them only together with a detector description that carries that readout response.

A new measurement depends on the FT pool, the clean mask and the detector description, so it will not reproduce these values to the last digit.

## Known limitations

- The transfer function comes from the detector description. If the description of the readout or trigger chain is inaccurate, so is the Vrms.
- The real FLOWER board selects its VGA gain so that the noise fills about 5 ADC counts of its 8-bit digitizer. `triggerBoardResponse` does the same, but for the same input Vrms it selects a lower gain stage than the gain codes recorded by the real board. The cause is not known. The ADC-count threshold of the simulation therefore does not correspond to exactly the same voltage as in the hardware.
- The values are specific to the station, the detector description and the FT data. Measure them again when one of these changes.
