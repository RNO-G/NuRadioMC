# Production workflow

Snakemake workflow that runs `../simulate.py` in the measured-noise mode at scale. It throws chunks of events until each energy bin reaches a target number of triggered events and then writes, for each bin, a manifest of the chunks needed to reach that target. One chunk is one SLURM job and produces a NUR file, an HDF5 file and a per-event ledger CSV.

All site-specific values are in `config/config.yaml`. The Snakefile and the scripts contain no absolute paths.

## Requirements

- `snakemake` version 8 or later with the SLURM executor plugin (`snakemake-executor-plugin-slurm`). Neither is a NuRadioMC dependency.
- In the environment of the jobs: NuRadioMC, `mattak`, `pandas` and, for `--proposal`, `proposal`.
- The inputs of the measured-noise mode listed in [`../README.md`](../README.md): the FT run files of the station, a clean mask, the trigger-path Vrms and the ADC clip thresholds.

## Layout

```
production/
  Snakefile                     # rules: throw_chunk, truncate_lgE, register_dataset
  config/
    config.yaml.example         # template, copy to config.yaml
    config_station13.yaml       # settings of the 2022 station 13 simulation
    config_station23.yaml       # settings of the 2022 station 23 simulation
    profile/config.yaml         # snakemake profile (slurm executor, jobs, retries)
  scripts/
    truncate_to_target.py       # picks the chunks that reach the target, writes manifest.txt
    write_readme.py             # writes the dataset README into data_dir
  workflow_logs/                # per-chunk logs (created at run time)
```

Output in `data_dir`:

```
<data_dir>/
  lgE16.0/
    lgE16.0_c0000.nur           # waveforms
    lgE16.0_c0000.hdf5          # NuRadioMC output
    lgE16.0_c0000_ledger.csv    # one row per thrown event (status column)
    ...
    manifest.txt                # chunks kept for the target-trigger sample
  lgE16.5/ ...
  README.md                     # dataset summary, written by the workflow
```

## Config keys

Relative paths are relative to this directory, from which `snakemake` is started.

| Key | Meaning |
|-----|---------|
| `sim_script` | path to `simulate.py` |
| `sim_config` | path to the NuRadioMC YAML config (`RNO_config.yaml`) |
| `python_bin` | interpreter that runs the simulation (default `python3`) |
| `pythonpath` | prepended to `PYTHONPATH`, so that the simulation imports this checkout; empty to use the installed NuRadioMC |
| `env_setup` | shell command run in every chunk job before the simulation, for example to activate an environment; empty if the submitting environment is already complete |
| `data_dir` | output directory for all bins |
| `station_id` | RNO-G station ID |
| `detector_file` | detector description file; an empty string queries the MongoDB |
| `event_time` | detector time |
| `ft_noise_dir` | directory of the forced-trigger run files |
| `ft_clean_mask` | clean mask NPZ |
| `trigger_vrms` | YAML with the trigger-path Vrms of the station |
| `clip_thresholds` | YAML with the per-channel ADC clip bounds; empty uses the uniform `pedestal_voltage` clip |
| `pedestal_voltage` | ADC pedestal in volts for the uniform clip |
| `fiducial_rmax` | maximum radius of the fiducial volume in m; empty uses the volume of `sim_config` |
| `flavor` | neutrino flavor (`e`, `mu`, `tau`, `all`) |
| `interaction_type` | `cc`, `nc` or `ccnc` |
| `ch0_trigger_model` | channel 0 trigger model of `simulate.py` (default `normal`) |
| `ft_seed_base` | added to the chunk ID to get the FT seed of a chunk |
| `target_triggers_per_bin` | triggered events to reach per energy bin |
| `target_per_bin` | optional per-bin targets that replace `target_triggers_per_bin` |
| `safety_margin` | factor on the number of thrown events estimated from the trigger rate |
| `finalize_only` | if true, no chunk is thrown and the manifests are built from the ledgers on disk |
| `energies` | list of lgE bin labels; the simulation gets `10^lgE` eV |
| `trigger_rates` | estimated trigger probability per thrown event per bin; sets the number of chunks |
| `thrown_per_chunk` | events thrown per chunk per bin |
| `slurm_resources` | `mem_mb` and `runtime_min` per bin for the first attempt; retries get more |
| `accounts` | list of `{account, partition, weight}` among which the chunks are distributed |

The number of chunks of a bin is `ceil(target / trigger_rate * safety_margin / thrown_per_chunk)`.

## Use

```bash
cd production
cp config/config.yaml.example config/config.yaml
# edit config/config.yaml
snakemake -n                    # dry-run, prints the plan
```

One chunk as a pilot, to check the job resources and the submission:

```bash
snakemake --executor slurm --jobs 1 --workflow-profile config/profile \
  <data_dir>/lgE18.5/lgE18.5_c0000_ledger.csv
```

Full production. Start it in a terminal multiplexer such as `tmux`, since the `snakemake` process has to stay alive until the last job is done:

```bash
snakemake --executor slurm --jobs 200 --workflow-profile config/profile
```

To resume, run the same command again. Chunks with partial outputs are thrown again. To raise the target, edit `target_triggers_per_bin` and run again: only the missing chunks are thrown and the manifests are rewritten.

If chunks were lost, for example to preemption, and should not be thrown again, set `finalize_only: true` and run again. The manifests are then built from the ledgers on disk, and the chunk IDs may have gaps.

The `trigger_rates`, `thrown_per_chunk` and `slurm_resources` in the example config are estimates. Measure them with a pilot for your station and settings before a large production. The trigger rate falls steeply towards the lowest energy bin, so an extrapolation from higher energies is unreliable there.

## Account routing

A chunk is assigned to an account by `chunk_id % 10`, in proportion to the weights in `accounts` (weights of 100 and 100 give an even split). Edit the list to match the allocations and partitions of your cluster. The defaults of the profile in `config/profile/config.yaml` are placeholders as well.

## Analysis sample

Chunks beyond the target stay on disk. The analysis sample of a bin is the list of NUR files in its `manifest.txt`.

## Settings of the 2022 simulations

`config/config_station13.yaml` and `config/config_station23.yaml` hold the settings of the 2022 cosmic-ray proxy simulations for stations 13 and 23: electron-neutrino neutral-current interactions in a disc of 200 m radius in the top metre of the ice, the calibrated readout response, and for station 13 the `measured_8x` channel 0 trigger model. The `/path/to` entries have to be filled in. The detector description with the calibrated readout response that these simulations used is not in the detector database.
