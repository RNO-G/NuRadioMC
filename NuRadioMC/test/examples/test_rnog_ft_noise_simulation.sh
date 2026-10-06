#!/bin/bash
set -e

python3 NuRadioMC/test/examples/test_rnog_ft_noise_helpers.py

cd NuRadioMC/examples/08_RNO_G_trigger_simulation

# Thermal-noise mode with the shallow fiducial volume. The event generator of this example
# is not seeded, so only the bookkeeping is checked: one ledger row per thrown event.
python3 simulate.py --station_id 11 -e 1e19 -n 50 --fiducial_rmax 200 --nur_output \
    --data_dir ci_test_data --output_file test.hdf5

python3 - <<'EOF'
import sys
import pandas as pd

ledger = pd.read_csv("ci_test_data/test_ledger.csv")
print(ledger["status"].value_counts().to_string())
if len(ledger) != 50:
    sys.exit(f"Expected 50 ledger rows, found {len(ledger)}.")
if not ledger["status"].isin(["triggered", "trigger_failed", "efield_cut"]).all():
    sys.exit("Unexpected status in the ledger.")
EOF

rm -rf ci_test_data
