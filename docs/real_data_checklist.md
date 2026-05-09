# Real-data checklist

After downloading the public MatR batch files:

1. Place raw files under `data/raw/chueh_toyota_fast_charge/`.
2. Start with a small parse:

   ```bash
   python scripts/preprocess_chueh_toyota.py \
     --raw-dir data/raw/chueh_toyota_fast_charge \
     --out data/processed/chueh_toyota_fast_charge_smoke \
     --max-cells-per-batch 2 \
     --first-n-cycles 100
   ```

3. Inspect `data/processed/chueh_toyota_fast_charge_smoke/qc_summary.json`.
4. Confirm that metadata has real `cycle_life` labels and protocol columns.
5. Confirm that cycle summary contains at least `cycle_index` and `discharge_capacity`.
6. Only then process the full batches.
7. Do not commit raw data, processed data, checkpoints, or run outputs.

Known risk: MatR `.mat` structs can vary in their field layout. If preprocessing fails, inspect the HDF5 field names and harden `src/battery_aar/data/matr_io.py` while adding a loader test that captures the discovered structure.
