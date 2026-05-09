# Dataset notes: Chueh/Toyota fast-charging data

Target dataset page:

https://data.matr.io/1/projects/5d80e633f405260001c0b60a

Core facts from the project description provided by the user:

- Commercial A123 APR18650M1A LFP/graphite 18650 cells.
- Nominal capacity: 1.1 Ah. Nominal voltage: 3.3 V.
- Cells cycled under fast-charging conditions in a 30 °C forced-convection chamber.
- Protocols are four current steps often written as `CC1-CC2-CC3-CC4`, corresponding to six-step, 10-minute fast-charge procedures in the paper context.
- Upper/lower cutoff potentials: 3.6 V and 2.0 V.
- Discharge: 4C for all cells.
- Five batches of roughly 48 cells each.
- First four batches: CLO test, typically 100-120 cycles; first 100 cycles are used for early prediction.
- Final batch: validation batch cycled to failure or beyond.
- Temperature was collected only for the validation batch and is known to be imperfect.
- Internal resistance measurements were obtained during charging at 80% SOC by averaging pulses.
- Data are provided as MATLAB structs and raw CSVs; the MATLAB structs can be loaded in Python with `h5py` for HDF5-backed `.mat` files.
- Data are released under CC BY 4.0 according to the user-provided project description.

Important implementation caveats:

1. The MatR web app is JavaScript-based, so downloads may require manual browser interaction.
2. The loader in this prototype is best-effort. After the first raw batch is downloaded, inspect field names and harden `src/battery_aar/data/matr_io.py` against the exact struct layout.
3. Keep raw files out of git. Commit only code, tests, and small synthetic fixtures.
