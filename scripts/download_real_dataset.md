# Getting the real dataset (Kaggle)

The video uses the Kaggle dataset titled **"CT Scans for COVID-19 Classification"**
(folders `nCT`, `pCT`, `NiCT`). Totals seen on screen: **nCT 9,979 · NiCT 5,705 · pCT 4,001**.

1. Create a Kaggle API token (Kaggle -> Settings -> API) and place it at `~/.kaggle/kaggle.json`.
2. Find the exact slug (search the title on kaggle.com) and download:
   ```bash
   kaggle datasets download -d <owner>/<slug> -p data/real --unzip
   ```
3. Make sure `data/real/` directly contains `nCT/ pCT/ NiCT/` (flatten nested folders if needed).
4. Verify the counts above, then `make train DATA=data/real EPOCHS=10`.

The dataset licence/terms must be checked on the Kaggle page before any redistribution.
