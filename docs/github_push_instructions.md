# Pushing this repository

1. Create the repo on GitHub (suggested name: `wellbore-tvt-prediction`), public, NO auto-README.
2. Locally:
   ```
   cd wellbore-tvt-prediction
   git init && git add . && git commit -m "ROGII Wellbore TVT prediction — Silver medal solution (241/6125)"
   git branch -M main
   git remote add origin https://github.com/<your-username>/wellbore-tvt-prediction.git
   git push -u origin main
   ```
3. On the repo page: add topics (`kaggle`, `geosteering`, `time-series`, `hidden-markov-models`, `geoscience`, `machine-learning`), set the description to
   "Silver-medal (top 4% of 6,125) solution for blind-zone TVT prediction in horizontal wells — CPU-only structural tracker + deployment-side dose calibration", and pin it on your profile.
4. Before first push, confirm `neural/rogii-seq-unet-gpu-v5.ipynb` is your own notebook; remove it if not.
5. After arXiv posting: add the arXiv badge/link to README, and link the repo from your Kaggle profile, LinkedIn, and the papers.
