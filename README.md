# Sparse + Low‑Rank Engine

Implements Robust PCA (Candès et al., 2011) to decompose the ETF correlation matrix into a low‑rank component (systematic market structure) and a sparse component (idiosyncratic shocks). The score for each ETF is the magnitude of its row in the sparse component – a measure of how much it is affected by unique, uncorrelated events. Multi‑window evaluation selects the best window per ETF.

- **Decomposition:** low‑rank + sparse via inexact ALM
- **Score:** sum of absolute values of sparse matrix row
- **Windows:** 63, 252, 504, 1008, 2016, 4032 days (best per ETF)
- **Output:** top 3 ETFs per universe

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
