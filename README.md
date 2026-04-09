# Changepoint-Detection-in-the-Presence-of-Outliers

This repository contains a Machine Learning for Time Series project as part of the MVA Master. The project focuses on the R-FPOP algorithm, demonstrating how bounded loss functions can accurately detect structural changes in time series data while remaining robust to extreme outliers.

### Reference Paper
Fearnhead, P., & Rigaill, G. (2019). Changepoint Detection in the Presence of Outliers. Journal of the American Statistical Association, 114(525), 169–183.

### Implemented Methods and experiments
The repository implements the dynamic programming algorithm described in the paper with three distinct cost functions (L2 loss, Huber loss, Biweight loss) to compare sensitivity and robustness. 

The analysis is performed on two types of datasets:
- Simulated Scenarios: Reproduction of the six benchmark scenarios described in the article (varying noise levels, Student-t noise, short segments) to validate the theoretical properties of the Biweight loss.
- Real-world Economic Indicators: Application of the algorithms to financial time series from the FRED database, including: inflation expectations, GDP growth rates (Japan, UK, Germany), market volatility and credit spreads. 

### Results
The results of our experiments are available in the "final_notebook.ipynb" notebook. A detailed analysis is also available in the "Report of the project.pdf" file.

### Requirements
Python 3.x

numpy

pandas

matplotlib

statsmodels 

## Production-Oriented Package

The repository now includes a minimal package in `src/rfpop` that keeps only the parts identified as production-relevant:

- Core RFPOP algorithm and robust loss builders
- Parameter selection with BIC-based search and elbow-based search
- Segment plotting utility for detected changepoints

Cross-validation has intentionally not been included in this production path because it was unstable during project experiments.

### Project Structure

- `src/rfpop/core.py`: core RFPOP dynamic programming implementation
- `src/rfpop/tuning.py`: penalty and K calibration, BIC and elbow parameter selection
- `src/rfpop/plotting.py`: plotting utilities for changepoints and segment means
- `tests/`: smoke tests for core execution and tuning flow

### Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Example usage:

```python
import numpy as np
from rfpop import plot_segments, select_params_elbow

y = np.array([0.1, 0.2, 0.2, 4.9, 5.0, 5.1, 0.3, 0.2])

search = select_params_elbow(y, loss="biweight")
best = search["best"]

fig, ax, cps = plot_segments(
	y,
	loss="biweight",
	beta=best["beta"],
	k_value=best["k"],
)
```

