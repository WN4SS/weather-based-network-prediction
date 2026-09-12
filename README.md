# A Weather-based Framework to Predict Signal Strength in Cellular Networks using ML
[[arXiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.173198458.82259958/v2)] [[Project Page]()] [[Bibtex]()]
## Installing Dependencies
We recommend using python venv to manage dependencies
```zsh
git clone --depth 1 https://github.com/UB-IoT-Lab/UB-IoT-Lab-CBRS-SignalStrength-ML.git
python -m cbrs_venv python=3.12
source cbrs_venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation
Download the dataset from [Link](https://github.com/UB-IoT-Lab/CBRSdata). Please see [docs/dataset.md](docs/dataset.md) for instructions on replicating the preprocessing on KPI and weather datasets.

# Training
Once the datasets are setup, run the following command to train using the optimal feature combination
~~~zsh
python3 experiment.py --version 7
~~~
## Citing the paper
If you find this code useful in your research, please consider citing our paper:

> Devasenapathy, K., Dash, B. K., & Malandra, F. (2025). A Weather-based Framework to Predict Signal Strength in Cellular Networks using ML. Authorea Preprints.
> [arXiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.173198458.82259958/v2)

BibTeX entry:
```bibtex
@article{devasenapathy2025weather,
  title={A Weather-based Framework to Predict Signal Strength in Cellular Networks using ML},
  author={Devasenapathy, Kishorkumar and Dash, Biswajit Kumar and Malandra, Filippo},
  year={2025},
  publisher={TechRxiv}
}
