# A Machine Learning Framework for Weather-Based Signal Strength Prediction in Private LTE/5G Networks
[[arXiv](https://www.techrxiv.org/users/692934/articles/1239831-a-machine-learning-framework-for-weather-based-signal-strength-prediction-in-private-lte-5g-networks)] [[Project Page]()] [[Bibtex]()]
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

> Devasenapathy, K., Caezza, J. A., & Malandra, F. (2024). A Machine Learning Framework for Weather-Based Signal Strength Prediction in Private LTE/5G Networks. Authorea Preprints.
> [arXiv](https://www.techrxiv.org/users/692934/articles/1239831-a-machine-learning-framework-for-weather-based-signal-strength-prediction-in-private-lte-5g-networks)

BibTeX entry:
```bibtex
@article{your_paper_key,
  author    = {Your Name and Co-authors},
  title     = {Title of the Paper},
  journal   = {Journal/Conference Name},
  year      = {Year},
  volume    = {Volume},
  number    = {Number},
  pages     = {Pages},
  doi       = {DOI link}
}