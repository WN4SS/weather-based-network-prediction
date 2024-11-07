# A Machine Learning Framework for Weather-Based Signal Strength Prediction in Private LTE/5G Networks
[[arXiv]()] [[Project Page]()] [[Bibtex]()]
## Installing Dependencies
We recommend using python venv to manage dependencies
```zsh
git clone --depth 1 https://github.com/UB-IoT-Lab/UB-IoT-Lab-CBRS-SignalStrength-ML.git
python -m cbrs_venv python=3.12
source cbrs_venv/bin/activate
pip install -r requirements.txt
```

## Dataset Preparation
Please see [docs/dataset.md](docs/dataset.md) for instructions on preprocessing the KPI and weather datasets.

# Training
Once the datasets are setup, run the following command to train using the optimal feature combination
~~~zsh
python3 experiment.py --version 7
~~~
# Evaluation

## Citing the paper
If you find this code helpful, please cite:



