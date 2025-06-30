# Fake News Detection using TensorFlow

This repository contains a Jupyter notebook demonstrating how to build a deep learning model to classify news articles as **real** or **fake**. The dataset used in this project is provided in `news.csv` and the complete workflow can be found in `FakeNewsProject.ipynb`.

## Project Structure

- `FakeNewsProject.ipynb` – main notebook with data preparation, model training and evaluation
- `news.csv` – dataset of news headlines and articles with labels indicating whether they are real or fake
- `environment.yml` – conda environment specification for running the notebook

## Requirements

The recommended way to run the notebook is inside a Conda environment. Create the environment and activate it with:

```bash
conda env create -f environment.yml
conda activate fake-news-env
```

Then start Jupyter:

```bash
jupyter notebook FakeNewsProject.ipynb
```

The notebook uses common Python packages including TensorFlow, pandas, NumPy and scikit-learn. All required packages are listed in `environment.yml`.

## Dataset

The dataset (`news.csv`) contains news articles with the following columns:

1. **id** – numerical identifier
2. **title** – article headline
3. **text** – article body text
4. **label** – `REAL` or `FAKE`

The notebook performs basic text preprocessing and trains a neural network on this data to predict the label of new articles.

## Results

The provided notebook demonstrates training a stacked Bidirectional LSTM network with early stopping. Using this architecture the model can achieve test accuracy above 90% on the supplied dataset.

Feel free to experiment with the notebook and modify the model or preprocessing steps to further improve performance.
