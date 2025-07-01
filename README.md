# Fake News Detection using TensorFlow

This project documents the development of a high-performance deep learning model to classify news articles as "REAL" or "FAKE". This work builds upon and significantly improves the methodology presented in a baseline GeeksforGeeks article (https://www.geeksforgeeks.org/nlp/fake-news-detection-model-using-tensorflow-in-python/).

The primary goal was to leverage advanced NLP techniques to create a more robust and accurate classifier. The full development process, including data preprocessing, model training, and experimentation, is detailed in the accompanying Jupyter Notebook (`FakeNewsProject.ipynb`)

## Key Features & Improvements
- Advanced Model Architecture: This project implements a powerful stacked Bidirectional LSTM architecture, which proved more effective at capturing complex contextual relationships in text compared to the Conv1D + LSTM model from the baseline article.

- Superior Performance: Our final model achieved a test accuracy of over 91%, a substantial improvement over the ~76% validation accuracy from the original methodology.

- Pre-trained Embeddings: We successfully experimented with pre-trained GloVe embeddings, which yielded the highest accuracy of all tested models, demonstrating the power of transfer learning.

- Robust Training: The EarlyStopping callback was used to automate the training process, finding the optimal number of epochs and preventing model overfitting.

## Technology Stack
- Python 3.10

- TensorFlow & Keras

- Scikit-learn

- Pandas

- NumPy

- Jupyter Notebook / Google Colab

- Conda for local environment management

## How to Run This Project
This project can be run either in Google Colab (recommended) or on a local machine.

**Option 1: Running in Google Colab (Recommended)**
This is the easiest way to run the project as it requires no local installation.

1. Open in Colab: Navigate to the FakeNewsProject.ipynb file in the GitHub repository.

2. Click the "Open in Colab" button at the top of the file viewer.

3. Upload Data: In the Colab environment, use the file browser on the left to upload the news.csv dataset.

4. Run the Notebook: Execute the cells sequentially. The necessary libraries are pre-installed in the Colab environment.

**Option 2: Running Locally (with Conda)**

Prerequisites
You must have Miniconda or Anaconda installed.

**1. Clone the Repository**
```bash
git clone https://github.com/yyuan15/Fake-News-Detection-using-TensorFlow.git
cd Fake-News-Detection-using-TensorFlow
```
**2. Create the Conda Environment**

This repository includes an `environment.yml` file that specifies all the necessary dependencies. Create the Conda environment using the following command:
```bash
conda env create -f environment.yml
```

Once the environment is created, activate it:
```bash
conda activate fakenews_env
```

**3. Launch Jupyter Notebook**
With the environment active, launch Jupyter Notebook:
```bash
jupyter notebook
```

Open the `FakeNewsProject.ipynb` file and run the cells sequentially to execute the project.


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
