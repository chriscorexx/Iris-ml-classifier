# 🌸 Iris ML Classifier

A beginner-friendly Machine Learning classification project that predicts the species of an Iris flower using **Logistic Regression** and the classic Iris dataset.

## 📌 Overview

This project demonstrates a complete basic Machine Learning workflow:

**Dataset → Train/Test Split → Model Training → Prediction → Evaluation → Model Saving**

The model is trained to classify Iris flowers into three species:

* Setosa
* Versicolor
* Virginica

## 🧠 Machine Learning Model

**Algorithm:** Logistic Regression

The project uses the Iris dataset available through `scikit-learn`.

### Features

The model uses four measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Target

The target is the Iris flower species.

## ⚙️ Tech Stack

* Python
* NumPy
* Scikit-learn
* Pickle

## 📂 Project Structure

```text
Iris-ml-classifier/
├── iris_classification.py
├── iris_model.pkl
├── .gitignore
└── README.md
```

### Files

| File                     | Description                                                   |
| ------------------------ | ------------------------------------------------------------- |
| `iris_classification.py` | Loads the dataset, trains the model and evaluates predictions |
| `iris_model.pkl`         | Saved trained Logistic Regression model                       |
| `.gitignore`             | Specifies files ignored by Git                                |
| `README.md`              | Project documentation                                         |

## 🔄 ML Workflow

### 1. Load Dataset

The Iris dataset is loaded using `load_iris()` from Scikit-learn.

### 2. Split the Data

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

A fixed `random_state` is used to make the split reproducible.

### 3. Train the Model

A Logistic Regression classifier is trained using the training data.

### 4. Make Predictions

The trained model predicts the species of flowers in the testing dataset.

### 5. Evaluate the Model

The predictions are evaluated using classification accuracy.

### 6. Save the Model

The trained model is saved as:

```text
iris_model.pkl
```

This allows the trained model to be reused without training it again.

## 📊 Result

The current model achieved:

```text
Accuracy: 100.00%
```

on the test split used in the project.

> Note: Accuracy can vary depending on the train/test split and model configuration.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/chriscorexx/Iris-ml-classifier.git
cd Iris-ml-classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

Install the required library:

```bash
pip install scikit-learn
```

## ▶️ Run the Project

```bash
python iris_classification.py
```

## 🎯 What I Learned

Through this project, I practiced:

* Loading datasets with Scikit-learn
* Understanding features and targets
* Train/test splitting
* Logistic Regression
* Model training and prediction
* Model evaluation
* Saving trained ML models
* Building a basic end-to-end ML workflow

## 🔮 Future Improvements

Possible improvements include:

* Add a confusion matrix
* Add a classification report
* Compare multiple classification algorithms
* Add data visualization
* Build a prediction interface
* Create a REST API using FastAPI
* Deploy the model as a web application

## 👨‍💻 Author

**Chris D'Souza**

Computer Engineering Student
Interested in Machine Learning, Software Development, AI and IoT.
