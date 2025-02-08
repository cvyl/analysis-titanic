# Titanic Survival Prediction

This repository contains a Jupyter Notebook that explores the famous Titanic dataset from Kaggle. The goal is to analyze passenger data and build a machine learning model to predict survival outcomes.

## Overview
This project includes:
- **Data Exploration:** Understanding the dataset and its structure.
- **Exploratory Data Analysis (EDA):** Visualizing key trends and patterns.
- **Feature Engineering:** Creating new variables to improve model performance.
- **Model Building:** Training a basic Logistic Regression model for prediction.

## Dataset Description
The dataset includes various features about passengers such as age, sex, ticket class, and survival status.

**Key Columns:**
- `Survived`: Target variable (1 = Survived, 0 = Did not survive)
- `Pclass`: Passenger class (1st, 2nd, 3rd)
- `Sex`: Gender of passenger
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Ticket price
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Getting Started
### Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   ```
2. Navigate to the folder:
   ```bash
   cd YOUR_REPOSITORY
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook updated_analysis.ipynb
   ```

## Sample Code
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
train_data = pd.read_csv('train.csv')

# Survival count
sns.countplot(x='Survived', data=train_data)
plt.title('Survival Count')
plt.show()
```
I'm new to Jupyter Notebooks and data analysis, so this project is a learning experience for me. Any feedback or suggestions are welcome!

