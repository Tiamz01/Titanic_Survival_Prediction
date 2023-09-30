# Titanic Survival Prediction

This repository contains a Python Jupyter Notebook that demonstrates the process of predicting passenger survival on the Titanic using machine learning. The notebook covers various stages of the data science pipeline, including data loading, data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation.

## Dependencies

Before running the notebook, make sure you have the following libraries installed in your Python environment:

- pandas
- matplotlib
- seaborn
- numpy
- scikit-learn
- xgboost

You can install these libraries using pip:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn xgboost
```

## Data

The dataset used for this project consists of two CSV files:

- `train.csv`: Contains training data with passenger information and survival labels.
- `test.csv`: Contains test data for which you need to predict survival.

## Notebook Contents

1. **Importing Libraries**: The necessary Python libraries are imported to kickstart the project.

2. **Reading the Dataset**: The training and test datasets are loaded into Pandas DataFrames.

3. **Data Validation and Preprocessing**: This section includes data validation and preprocessing steps:
   - Handling missing values in columns like "Age," "Cabin," and "Embarked."
   - Data imputation for missing age values and cabin values.
   - Data encoding for categorical columns.

4. **Exploratory Data Analysis (EDA)**: Exploration of the dataset to gain insights:
   - Demographic analysis of passengers, including gender distribution and passenger class.
   - Survival analysis by passenger class and gender.
   - Analysis of family sizes on board.

5. **Feature Engineering**: Creating new features based on available data.

6. **Predicting Survival**: Using a Decision Tree Classifier to predict passenger survival.

7. **Model Evaluation**: Evaluating the model's performance with metrics like accuracy and cross-validation.

8. **Scaling Features**: Scaling features using StandardScaler and re-evaluating the model.

9. **Verifying Model Performance**: Comparing model performance before and after scaling features.

10. **Submission**: Generating the final submission file for the Kaggle competition.

## Results and Visualization

To give you a glimpse of the exploratory data analysis (EDA) conducted in this project, here is a visualization of the demographic distribution of passengers on board:

![Demographic Distribution](images/demographic_distribution.png)

This visualization illustrates the gender distribution among passengers, providing valuable insights into the composition of the dataset.

Feel free to explore the notebook for a comprehensive understanding of the Titanic survival prediction project.

## How to Use the Notebook

You can follow the notebook step by step to understand the process of data preprocessing, EDA, feature engineering, and model building for the Titanic dataset. Feel free to modify and experiment with the code to enhance the model's performance or apply different algorithms.

To run the notebook, make sure you have the required dependencies installed and adjust the file paths for reading the dataset if necessary.
