# Project: Weather Prediction Models

## Overview
This project focuses on predicting weather conditions using various machine learning models. The models include Linear Regression, K-Nearest Neighbors (KNN), Decision Tree, Logistic Regression, and Support Vector Machine (SVM). The dataset used for this project is a weather dataset, processed and analyzed to predict future weather conditions.

In this project, The original source of the data is Australian Government's Bureau of Meteorology and the latest data can be gathered from [http://www.bom.gov.au/climate/dwo/](http://www.bom.gov.au/climate/dwo/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01).

Path to Download Data:

```https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'```
## Installation
To run this project, you need Python installed on your machine. Additionally, install the following libraries:
- Pandas
- Scikit-learn
- Numpy

You can install them using pip:
```bash
pip install pandas scikit-learn numpy
```

## Dataset
The dataset, `Weather_Data.csv`, contains various weather-related features. It is downloaded from a provided URL using the `requests` library. After downloading, the dataset undergoes preprocessing, including one-hot encoding of categorical variables and dropping unnecessary columns.

## Models
The project implements the following machine learning models:
1. **Linear Regression**: Predicts continuous values.
2. **K-Nearest Neighbors (KNN)**: A classification algorithm for categorizing data points based on their neighbors.
3. **Decision Tree**: A tree-like model for decision making.
4. **Logistic Regression**: Used for binary classification problems.
5. **Support Vector Machine (SVM)**: A classification algorithm for finding the optimal hyperplane which best separates the classes.

## Metrics
The models are evaluated based on various metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared (R2)
- Accuracy Score
- Jaccard Index
- F1-Score
- Log Loss

These metrics help in determining the performance and suitability of each model for the given dataset.

## Usage
1. **Data Preparation**: The script first downloads and processes the data.
2. **Model Training**: Each model is trained on the training dataset.
3. **Prediction**: The models make predictions on the test dataset.
4. **Evaluation**: Models are evaluated using the above metrics.

## Sample Output
The results of the model evaluations are as follows:

| Metric          | Linear Regression | KNN     | Tree    | Logistic Regression | SVM     |
|-----------------|-------------------|---------|---------|---------------------|---------|
| MAE             | 0.256315          | NaN     | NaN     | NaN                 | NaN     |
| MSE             | 0.115720          | NaN     | NaN     | NaN                 | NaN     |
| R2              | 0.427133          | NaN     | NaN     | NaN                 | NaN     |
| Accuracy Score  | NaN               | 0.818321| 0.818321| 0.827481            | 0.722137|
| Jaccard Index   | NaN               | 0.425121| 0.480349| 0.484018            | 0.722137|
| F1-Score        | NaN               | 0.596610| 0.648968| 0.652308            | 0.605622|
| Log Loss        | NaN               | NaN     | NaN     | 0.380085            | NaN     |

## Contribution
Contributions to this project are welcome. You can improve the models, add new ones, or enhance the data preprocessing steps. Please ensure to follow coding standards and comment your code where necessary.

## License
This project is open-source and available under the [MIT License](LICENSE.txt).

---


