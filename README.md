<img src = "https://github.com/KhangKuro/Credit-Card-Fraud-Detection/blob/main/CCF.png" />

# Credit Card Fraud Detection 🛡️

## Overview ℹ️

Credit card fraud is a significant concern for financial institutions and cardholders alike. This project aims to develop an effective fraud detection system using machine learning algorithms. After analyzing fraudulent transaction clusters, various models are trained and evaluated to identify the most suitable approach for fraud detection.

<img src = "https://github.com/KhangKuro/Credit-Card-Fraud-Detection/blob/main/Idea.png" />

## Model Development 🚀

<img src = "https://github.com/KhangKuro/Credit-Card-Fraud-Detection/blob/main/Model.png" />

### Data Preprocessing 🛠️

The dataset undergoes preprocessing steps similar to the analysis phase, including cleaning, feature engineering, and normalization.

<img src = "https://github.com/KhangKuro/Credit-Card-Fraud-Detection/blob/main/erd.png" />

### Handling Class Imbalance ⚖️

To address class imbalance in the dataset, several techniques are applied:

- **SMOTE (Synthetic Minority Over-sampling Technique):** Generates synthetic samples for the minority class to balance the dataset.
- **Random Over Sampling:** Randomly duplicates samples from the minority class to balance the dataset.
- **Random Under Sampling:** Randomly removes samples from the majority class to balance the dataset.

### Model Training and Evaluation 📊

The following machine learning models are trained and evaluated:

1. **Random Forest**
2. **XGBoost**
3. **MLPClassifier**
4. **Logistic Regression**
5. **Decision Tree**

### Model Performance 📈

The performance of each model is evaluated based on the following metrics:

- **Training Score:** Accuracy score on the training dataset.
- **Testing Score:** Accuracy score on the testing dataset.
- **Accuracy:** Overall accuracy of the model.
- **F1 Score:** Harmonic mean of precision and recall.
- **Precision:** Proportion of true positive predictions out of all positive predictions.
- **Recall:** Proportion of true positive predictions out of all actual positives.

## Results 🏆

The table below summarizes the training and testing scores, as well as the performance metrics for each model:

| Model Name                           | Training Score | Testing Score | Accuracy | F1 Score | Precision | Recall   |
|--------------------------------------|----------------|---------------|----------|----------|-----------|----------|
| Random Forest - Without Balancing    | 0.999998       | 0.998874      | 0.998874 | 0.998811 | 0.942857  | 0.753846 |
| Random Forest - Random Over Sampling | 1.000000       | 0.998902      | 0.998902 | 0.998856 | 0.924268  | 0.779487 |
| Random Forest - Random Under Sampling| 1.000000       | 0.974212      | 0.974212 | 0.983942 | 0.126563  | 0.962704 |
| Random Forest - SMOTE                | 0.946626       | 0.974212      | 0.995750 | 0.996366 | 0.472258  | 0.861072 |
| XGBoost - Random Over Sampling       | 0.997287       | 0.994199      | 0.994199 | 0.995372 | 0.394070  | 0.935664 |
| XGBoost - SMOTE                      | 0.988082       | 0.952071      | 0.952071 | 0.972100 | 0.070983  | 0.944522 |
| MLPClassifier - Random Over Sampling | 0.978773       | 0.971887      | 0.971887 | 0.982675 | 0.114738  | 0.935664 |
| MLPClassifier - SMOTE                | 0.965452       | 0.966953      | 0.966953 | 0.980026 | 0.097838  | 0.919814 |
| Logistic Regression - SMOTE          | 0.853682       | 0.906339      | 0.906339 | 0.947276 | 0.031066  | 0.770629 |
| Logistic Regression - Random Over Sampling| 0.853325  | 0.936842      | 0.936842 | 0.963885 | 0.044760  | 0.755245 |
| Decision Tree - SMOTE                | 1.000000       | 0.971002      | 0.971002 | 0.982158 | 0.105055  | 0.866200 |
| Decision Tree - Random Over Sampling | 1.000000       | 0.998195      | 0.998195 | 0.998216 | 0.754004  | 0.790210 |

## Conclusion 🎉

Based on the evaluation results, the Random Forest model with Random Over Sampling achieves the highest testing score of 99.89% and balanced performance across all metrics. This model is recommended for credit card fraud detection due to its robustness and accuracy.

## Acknowledgements 🙏

We would like to express our sincere gratitude to the following sources for their valuable insights and resources:

- [Kaggle - Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/index.html)
- [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Yellowbrick Documentation](https://www.scikit-yb.org/en/latest/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) - A powerful ensemble learning method provided by scikit-learn for classification tasks.
- [XGBoost](https://xgboost.readthedocs.io/en/latest/) - An efficient and scalable gradient boosting library widely used for classification and regression problems.
- [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) - A multi-layer perceptron classifier implemented in scikit-learn, suitable for classification tasks.
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) - A linear model used for binary classification tasks, provided by scikit-learn.
- [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) - A simple yet effective classification algorithm implemented in scikit-learn.

We express our gratitude for their remarkable algorithms and implementations, which have significantly contributed to the success of our fraud detection model.

## Disclaimer ⚠️

This study is a part of a data science seminar and is intended for educational purposes only. The analysis and findings presented here are based on simulated data and should not be used for any commercial purposes or real-world decision-making without proper validation and authorization.
