# Customer-Churn-Prediction
- Developed and compared several machine learning models (Logistic Regression, Gradient Boosting Classifier) demonstrated strong predictive performacne with an AUC score of 0.84 in identifying likely churners
- Fine-tuned the performacne of the classification models using GridSearchCV to achieve the final AUC score
- Performed interpretability analysis (using SHAP) to understand the factors driving the model's predictions, revealing specific customer attributes that strongly influence churn probability. The analysis revealed that contract type, monthly charges, and tenure were the strongest predictors of churn.
- Built a client-facing RESTful API for the model using Flask
