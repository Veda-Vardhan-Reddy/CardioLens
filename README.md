# CardioLens — Heart Disease ML Comparative Study
Python target: **3.11.9**. Django web app + scikit-learn.

## Base paper / existing system
Ahmad et al., “Comparative Study of Optimum Medical Diagnosis of Human Heart Disease Using Machine Learning Technique With and Without Sequential Feature Selection,” IEEE Access, 2022, DOI: 10.1109/ACCESS.2022.3153047. It compares LDA, Random Forest, Gradient Boosting, Decision Tree, SVM and KNN, with/without SFS; uses five-fold CV and metrics including accuracy, sensitivity, specificity, F1 and ROC-AUC. Reported values are specific to that study's data and protocol.

## Proposed extension
Eight pipelines including Logistic Regression and Random Forest + forward SFS; stratified 80/20 test split; stratified five-fold CV on training data; balanced accuracy, specificity and confusion counts; pipeline preprocessing; saved model and report. SFS is run within CV folds to reduce leakage.

## Windows setup
```bat
py -3.11 --version
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_models.py
python manage.py migrate
python manage.py check
python manage.py runserver
```
Visit http://127.0.0.1:8000/

First training needs internet to download `https://raw.githubusercontent.com/plotly/datasets/master/heart.csv`, or put a compatible CSV in `data/heart.csv`. Required feature columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal; target column `target` or `num`. Official UCI dataset info: https://archive.ics.uci.edu/dataset/45/heart+disease

## Structure
`heart_project/` settings; `heart_app/` pages/CSS; `ml_engine/pipeline.py` training/inference; `data/` CSV; `models/` saved artifact; `reports/` comparison CSV.

## Medical safety
Educational prototype only, not a validated medical device. Do not use for diagnosis, treatment or emergency triage. The probability is not validated as calibrated individual risk.
