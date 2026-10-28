
from django.urls import path
from . import views

urlpatterns = [
    # Signup, login and logout
    path("signup/", views.signup_view, name="signup"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),

    # Main workflow
    path("", views.dashboard, name="dashboard"),
    path("upload/", views.upload_dataset, name="upload_dataset"),
    path("preprocess/", views.preprocess, name="preprocess"),
    path("split/", views.split_dataset, name="split_dataset"),
    path("train/", views.train_models, name="train_models"),

    # Evaluation and graphs
    path("model-comparison/", views.comparison, name="comparison"),
    path("evaluation/", views.comparison, name="evaluation"),

    # Two prediction options
    path("predict/", views.predict_view, name="predict"),
    path("predict-csv/", views.csv_predict, name="csv_predict"),

    path("about/", views.about, name="about"),
]
