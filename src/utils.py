import os

import sys
from typing import Dict

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomeException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f_obj:
            dill.dump(obj, f_obj)

    except Exception as e:
        raise CustomeException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: Dict, params: Dict):
    reports = {}

    for model_label, model in models.items():
        param = params[model_label]

        gs = GridSearchCV(model, param_grid=param, cv=3)
        gs.fit(X_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)

        reports[model_label] = test_model_score

    return reports


def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustomeException(e, sys)
