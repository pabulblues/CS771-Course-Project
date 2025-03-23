import pandas as pd
import numpy as np
from utils import (
    print_accuracy,
    remove_common_characters,
    get_char_columns,
    find_common_characters,
    process_strings,
    print_delimiter,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer


def generate_submission_txt(model, x_test, file_name):
    """
    Generate submission file for the model
    """

    preds = model.predict(x_test)
    with open(file_name, "w") as f:
        for pred in preds:
            f.write(str(pred) + "\n")
    print(f"Submission file generated at {file_name}")
    print_delimiter()


def save_emoticons():
    train_df = pd.read_csv("datasets/train/train_emoticon.csv")
    valid_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
    test_df = pd.read_csv("datasets/test/test_emoticon.csv")

    train_df["input_emoticon"] = remove_common_characters(train_df["input_emoticon"])
    valid_df["input_emoticon"] = remove_common_characters(valid_df["input_emoticon"])
    test_df["input_emoticon"] = remove_common_characters(test_df["input_emoticon"])

    train_df = get_char_columns(train_df)
    valid_df = get_char_columns(valid_df)
    test_df = get_char_columns(test_df)

    y_train = train_df["label"]
    y_valid = valid_df["label"]

    train_df = train_df.drop("label", axis=1)
    valid_df = valid_df.drop("label", axis=1)

    oh_encoder = OneHotEncoder(handle_unknown="ignore")
    oh_encoder.fit(train_df)

    x_train = pd.DataFrame(oh_encoder.transform(train_df).toarray())
    x_valid = pd.DataFrame(oh_encoder.transform(valid_df).toarray())
    x_test = pd.DataFrame(oh_encoder.transform(test_df).toarray())

    params = {"C": 10, "penalty": "l1", "solver": "liblinear"}

    model = LogisticRegression(**params, max_iter= 10000, random_state = 42)
    model.fit(x_train, y_train)
    y_valid_pred = model.predict(x_valid)
    
    print_accuracy(y_valid, y_valid_pred, title = "emoticons dataset")
    
    generate_submission_txt(model, x_test, file_name='pred_emoticon.txt')
    print_delimiter()


def save_features():
    train_df = np.load("datasets/train/train_feature.npz", allow_pickle=True)
    valid_df = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
    test_df = np.load("datasets/test/test_feature.npz", allow_pickle=True)

    x_train = train_df["features"]
    y_train = train_df["label"]
    x_valid = valid_df["features"]
    y_valid = valid_df["label"]
    x_test = test_df["features"]

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    params = {'C': 10.0, 'fit_intercept': True, 'penalty': 'l2', 'solver': 'lbfgs'}
    model = LogisticRegression(**params, max_iter= 10000, random_state = 42)
    model.fit(x_train, y_train)
    y_valid_pred = model.predict(x_valid)
    
    print_accuracy(y_valid, y_valid_pred, title = "features dataset")
    
    generate_submission_txt(model, x_test, file_name='pred_deepfeat.txt')
    print_delimiter()
    
def save_text_seq():
    train_df = pd.read_csv("datasets/train/train_text_seq.csv")
    valid_df = pd.read_csv("datasets/valid/valid_text_seq.csv")
    test_df = pd.read_csv("datasets/test/test_text_seq.csv")

    train_df["input_str"] = process_strings(train_df["input_str"])
    valid_df["input_str"] = process_strings(valid_df["input_str"])
    test_df["input_str"] = process_strings(test_df["input_str"])

    y_train = train_df["label"].values
    y_valid = valid_df["label"].values

    _x_train = train_df["input_str"].values
    _x_valid = valid_df["input_str"].values
    _x_test = test_df["input_str"].values

    vectorizer = CountVectorizer(ngram_range=(3, 5), analyzer="char")  # Extract n-grams

    x_train = vectorizer.fit_transform(_x_train)
    x_valid = vectorizer.transform(_x_valid)
    x_test = vectorizer.transform(_x_test)

    params = {
        "colsample_bytree": 1.0,
        "eval_metric": "logloss",
        "gamma": 0.2,
        "learning_rate": 0.1,
        "max_depth": 7,
        "min_child_weight": 3,
        "n_estimators": 500,
        "subsample": 1.0,
    }
    model = XGBClassifier(**params)
    model.fit(x_train, y_train)
    y_valid_pred = model.predict(x_valid)    
    print_accuracy(y_valid, y_valid_pred, title = "text sequences dataset")
    
    generate_submission_txt(model, x_test, file_name='pred_text_seq.txt')
    print_delimiter()

def save_combined() :
    # same as save_features()
    train_df = np.load("datasets/train/train_feature.npz", allow_pickle=True)
    valid_df = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
    test_df = np.load("datasets/test/test_feature.npz", allow_pickle=True)

    x_train = train_df["features"]
    y_train = train_df["label"]
    x_valid = valid_df["features"]
    y_valid = valid_df["label"]
    x_test = test_df["features"]

    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    params = {'C': 10.0, 'fit_intercept': True, 'penalty': 'l2', 'solver': 'lbfgs'}
    model = LogisticRegression(**params, max_iter= 10000, random_state = 42)
    model.fit(x_train, y_train)
    y_valid_pred = model.predict(x_valid)
    
    print_accuracy(y_valid, y_valid_pred, title = "combined")
    
    generate_submission_txt(model, x_test, file_name='pred_combined.txt')
    print_delimiter()

if __name__ == "__main__":
    save_emoticons()
    save_features()
    save_text_seq()
    save_combined()
