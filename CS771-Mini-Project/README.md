# CS771-Mini-Project

This respositry consists of the submission for the First Mini Project for the course [CS771](https://web.cse.iitk.ac.in/users/piyush/courses/ml-autumn24/index.html), Fall 2024, completed under the instruction of Prof. Piyush Rai, Department of CSE, IIT Kanpur

## Team 

| **Name** | **Roll Number** |
|----------|-----------------|
| Anushka Singh | 220188 |
| Arush Upadhyaya | 220213 |
| Aujasvit Datta | 220254 |
| Pahal Dhruvin Patel | 220742 |
| Pranav Agrawal | 220791 |

## Contents

- `17.py` : main file to generate and save predictions
- `utils.py` : utility functions used in `17.py`
- `pred_emoticon.txt` : predictions for the emoticons dataset
- `pred_deepfeat.txt` : predictions for the deep features dataset
- `pred_text_seq.txt` : predictions for the text sequences dataset
- `emoticons/` : jupyter notebooks containing experiments and EDA for emoticons dataset
- `features/` : jupyter notebooks containing experiments and EDA for features dataset
- `text_seq/` : jupyter notebooks containing experiments and EDA for text sequences dataset
- `combined/` : jupter notebooks containing experiments and EDA for all datasets combined
- `common/` : helper functions used in experiments

## Instructions

- Install the dependencies
```bash
pip install -r requirements.txt
```

- Download the dataset, make sure the `datasets/` directory is present in the root

- Run `17.py` to generate the prediction files &rarr;
```bash
python 17.py
```

## Approaches

### Dataset-1

- Preprocessing : 
    1. Removed dummy emojis, that are occuring in all the input emoji strings
    2. Columnarised the emoji strings into one column per character
    3. One hot encoded the categorical columns
- Model : Logistic Regression
- Best Parametres 

    |**Parameter**|**Value**|
    |-------------|---------|
    | C | 10|
    | penalty| L1|
    |Solver| Liblinear|

- Achieved Accuracy on Validation Set : **97.13%**

### Dataset-2

- Preprocessing : None
- Model : Logistic Regression
- Best Parametres
    |**Parameter**|**Value**|
    |-------------|---------|
    | C | 10.0 |
    | fit_intercept | True |
    | penalty | l2 |
    | solver | lbfgs |
    
- Achieved Accuracy on Validation Set : **98.77%**

### Dataset-3

- Preprocessing
    1. Removed substrings occuring in all the input strings
    2. Converted the input strings into n-gram respresentation, with $n_range = (3, 5)$
- Model : Logistic Regression
- Best Parametres 

    |**Parameter**|**Value**|
    |-------------|---------|
    | colsample_bytree | 1.0 |
    | eval_metric | logloss |
    | gamma | 0.2 |
    | learning_rate | 0.1 |
    | max_depth | 7 |
    | min_child_weight | 3 |
    | n_estimators | 500 |
    | subsample | 1.0 |

- Achieved Accuracy on Validation Set : **93.05%**

### Task 2
- Model : Logistic Regression
- Best Parametres : 

    |**Parameter**|**Value**|
    |-------------|---------|
    | C | 10.0 |
    | fit_intercept | True |
    | penalty | l2 |
    | solver | lbfgs |

- Achieved Accuracy on Validation Set : **98.77%**

We used the seed 42 for all the probabilistic models that we attempted to run.

