import typing
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def get_char_columns(df):
    for i in range(len(df['input_emoticon'][0])):
        df[f'c_{i+1}'] = df['input_emoticon'].apply(lambda x, _i=i: x[_i])
        
    columns = df.columns.to_list()
    if 'label' in columns:
        columns.remove('label')
    columns.remove('input_emoticon')
    
    return df[columns + (['label'] if 'label' in df.columns else []) ]

def getdfs(data : str):
    '''
    preprocess and return train_df and val_df
    '''
    if data not in ['text_seq', 'feature' , 'emoticon']:
        raise ValueError("Invalid data type")
    
    if data == 'feature' : 
        train_df = np.load(f"../datasets/train/train_{data}.npz")
        valid_df = np.load(f"../datasets/valid/valid_{data}.npz")
    else :
        train_df = pd.read_csv(f"../datasets/train/train_{data}.csv")
        valid_df = pd.read_csv(f"../datasets/valid/valid_{data}.csv")
    
    return train_df, valid_df

def one_hot_encode(train_df, valid_df):
    '''
    one hot encode the character columns for emoticons
    '''
    y_train = train_df['label']
    y_val = valid_df['label']
    
    new_train_df = train_df.drop('label', axis=1)
    new_valid_df = valid_df.drop('label', axis = 1)
    
    oh_encoder = OneHotEncoder(handle_unknown = 'ignore')
    oh_encoder.fit(new_train_df)

    
    new_train_df = pd.DataFrame(oh_encoder.transform(new_train_df).toarray())
    new_valid_df = pd.DataFrame(oh_encoder.transform(new_valid_df).toarray())
    
    return new_train_df, new_valid_df, y_train, y_val


def remove_substrings(input_string, substrings):
    """
    Removes all occurrences of substrings from the input string.

    Parameters:
    input_string (str): The string to remove substrings from.
    substrings (list): List of substrings to remove from the input string.

    Returns:
    str: The input string with substrings removed.
    """
    for substring in substrings:
        input_string = input_string.replace(substring, "")
    return input_string


def process_strings ( strs : typing.List[str] )-> typing.List[str]:
    strs = [x.lstrip('0') for x in strs]
    
    repeat_emo_code = {
        '🙼' : '284',
        '🛐' : '464', 
        '🙯' : '262',
        '😛' : '15436', 
        '😣' : '614',
        '😑' : '1596', 
        '🚼' : '422'
    }

    # Example usage
    substrings = repeat_emo_code.values()

    # Remove the substrings
    strs = [remove_substrings(input_string, substrings) for input_string in strs]

    padded_strs = []

    for s in strs:
        if len(s) < 15:
            s = s + '0'*(15-len(s))
        padded_strs.append(s)

    return padded_strs

def find_common_characters(strings):
    # Convert the first string to a set of characters
    common_chars = set(strings[0])

    # Intersect with characters from all other strings
    for string in strings[1:]:
        common_chars &= set(string)
    
    return common_chars

def remove_common_characters(strings):
    new_strings = []
    
    common_chars = find_common_characters(strings)
    
    for string in strings:
        # Remove all common characters from the string
        new_string = ''.join(char for char in string if char not in common_chars)
        new_strings.append(new_string)

    return new_strings