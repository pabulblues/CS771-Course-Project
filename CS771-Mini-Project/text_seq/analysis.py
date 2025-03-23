import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt

train_seq = pd.read_csv('../datasets/train/train_text_seq.csv')
valid_seq = pd.read_csv('../datasets/valid/valid_text_seq.csv')
test_seq = pd.read_csv('../datasets/test/test_text_seq.csv')

all_digits = ''.join(train_seq['input_str'].astype(str))

digit_counts = Counter(all_digits)
for digit, count in digit_counts.items():
    print(f"Digit: {digit}, Count: {count}")
    
df = pd.DataFrame(train_seq)
digit_counts = Counter(all_digits)


digits, counts = zip(*sorted(digit_counts.items()))  

plt.figure(figsize=(10, 6))
plt.bar(digits, counts, color='skyblue')
plt.xlabel('Digits')
plt.ylabel('Frequency')
plt.title('Frequency of Digits in input_str Column')
plt.show()

#Removing all the padded zeroes from the data 
df['input_str'] = df['input_str'].str.replace('0', '')

# Verify the changes
print(df.head())

