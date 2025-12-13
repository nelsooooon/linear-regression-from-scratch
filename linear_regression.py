import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gradient_descent(m_now, b_now, points, learning_rate):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        xi = points.iloc[i].x
        yi = points.iloc[i].y
        
        m_gradient += -(2/n) * xi * (yi - (m_now * xi + b_now))
        b_gradient += -(2/n) * (yi - (m_now * xi + b_now))
        
    m = m_now - m_gradient * learning_rate
    b = b_now - b_gradient * learning_rate
    
    return m, b

def drop_outlier(df):
    df_clean = df.copy()
    
    for col in ['x', 'y']:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        
        IQR = Q3 - Q1
        lower = Q1 - (1.5 * IQR)
        upper = Q3 + (1.5 * IQR)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        
    return df_clean

train_path = 'res/train.csv'
test_path = 'res/test.csv'

data_train = pd.read_csv(train_path)
data_test = pd.read_csv(test_path)
data_train_clean = data_train.dropna()
data_train_clean = drop_outlier(data_train_clean)

m = 0
b = 0
learning_rate = 0.0001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
        
    m, b = gradient_descent(m, b, data_train_clean, learning_rate)
    
print(m, b)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(x=data_train_clean.x, y=data_train_clean.y, color='black', alpha=0.5)
axes[1].scatter(x=data_test.x, y=data_test.y, color='black', alpha=0.5)

for i in range(len(axes)):
    axes[i].plot(list(range(1, 101)), [m * x + b for x in range(1, 101)], color='red')

fig.tight_layout()
plt.show()