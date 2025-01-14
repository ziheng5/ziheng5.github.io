---
title: Kaggle：Playground S4 E8
date: 2024-12-09 15:06:59
tags: 
    - 深度学习
    - PyTorch
categories: 
    - Kaggle 练习
description: |
    一道以一维数据为训练数据集的二分类问题
---
> 🔗比赛链接：https://www.kaggle.com/competitions/playground-series-s4e8
>
> 训练数据集、测试数据集和提交样例都需要自行从比赛页面下载。

## 一、📦导入必要的包
```Python
# 数据处理
import numpy as np
import pandas as pd

# 搭建模型
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef

# PyTorch
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

# 探索特征
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot

import os
```

## 二、🪵导入训练数据与测试数据
### 1. 导入训练数据
```Python
# 导入训练数据
train = pd.read_csv("train.csv")
train.head()
```

输出如下（由于 categories 过长，这里展示一部分结果）：

```Terminal
   id class  cap-diameter cap-shape  ... ring-type spore-print-color habitat season
0   0     e          8.80         f  ...         f               NaN       d      a
1   1     p          4.51         x  ...         z               NaN       d      w
2   2     e          6.94         f  ...         f               NaN       l      w
3   3     e          3.88         f  ...         f               NaN       d      u
4   4     e          5.85         x  ...         f               NaN       g      a

[5 rows x 22 columns]
```

### 2. 导入测试数据
```Python
# 导入测试数据
test = pd.read_csv("test.csv")
test.head()
```

输出如下（由于 categories 过长，这里展示一部分结果）：

```Terminal
        id  cap-diameter cap-shape  ... spore-print-color habitat season
0  3116945          8.64         x  ...               NaN       d      a
1  3116946          6.90         o  ...               NaN       d      a
2  3116947          2.00         b  ...               NaN       d      s
3  3116948          3.47         x  ...               NaN       d      u
4  3116949          6.17         x  ...               NaN       d      u

[5 rows x 21 columns]
```

> 观察数据内容发现 test.csv 中没一行没有 “class”，即 test.csv 是需要被分类的数据。

## 三、🔍数据分析
### 1. 数据信息
```Python
train.info(verbose=True, show_counts=True)
```

输出如下：

```Terminal
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3116945 entries, 0 to 3116944
Data columns (total 22 columns):
 #   Column                Non-Null Count    Dtype  
---  ------                --------------    -----  
 0   id                    3116945 non-null  int64  
 1   class                 3116945 non-null  object 
 2   cap-diameter          3116941 non-null  float64
 3   cap-shape             3116905 non-null  object 
 4   cap-surface           2445922 non-null  object 
 5   cap-color             3116933 non-null  object 
 6   does-bruise-or-bleed  3116937 non-null  object 
 7   gill-attachment       2593009 non-null  object 
 8   gill-spacing          1858510 non-null  object 
 9   gill-color            3116888 non-null  object 
 10  stem-height           3116945 non-null  float64
 11  stem-width            3116945 non-null  float64
 12  stem-root             359922 non-null   object 
 13  stem-surface          1136084 non-null  object 
 14  stem-color            3116907 non-null  object 
 15  veil-type             159452 non-null   object 
 16  veil-color            375998 non-null   object 
 17  has-ring              3116921 non-null  object 
 18  ring-type             2988065 non-null  object 
 19  spore-print-color     267263 non-null   object 
 20  habitat               3116900 non-null  object 
 21  season                3116945 non-null  object 
dtypes: float64(3), int64(1), object(18)
memory usage: 523.2+ MB
```

从上面的输出可以看到，训练数据中，第 0 个 column 是 id，第 1 个 column 是 class，后面的 columns 均为详细参数。


```Python
test.info(verbose=True, show_counts=True)
```

输出如下：

```Terminal
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2077964 entries, 0 to 2077963
Data columns (total 21 columns):
 #   Column                Non-Null Count    Dtype  
---  ------                --------------    -----  
 0   id                    2077964 non-null  int64  
 1   cap-diameter          2077957 non-null  float64
 2   cap-shape             2077933 non-null  object 
 3   cap-surface           1631060 non-null  object 
 4   cap-color             2077951 non-null  object 
 5   does-bruise-or-bleed  2077954 non-null  object 
 6   gill-attachment       1728143 non-null  object 
 7   gill-spacing          1238369 non-null  object 
 8   gill-color            2077915 non-null  object 
 9   stem-height           2077963 non-null  float64
 10  stem-width            2077964 non-null  float64
 11  stem-root             239952 non-null   object 
 12  stem-surface          756476 non-null   object 
 13  stem-color            2077943 non-null  object 
 14  veil-type             106419 non-null   object 
 15  veil-color            251840 non-null   object 
 16  has-ring              2077945 non-null  object 
 17  ring-type             1991769 non-null  object 
 18  spore-print-color     178347 non-null   object 
 19  habitat               2077939 non-null  object 
 20  season                2077964 non-null  object 
dtypes: float64(3), int64(1), object(17)
memory usage: 332.9+ MB
```

### 2. 去除 NaN
```Python
# 下面的这些 columns 基本上没有有用信息（全是 NaN）
mostly_nan_cols = ['stem-root', 'stem-surface', 'veil-type', 'veil-color', 'spore-print-color']

# 这些是含有大量 NaN 的 columns 
many_nan_cols = ['gill-spacing', 'cap-surface', 'gill-attachment', 'ring-type']

# 去掉全是 NaN 的 columns
train.drop(mostly_nan_cols, axis=1, inplace=True)
test.drop(mostly_nan_cols, axis=1, inplace=True)
```



```Python
# 查看 NaN 的数量
pd.DataFrame([train.isna().sum(), test.isna().sum()]).T
```

输出如下：

```Terminal
                              0         1
id                          0.0       0.0
class                       0.0       NaN
cap-diameter                4.0       7.0
cap-shape                  40.0      31.0
cap-surface            671023.0  446904.0
cap-color                  12.0      13.0
does-bruise-or-bleed        8.0      10.0
gill-attachment        523936.0  349821.0
gill-spacing          1258435.0  839595.0
gill-color                 57.0      49.0
stem-height                 0.0       1.0
stem-width                  0.0       0.0
stem-color                 38.0      21.0
has-ring                   24.0      19.0
ring-type              128880.0   86195.0
habitat                    45.0      25.0
season                      0.0       0.0
```

### 3. 删除离谱数值

```Python
# 查看每一个 column 中的元素类别有多少
train.nunique()
```

输出如下：

```Terminal
id                      3116945
class                         2
cap-diameter               3913
cap-shape                    74
cap-surface                  83
cap-color                    78
does-bruise-or-bleed         26
gill-attachment              78
gill-spacing                 48
gill-color                   63
stem-height                2749
stem-width                 5836
stem-color                   59
has-ring                     23
ring-type                    40
habitat                      52
season                        4
dtype: int64
```

发现有的 columns 中**元素类别特别多**，所以有必要对数据做一些修剪。

接着运行：

```Python
# 所有 columns
categorical_cols = many_nan_cols + ['cap-shape', 'cap-color', 'does-bruise-or-bleed', 'gill-color', 'stem-color', 'has-ring', 'habitat', 'season']

# 由于过小的数据对我们训练结果的影响微乎其微，所以我们可以将小于100的值全部用 “rare” 替换掉
for c in categorical_cols:
    col_count = train[c].value_counts()
    rare_keys = col_count[col_count<100].index
    train[c] = train[c].replace(rare_keys, 'rare')
    
    col_count = test[c].value_counts()
    rare_keys = col_count[col_count<100].index
    test[c] = test[c].replace(rare_keys, 'rare')

# 再次查看每一个 column 中元素类别数
train.nunique()
```

输出如下：

```Terminal
id                      3116945
class                         2
cap-diameter               3913
cap-shape                     8
cap-surface                  12
cap-color                    13
does-bruise-or-bleed          3
gill-attachment               8
gill-spacing                  4
gill-color                   13
stem-height                2749
stem-width                 5836
stem-color                   14
has-ring                      3
ring-type                     9
habitat                       9
season                        4
dtype: int64
```

通过上述操作，类别数减少了一部分

### 4. 填充 NaNs
```Python
for c in train.columns:
    train[c] = train[c].fillna(train[c].mode().values[0])
train.isna().sum()
```

输出如下：

```Terminal
id                      0
class                   0
cap-diameter            0
cap-shape               0
cap-surface             0
cap-color               0
does-bruise-or-bleed    0
gill-attachment         0
gill-spacing            0
gill-color              0
stem-height             0
stem-width              0
stem-color              0
has-ring                0
ring-type               0
habitat                 0
season                  0
dtype: int64
```

再对 test 数据进行上述操作：

```Python
for c in test.columns:
    test[c] = test[c].fillna(test[c].mode().values[0])
test.isna().sum()
```

输出如下：

```Terminal
id                      0
cap-diameter            0
cap-shape               0
cap-surface             0
cap-color               0
does-bruise-or-bleed    0
gill-attachment         0
gill-spacing            0
gill-color              0
stem-height             0
stem-width              0
stem-color              0
has-ring                0
ring-type               0
habitat                 0
season                  0
dtype: int64
```

## 四、🧠模型
### 1. 数据分割
```Python
# 将原始 train 数据分割为 values 和 labels
X = train.drop(['class', 'id'], axis=1)
y = train['class']

# 将 70% 的 train 数据用于训练，30% 的数据用于测试（不是待测数据集 test.csv）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=4)
```

### 2. 将 categories 转变为整型数据
```Python
oe = OrdinalEncoder()
oe.fit(X_train[categorical_cols])
X_train[categorical_cols] = oe.transform(X_train[categorical_cols])
X_test[categorical_cols] = oe.transform(X_test[categorical_cols])
test[categorical_cols] = oe.transform(test[categorical_cols])

encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)
```

### 3. 特征选择
```Python
def select_features(X_train, y_train):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    return X_train_fs, fs


# applying feature selection on the training data
X_train_fs, fs = select_features(X_train[categorical_cols], y_train)

# what are scores for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
    
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
```

输出如下：


```Terminal
Feature 0: 20197.118520
Feature 1: 10012.712954
Feature 2: 50523.701865
Feature 3: 39612.848786
Feature 4: 32995.118490
Feature 5: 8665.637682
Feature 6: 5011.635136
Feature 7: 20938.868347
Feature 8: 22027.600984
Feature 9: 8278.886174
Feature 10: 4959.358699
Feature 11: 10198.071977
```

![figure1](./images/kaggle_s4e8/figure1.png)

### 4. 准备好 PyTorch 需要用到的数据
```Python
# 使用 GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Convert your training and test data to tensors and move them to the correct device
X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

# size of training data
X_train.shape
```

输出如下：

```Terminal
torch.Size([2181861, 15])
```

### 5. 模型搭建
```Python
# Notes:
# dropout 设置为 0.5 时效果不佳，0.2 时效果较好
# `self.layers.append(nn.Dropout(dropout_rate))`
class Deep(nn.Module):
    def __init__(self, input_size=X_train.shape[1], hidden_size=128, num_layers=5, dropout_rate=0.2):
        super(Deep, self).__init__()
        
        # Create a list of layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_size*2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_size*2))
        
        # Wide hidden layer
        self.layers.append(nn.Linear(hidden_size*2, hidden_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hidden_size))
        
        # Hidden layers
        for _ in range(num_layers - 3):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(hidden_size))
        
        hiddent_last = int(hidden_size/2)
        
        # Smaller Hidden layer
        self.layers.append(nn.Linear(hidden_size, hiddent_last))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(hiddent_last))
        
        # Output layer
        self.output = nn.Linear(hiddent_last, 1)
        
        # Apply Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(self.output(x))
        return x

def model_train(model, X_train, y_train, X_val, y_val):
    # Defining the loss function and optimizer
    loss_fn = nn.BCELoss()  # Binary cross-entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 250  # Number of epochs to run
    batch_size = 1024*10  # Size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = -np.inf  # Initialize to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # Take a batch
                X_batch = X_train[start:start+batch_size].to(device)
                y_batch = y_train[start:start+batch_size].to(device)
                # Forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Update weights
                optimizer.step()
                # Print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # Evaluate accuracy at the end of each epoch
        model.eval()
        y_pred = model(X_val.to(device))
        acc = (y_pred.round() == y_val.to(device)).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # Restore the best model and return the best accuracy
    model.load_state_dict(best_weights)
    return best_acc
```

下面开始训练：

```Python
# 定义 5 倍交叉验证测试工具
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores_wide = []

for train_idx, test_idx in kfold.split(X_train.cpu(), y_train.cpu()):  # Note: sklearn needs CPU-based numpy arrays
    # Create model, train, and get accuracy
    model = Deep().to(device)
    acc = model_train(model, X_train[train_idx], y_train[train_idx], X_train[test_idx], y_train[test_idx])
    print("Accuracy (deep): %.4f" % acc)
    cv_scores_wide.append(acc)

# 评估训练效果
acc = np.mean(cv_scores_wide)
std = np.std(cv_scores_wide)
print("Model accuracy: %.4f%% (+/- %.4f%%)" % (acc * 100, std * 100))
```

输出如下：

```Terminal
Accuracy (deep): 0.9879
Accuracy (deep): 0.9880
Accuracy (deep): 0.9882
Accuracy (deep): 0.9879
Accuracy (deep): 0.9880
Model accuracy: 98.8006% (+/- 0.0103%)
```

### 6. 训练效果检验
```Python
model = Deep().to(device)

model.load_state_dict(torch.load("model.pth"))
model.eval()

threshold = 0.5
y_pred = model(X_test)
y_pred = np.array((y_pred > threshold).float().cpu()) # 0.0 or 1.0

print(f"Accuracy Score: {accuracy_score(y_test.cpu(), y_pred)*100:.3f} %")
print(f"Matthews Correlation Coefficient (MCC): {matthews_corrcoef(y_test.cpu(), y_pred):.5f}")
```

输出如下：

```Terminal
Accuracy Score: 98.810 %
Matthews Correlation Coefficient (MCC): 0.97598
```

### 7. 处理待测数据成为待提交文件
```Python
X_real = test.drop(['id'], axis=1)
X_real = torch.tensor(X_real.values, dtype=torch.float32).to(device)

y_real = model(X_real)

y_real = np.array((y_real > threshold).int().cpu())

# getting the prepared submission file to edit on it
submission = pd.read_csv("sample_submission.csv")

# replacing the results with the new one
submission['class'] = y_real

# converting the 0s and 1s into e-or-p class
submission['class'] = submission['class'].astype('str')
submission.loc[(submission['class'] == '0'), 'class'] = 'e'
submission.loc[(submission['class'] == '1'), 'class'] = 'p'
```

## 五、提交检验成果
最后拿到了 0.975 的准确率💪