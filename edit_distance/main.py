import sys
from embedding import bert_embedding
import numpy as np

def cos_dist(q: np.ndarray):
    x = q[1::2]
    y = q[::2]

    n = x.shape[0]
    ans = []

    for i in range(n):
        xi = x[i]
        yi = y[i].T
        
        sqr_x = np.sqrt(np.sum(xi ** 2))
        sqr_y = np.sqrt(np.sum(yi ** 2))

        cos = xi @ yi / sqr_x / sqr_y

        ans.append((1 - cos) / 2.0)

    return ans

def l2_dist(q: np.ndarray):
    x = q[1::2]
    y = q[::2]    

    n = x.shape[0]
    ans = []
    
    for i in range(n):
        xi = x[i]
        yi = y[i].T
        
        sqr_x = np.sum(xi ** 2)
        sqr_y = np.sum(yi ** 2)

        ans.append(np.sqrt(sqr_x + sqr_y - 2 * xi @ yi))

    return ans

f = open(sys.argv[1])

seqs = []
edits = []

x = f.readline()
while len(x) > 0:
    y = x.strip().split(',')
    seqs.append(y[0])
    seqs.append(y[1])

    edits.append(float(y[2]) / ((len(y[0]) + len(y[1])) / 2.0))

    x = f.readline()

emb = bert_embedding(seqs, sys.argv[2])
dist = cos_dist(emb)

from scipy import stats
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

res = stats.spearmanr(dist, edits)
print("Spearman's rank correlation: {}".format(res.statistic))
res = stats.pearsonr(dist, edits)
print("Pearson correlation: {}".format(res.statistic))

X = np.array(dist)
y = np.array(edits)

X_train, y_train = X[1::2], y[1::2]
X_test, y_test = X[0::2], y[0::2]

model = LinearRegression()
model.fit(X_train.reshape(-1, 1), y_train)
y_pred = model.predict(X_test.reshape(-1, 1))

mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("MAPE: {}".format(mape))
print("MSE: {}".format(mse))