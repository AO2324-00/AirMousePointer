import numpy as np
import kalman

import matplotlib.pyplot as plt

G = np.array([[1]])
F = np.array([[1]])
W = np.array([[1]])

T = 50
K = 10
x0 = 0

x = np.arange(0, T) #サンプリング点の生成
#y = x*0.8 
#y = np.full(T, 10)
y=(x-10)**2

# 初期状態のフィルタリング分布のパラメータ
m0 = np.array([[0]])
C0 = np.array([[0]])

# 結果を格納するarray
m = np.zeros((T, 1))
C = np.zeros((T, 1, 1))
a_pred = np.zeros((K, 1))
R_pred = np.zeros((K, 1, 1))
s = np.zeros((T, 1))
S = np.zeros((T, 1, 1))

# カルマンフィルター
for t in range(T):
    if t == 0:
        m[t], C[t] = kalman.filter(m0, C0, y[t:t+1])
    else:
        m[t], C[t] = kalman.filter(m[t-1:t], C[t-1:t], y[t:t+1])


# カルマン予測
for t in range(K):
    if t == 0:
        a = G @ m[T-1:T]
        R = G @ C[T-1:T] @ G.T + W
        a_pred[t] = a
        R_pred[t] = R
    else:
        a_pred[t], R_pred[t] = kalman.prediction(a_pred[t-1], R_pred[t-1])

# カルマン平滑化
for t in range(T):
    t = T - t - 1
    if t == T - 1:
        s[t] = m[t]
        S[t] = C[t]
    else:
        s[t], S[t] = kalman.smoothing(s[t+1], S[t+1], m[t], C[t])
print(y)
result, RESULT = kalman.calcState(y, y[0], 0, 1000, 10000)
print(result)

plt.plot(x, y, color="Black")
#plt.plot(x, m.flatten())
#plt.plot(x, a_pred.flatten())
#plt.plot(x, s.flatten())
#print(len(x), len(y),len(s.flatten()))
plt.plot(x, result, color="Blue")
plt.plot(x, RESULT, color="Red")
#print(result)
plt.show()