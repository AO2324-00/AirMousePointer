import numpy as np

G = np.array([[1]])
F = np.array([[1]])
W = np.array([[1]])
#V = np.array([[10]])
V = np.array([[10]])

'''
def filter(m, C, y, G=G, F=F, W=W, V=V):
    """
    Kalman Filter
    m: 時点t-1のフィルタリング分布の平均
    C: 時点t-1のフィルタリング分布の分散共分散行列
    y: 時点tの観測値
    """
    a = G @ m
    R = G @ C @ G.T + W
    f = F @ a
    Q = F @ R @ F.T + V

    K = (np.linalg.solve(Q.T, F @ R.T)).T

    m = a + K @ (y - f)
    C = R - K @ F @ R
    return m, C
'''

def filter(y, xPre, pPre, sigmaW, sigmaV):
    """
    y       :当期の観測値
    xPre    :前期の状態
    pPre    :前期の状態の予測誤差の分散
    sigmaW  :状態方程式のノイズの分散
    sigmaV  :観測方程式のノイズの分散
    """
    # お手製のカルマンフィルタ関数
  
    # 状態の予測(ローカルレベルモデルなので、予測値は、前期の値と同じ)
    xForecast = xPre[-1]
  
    # 状態の予測誤差の分散
    pForecast = pPre[-1] + sigmaW
  
    # カルマンゲイン
    kGain = pForecast / (pForecast + sigmaV)
  
    # カルマンゲインを使って補正された状態
    xFiltered = xForecast + kGain * (y - xPre[-1])
  
    # 補正された状態の予測誤差の分散
    pFiltered = (1 - kGain) * pForecast
  
    return xFiltered, pFiltered

def prediction(a, R, G=G, W=W):
    """
    Kalman prediction
    """
    a = G @ a
    R = G @ R @ G.T + W
    return a, R

# カルマン平滑化  
# 固定区間平滑化を行う  
# 時点Tまでのカルマンフィルタリングが一旦完了しているものとする  
# aとRの計算は、カルマンフィルタリングの計算時に格納したものを使った方が、計算効率は良さそう
def smoothing(s, S, m, C, G=G, W=W):
    """
    Kalman smoothing
    """
    # 1時点先予測分布のパラメータ計算
    a = G @ m
    R = G @ C @ G.T + W
    # 平滑化利得の計算
    # solveを使った方が約30%速くなる
    A = np.linalg.solve(R.T, G @ C.T).T
    # A = C @ G.T @ np.linalg.inv(R)
    # 状態の更新
    s = m + A @ (s - a)
    S = C + A @ (S - R) @ A.T
    return s, S

def calcState(data, x0, P0, sigmaW=1000, sigmaV=10000):
    """
    data    :データ
    x0      :状態の初期値
    P0      :状態の予測誤差の分散の初期値
    sigmaW  :状態方程式のノイズの分散
    sigmaV  :観測方程式のノイズの分散
    """
    # カルマンフィルタを使って、状態を一気に推定する

    # サンプルサイズ
    N = len(data)

    # 状態の予測誤差の分散
    P = np.arange(N+1.0)

    # 「状態の予測誤差の分散」の初期値の設定
    P[0] = P0

    # 状態の推定値
    x = np.arange(N+1.0)

    # 「状態」の初期値の設定
    x[0] = x0

    # カルマンフィルタの逐次計算を行う
    for i in np.arange(N):
        xFiltered, pFiltered = filter(data[i],x[:i+1],P[:i+1], sigmaW, sigmaV)
        x[i + 1] = xFiltered
        P[i + 1] = pFiltered

    # 推定された状態を返す
    return x[1:], P[1:]

def calcSmoothState(x, P):

    # サンプルサイズ
    N = len(x)

    s = np.zeros((N, 1))
    S = np.zeros((N, 1, 1))

    for n in range(N):
        n = N - n - 1
        if n == N - 1:
            s[n] = x[n]
            S[n] = P[n]
        else:
            s[n], S[n] = smoothing(s[n+1], S[n+1], x[n], P[n])
    
    return s.flatten(), S.flatten()