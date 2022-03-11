import numpy as np

class Parceptron(object):

  """
  パラメータ
  ---------
  eta : float（学習率：0.0 < eta >= 1.0）
  n_iter : int（訓練データの訓練回数）
  random_state : int（重みを初期化するための乱数シード）

  属性
  ---------
  w_ : １次元配列（適合後の重み）
  errors_ : リスト（各エポックでの誤分類の数）
  """

  def __init__(self, eta=0.01, n_iter=50, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

    #訓練データに適合させる関数
  def fit(self, X, Y):
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
    self.errors_ = []

    for _ in range(self.n_iter):
      errors_ = 0
      for xi, terget in zip(X, y):
        update = self.eta * (target - self.predict(xi))
        self.w_