import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = np.array(
    [5, 15, 25, 35, 45, 55]).reshape((-1, 1))

y = np.array([5, 20, 14, 32, 22, 38])

plt.scatter(x,y)
plt.show()
model = LinearRegression(
    fit_intercept=True,
    normalize=False,
    copy_X=True,
    n_jobs=None,
    positive=False
)

model.fit(x, y)

r_sq = model.score(x, y)
intercept = model.intercept_
slope = model.coef_

print(f"{r_sq=}")
print(f"{intercept=}")
print(f"{slope=}")

y_pred = model.predict(x)

plt.scatter(x,y)
plt.plot(x, y_pred)
filename = "LinearRegression.png"
plt.savefig(filename)
