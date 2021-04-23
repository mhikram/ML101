import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

x = np.array(
    [a[0] for a in data]).reshape((-1, 1))

y = np.array([a[1] for a in data])

transformer = PolynomialFeatures(
    degree=3,
    include_bias=False,
    interaction_only = False
)

x_ = transformer.fit_transform(x)

model = LinearRegression(
    fit_intercept=True,
    normalize=False,
    copy_X=True,
    n_jobs=None,
    positive=False
)

model.fit(x_, y)

y_pred = model.predict(x_)

plt.scatter(x,y)
plt.plot(x, y_pred)


print(y_pred)

plt.show()

