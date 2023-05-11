import numpy as np
from sklearn.datasets import fetch_openml
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

img= np.zeros((28,28))
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

plt.figure()
plt.imshow(X[0,:].reshape((28,28)), cmap="gray")
plt.show()
print(y)


# skaliraj podatke, train/test split
X = X / 255.
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


# TODO: izgradite vlastitu mrezu pomocu sckitlearn MPLClassifier 
mlp_mnist = MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001).fit(X_train,y_train)

# TODO: evaluirajte izgradenu mrezu
print(mlp_mnist.score(X_train,y_train))
print(mlp_mnist.score(X_test,y_test))

# spremi mrezu na disk
filename = "NN_model.sav"
joblib.dump(mlp_mnist, filename)

