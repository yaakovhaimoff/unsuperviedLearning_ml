"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================
For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

                      Noam Mirjani 315216515
                    Yaakov Haimoff 318528510
"""

# %%
# Generate data
# -------------
#
# In order to learn good latent representations from a small dataset, we
# artificially generate more labeled data by perturbing the training data with
# linear shifts of 1 pixel in each direction.
import numpy as np
import time
from scipy.ndimage import convolve
from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def fitting_time():
    """ plot the fit time array"""
    plt.figure()
    axis = [i ** 2 for i in range(2, 21)]
    plt.plot(axis, fit_time)
    plt.xlabel('Number of components')
    plt.ylabel('Fitting time (seconds)')
    plt.title('RBM fitting time')


def rbm_precisions():
    plt.figure()
    axis = [i ** 2 for i in range(2, 21)]
    plt.plot(axis, rbm_predictions_avg_lst)
    # Plot the horizontal line
    plt.axhline(y=0.78, color='r', linestyle='-')
    plt.xlabel('Number of components')
    plt.ylabel('precisions')
    plt.title('precisions vs number of components')


def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    ]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode="constant", weights=w).ravel()

    X = np.concatenate(
        [X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]
    )
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


X, y = datasets.load_digits(return_X_y=True)
X = np.asarray(X, "float32")
X, Y = nudge_dataset(X, y)
X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# %%
# Models definition
# -----------------
#
# We build a classification pipeline with a BernoulliRBM feature extractor and
# a :class:`LogisticRegression <sklearn.linear_model.LogisticRegression>`
# classifier.
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
rbm = BernoulliRBM(random_state=0, verbose=False)

rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

# %%
# Training
# --------
#
# The hyperparameters of the entire model (learning rate, hidden layer size,
# regularization) were optimized by grid search, but the search is not
# reproduced here because of runtime constraints.
from sklearn.base import clone
from sklearn.metrics import precision_score

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 10

rbm_predictions_avg_lst = []
raw_pixel_predictions_avg_lst = []

fit_time = []
for x in range(2, 21):

    start = time.perf_counter()

    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = x ** 2
    logistic.C = 6000

    # Training RBM-Logistic Pipeline
    rbm_features_classifier.fit(X_train, Y_train)

    # Training the Logistic regression classifier directly on the pixel
    raw_pixel_classifier = clone(logistic)
    raw_pixel_classifier.C = 100.0
    raw_pixel_classifier.fit(X_train, Y_train)

    stop = time.perf_counter()
    fit_time.append(stop - start)

    # Get the predicted labels for the test set
    rbm_predictions = rbm_features_classifier.predict(X_test)
    raw_pixel_predictions = raw_pixel_classifier.predict(X_test)

    # Calculate the precision of the classifiers
    rbm_precision = precision_score(Y_test, rbm_predictions, average='macro')
    raw_pixel_precision = precision_score(Y_test, raw_pixel_predictions, average='macro')

    rbm_predictions_avg_lst.append(rbm_precision)
    raw_pixel_predictions_avg_lst.append(raw_pixel_precision)
    # print("Precision of RBM-Logistic classifier: {:.3f}".format(rbm_precision))
    # print("Precision of Logistic classifier: {:.3f}".format(raw_pixel_precision))

    # %%
    # Evaluation
    # ----------
    Y_pred = rbm_features_classifier.predict(X_test)

    # %%
    Y_pred = raw_pixel_classifier.predict(X_test)

    # %%
    # The features extracted by the BernoulliRBM help improve the classification
    # accuracy with respect to the logistic regression on raw pixels.

    # %%
    # Plotting
    # --------
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(rbm.components_):
        plt.subplot(x, x, i + 1)
        plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation="nearest")
        plt.xticks(())
        plt.yticks(())
        plt.suptitle(f"{x * x} components extracted by RBM", fontsize=16)
        plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

''' 2c'''

fitting_time()
rbm_precisions()

''' 2d1 '''
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("rbm.transform(X_train) shape:", rbm.transform(X_train).shape)
print("rbm.intercept_hidden_ shape:", rbm.intercept_hidden_.shape)

plt.show()
