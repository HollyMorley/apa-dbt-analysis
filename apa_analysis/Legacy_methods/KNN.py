import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances


class ImbalancedKNN(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors classifier that handles class imbalance by incorporating
    class weights in the voting process.

    Parameters:
    -----------
    n_neighbors : int, default=5
        Number of neighbors to use for prediction
    weights : str or dict, default='balanced'
        If 'balanced', automatically adjust weights inversely proportional to
        class frequencies. If dict, contains the weight for each class.
    """

    def __init__(self, n_neighbors=5, weights='balanced'):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the imbalanced KNN classifier.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns:
        --------
        self : object
            Returns self
        """
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

        # Compute class weights if 'balanced'
        if self.weights == 'balanced':
            class_counts = Counter(self.y_train)
            n_samples = len(self.y_train)
            self.class_weights_ = {
                cls: n_samples / (len(class_counts) * count)
                for cls, count in class_counts.items()
            }
        else:
            self.class_weights_ = self.weights

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        X = np.asarray(X)
        y_pred = np.zeros(X.shape[0])

        # Compute distances between test samples and training samples
        distances = euclidean_distances(X, self.X_train)

        # For each test sample
        for i in range(X.shape[0]):
            # For each training sample, adjust its distance by its class weight.
            # (A higher class weight will reduce the effective distance, favoring that sample.)
            effective_distances = distances[i] / np.array([self.class_weights_[label] for label in self.y_train])
            # Select the indices of the k training samples with the smallest effective distances.
            nearest_neighbors = np.argsort(effective_distances)[:self.n_neighbors]
            # Use uniform voting among the selected neighbors.
            k_nearest_labels = self.y_train[nearest_neighbors]
            votes = Counter(k_nearest_labels)
            y_pred[i] = votes.most_common(1)[0][0]

        return y_pred

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        Returns:
        --------
        P : array-like of shape (n_samples, n_classes)
            Class probabilities for each sample
        """
        X = np.asarray(X)
        classes = np.unique(self.y_train)
        probas = np.zeros((X.shape[0], len(classes)))

        # Compute distances between test samples and training samples
        distances = euclidean_distances(X, self.X_train)

        # For each test sample
        for i in range(X.shape[0]):
            # Get indices of k nearest neighbors
            effective_distances = distances[i] / np.array([self.class_weights_[label] for label in self.y_train])
            nearest_neighbors = np.argsort(effective_distances)[:self.n_neighbors]

            # Get their labels
            k_nearest_labels = self.y_train[nearest_neighbors]

            # Compute class probabilities as the fraction of neighbors in each class.
            vote_counts = Counter(k_nearest_labels)
            total_votes = sum(vote_counts.values())
            for idx, cls in enumerate(classes):
                probas[i, idx] = vote_counts.get(cls, 0) / total_votes

        return probas


# # Create and fit the classifier
# clf = ImbalancedKNN(n_neighbors=5, weights='balanced')
# clf.fit(X_train, y_train)
#
# # Make predictions
# y_pred = clf.predict(X_test)
#
# # Get probability estimates
# probas = clf.predict_proba(X_test)
#
# # Or use custom weights
# custom_weights = {0: 1.0, 1: 2.0}  # Give class 1 twice the weight
# clf = ImbalancedKNN(n_neighbors=5, weights=custom_weights)