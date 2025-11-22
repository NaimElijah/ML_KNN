import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]




# TODO: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """
    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    classifier = {
        "k": k,
        "x_train": x_train,
        "y_train": y_train
    }
    return classifier


def predictknn(classifier, x_test: np.array):
    """
    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    k = classifier["k"]
    x_train = classifier["x_train"]
    y_train = classifier["y_train"].reshape(-1).astype(int)

    # Precompute squared norms
    x_train_sq = np.sum(x_train ** 2, axis=1)
    x_test_sq = np.sum(x_test ** 2, axis=1)

    # Compute distance matrix: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    dists = np.sqrt(
        x_test_sq[:, None] +
        x_train_sq[None, :] -
        2 * (x_test @ x_train.T)
    )

    # Find k nearest neighbors for each test sample
    nn_idx = np.argpartition(dists, k, axis=1)[:, :k]

    # Majority vote for each row
    preds = np.array([
        np.bincount(y_train[neighbors]).argmax()
        for neighbors in nn_idx
    ])

    return preds.reshape(-1, 1)





def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifier = learnknn(5, x_train, y_train)

    preds = predictknn(classifier, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")



def Q2a_code():
    ## Calculations
    data = np.load('mnist_all.npz')

    digits = [1, 3, 4, 6]

    # Build training sources
    train_sets = [data[f'train{d}'] for d in digits]
    train_labels = digits

    # Build full test set
    X_test = np.vstack([data[f'test{d}'] for d in digits])
    y_test = np.hstack([np.full(len(data[f'test{d}']), d) for d in digits])

    sample_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    avg_err, min_err, max_err, k_used = [], [], [], 1

    for m in sample_sizes:
        errors = []

        for _ in range(10):
            # sample m training examples
            x_train, y_train = gensmallm(train_sets, train_labels, m)

            # train classifier
            clf = learnknn(k_used, x_train, y_train.reshape(-1, 1))

            # predict
            preds = predictknn(clf, X_test).reshape(-1)

            # error
            error = np.mean(preds != y_test)
            errors.append(error)

        # statistics
        avg_err.append(np.mean(errors))
        min_err.append(np.min(errors))
        max_err.append(np.max(errors))

    ## Plotting
    plt.figure(figsize=(10,6))
    plt.errorbar(sample_sizes, avg_err,yerr=[np.array(avg_err) - np.array(min_err),np.array(max_err) - np.array(avg_err)],
        fmt='o-',
        capsize=4,
        markersize=3
    )

    plt.xlabel("Training sample size (m)", fontsize=12)
    plt.ylabel("Average test error (k=1)", fontsize=12)
    plt.title("Q2(a) — Test Error vs Training Size (k=1)", fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.show()






def Q2d_code():
    data = np.load("mnist_all.npz")

    digits = [1, 3, 4, 6]

    # Build full test set once
    X_test = np.vstack([data[f"test{d}"] for d in digits])
    y_test = np.hstack([np.full(len(data[f"test{d}"]), d) for d in digits]).astype(int)

    # Training sources
    train_sets = [data[f"train{d}"] for d in digits]
    train_labels = digits

    sample_sizes = [50, 150, 500]
    ks = list(range(1, 16))

    for m in sample_sizes:
        print(f"\nRunning experiments for m = {m}")
        avg_err_for_k = []

        for k in ks:
            errors = []

            for _ in range(30):

                # Sample m training examples
                x_train, y_train = gensmallm(train_sets, train_labels, m)
                y_train = y_train.astype(int)

                # -------------------------
                # FAST vectorized distance matrix
                # -------------------------
                x_train_sq = np.sum(x_train ** 2, axis=1)
                x_test_sq = np.sum(X_test ** 2, axis=1)
                dists = np.sqrt(
                    x_test_sq[:, None] +
                    x_train_sq[None, :] -
                    2 * (X_test @ x_train.T)
                )

                # -------------------------
                # Get k nearest neighbors
                # -------------------------
                nn_idx = np.argpartition(dists, k, axis=1)[:, :k]

                # -------------------------
                # Majority vote per test point
                # -------------------------
                preds = np.array([
                    np.bincount(y_train[row]).argmax()
                    for row in nn_idx
                ])

                # Error
                errors.append(np.mean(preds != y_test))

            avg_err_for_k.append(np.mean(errors))

        # -------------------------
        # Plotting
        # -------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(ks, avg_err_for_k, marker='o')

        plt.xlabel("k (number of neighbors)", fontsize=12)
        plt.ylabel("Average test error", fontsize=12)
        plt.title(f"Q2(d) — Test Error vs k (Training size m = {m})", fontsize=14)
        plt.grid(True)
        plt.xticks(ks)
        plt.tight_layout()
        plt.show()







def Q2e_code():
    data = np.load("mnist_all.npz")

    digits = [1, 3, 4, 6]

    # Build full test set once
    X_test = np.vstack([data[f"test{d}"] for d in digits])
    y_test = np.hstack([np.full(len(data[f"test{d}"]), d) for d in digits]).astype(int)
    #################
    x_test, y_test = Q2e_corrupt_labels(X_test, y_test)   # corrupting as instructed in Q2e

    # Training sources
    train_sets = [data[f"train{d}"] for d in digits]
    train_labels = digits

    sample_sizes = [50, 150, 500]
    ks = list(range(1, 16))

    for m in sample_sizes:
        print(f"\nRunning experiments for m = {m}")
        avg_err_for_k = []

        for k in ks:
            errors = []

            for _ in range(30):

                # Sample m training examples
                x_train, y_train = gensmallm(train_sets, train_labels, m)
                x_train , y_train = Q2e_corrupt_labels(x_train , y_train)   # corrupting as instructed in Q2e
                y_train = y_train.astype(int)

                # -------------------------
                # FAST vectorized distance matrix
                # -------------------------
                x_train_sq = np.sum(x_train ** 2, axis=1)
                x_test_sq = np.sum(X_test ** 2, axis=1)
                dists = np.sqrt(
                    x_test_sq[:, None] +
                    x_train_sq[None, :] -
                    2 * (X_test @ x_train.T)
                )

                # -------------------------
                # Get k nearest neighbors
                # -------------------------
                nn_idx = np.argpartition(dists, k, axis=1)[:, :k]

                # -------------------------
                # Majority vote per test point
                # -------------------------
                preds = np.array([
                    np.bincount(y_train[row]).argmax()
                    for row in nn_idx
                ])

                # Error
                errors.append(np.mean(preds != y_test))

            avg_err_for_k.append(np.mean(errors))

        # -------------------------
        # Plotting
        # -------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(ks, avg_err_for_k, marker='o')

        plt.xlabel("k (number of neighbors)", fontsize=12)
        plt.ylabel("Average test error (Corrupted Case)", fontsize=12)
        plt.title(f"Q2(e) — Test Error vs k (Training size m = {m})", fontsize=14)
        plt.grid(True)
        plt.xticks(ks)
        plt.tight_layout()
        plt.show()


# corrupts the labels as instructed in Q2e
def Q2e_corrupt_labels(features, labels):
    # Total number of samples
    total_samples = len(labels)

    num_corrupted_samples = int(total_samples * 0.3)

    corrupted_indices = np.random.choice(total_samples, size=num_corrupted_samples, replace=False)

    valid_labels = np.array([1 ,3 ,4 ,6])
    labels[corrupted_indices] =[np.random.choice(valid_labels[valid_labels != current_labels]) for current_labels in labels[corrupted_indices]]

    return features, labels


def small_test():
    k = 1
    x_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([1, 0, 1])
    classifier = learnknn(k, x_train, y_train)
    x_test = np.array([[10, 11], [3.1, 4.2], [2.9, 4.2], [5, 6]])
    y_testprediction = predictknn(classifier, x_test)
    print(y_testprediction)


if __name__ == '__main__':
    # small_test()
    # simple_test()
    # Q2a_code()
    # Q2d_code()
    Q2e_code()


