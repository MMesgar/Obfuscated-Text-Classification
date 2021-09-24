from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import random
import collections

def random_uniform_baseline(args, train, dev, test):
    x_train, y_train = train[0], train[1]
    x_dev, y_dev = dev[0], dev[1]
    x_test = test[0]
    pred_dev = []
    pred_test = []

    labels = list(set(y_train))
    assert(labels == [0,1,2,3,4,5,6,7,8,9,10,11])

    for i in range(len(x_dev)):
        pred_dev.append(random.choice(labels))

    for i in range(len(x_test)):
        pred_test.append(random.choice(labels))

    dev_acc = accuracy_score(y_dev, pred_dev)*100

    return dev_acc, pred_test

def majority_baseline(args, train, dev, test):
    x_train, y_train = train[0], train[1]
    x_dev, y_dev = dev[0], dev[1]
    x_test = test[0]
    pred_dev = []
    pred_test = []

    most_freq_label = collections.Counter(y_train).most_common(1)[0][0]

    pred_dev = [most_freq_label]*len(x_dev)

    pred_test = [most_freq_label] * len(x_test)

    dev_acc = accuracy_score(y_dev, pred_dev)*100

    return dev_acc, pred_test


from sklearn.linear_model import LogisticRegression

def logistic_regression(args, train, dev, test):
    x_train, y_train = train[0], train[1]
    x_dev, y_dev = dev[0], dev[1]
    x_test = test[0]
    pred_dev = []
    pred_test = []

    tfv = TfidfVectorizer(min_df=3,
                          max_features=None,
                          strip_accents='unicode',
                          analyzer='char',
                          ngram_range=(1, 3),
                          use_idf=1,
                          smooth_idf=1,
                          sublinear_tf=1)

    # Fitting TF-IDF to the data
    tfv.fit(x_train)

    xtrain_tfv =  tfv.transform(x_train)
    xdev_tfv = tfv.transform(x_dev)
    xtest_tfv = tfv.transform(x_test)

    # Fitting a simple Logistic Regression on TFIDF
    clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=500)

    clf.fit(xtrain_tfv, y_train)

    dev_logits = clf.predict_proba(xdev_tfv)
    dev_predictions = list(dev_logits.argmax(1))

    dev_acc = accuracy_score(y_dev, dev_predictions)*100

    test_logits = clf.predict_proba(xtest_tfv)
    test_predictions = list(test_logits.argmax(1))

    return dev_acc, test_predictions
