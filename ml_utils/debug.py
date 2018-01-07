"""
debug helpers

"""
import itertools
import numpy as np
from keras.preprocessing import image
from ml_utils.viz import plots
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, 
        title='Confusion matrix', cmap=plt.cm.get_cmap('Blue')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    eg:

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(val_classes, preds)
    plot_confusion_matrix(cm, val_batches.class_indices)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plots_idx(idx, path, filenames, titles=None):
    """ plots_idx
    # We want both the classes...
    preds = lm.predict_classes(val_features, batch_size=batch_size)
    # ...and the probabilities of being a cat
    probs = lm.predict_proba(val_features, batch_size=batch_size)[:,0]
    #1. A few correct labels at random
    n_view = 4
    correct = np.where(preds==val_labels[:,1])[0]
    idx = permutation(correct)[:n_view]
    plots_idx(idx, probs[idx])

    #3. The images we most confident were cats, and are actually cats
    correct_cats = np.where((preds==0) & (preds==val_labels[:,1]))[0]
    most_correct_cats = np.argsort(probs[correct_cats])[::-1][:n_view]
    plots_idx(correct_cats[most_correct_cats], probs[correct_cats][most_correct_cats])

    #5. The most uncertain labels (ie those with probability closest to 0.5).
    most_uncertain = np.argsort(np.abs(probs-0.5))
    plots_idx(most_uncertain[:n_view], probs[most_uncertain])
    """
    plots([image.load_img(path + 'valid/' + filenames[i]) for i in idx], titles=titles)