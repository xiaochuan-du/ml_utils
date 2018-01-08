"""
Viz func
"""
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


def to_plot(img):
    " reorder images if needed to plot"
    if K.image_dim_ordering() == 'tf':
        return np.rollaxis(img, 0, 1).astype(np.uint8)
    else:
        return np.rollaxis(img, 0, 3).astype(np.uint8)


def plot(img):
    " plot image"
    plt.imshow(to_plot(img))


def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    """ plot images with titles
    batches = vgg.get_batches(path+'train', batch_size=4)
    imgs,labels = next(batches)
    plots(imgs, titles=labels)
    """
    if isinstance(ims[0], np.ndarray):
        ims = np.array(ims).astype(np.uint8)
        if ims.shape[-1] != 3:
            ims = ims.transpose((0, 2, 3, 1))
    fig = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i, img in enumerate(ims):
        subplot = fig.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if titles is not None:
            subplot.set_title(titles[i], fontsize=16)
        plt.imshow(img, interpolation=None if interp else 'none')
