# ml_utils
Machine learning utils to use

Stealing away for [fast.ai](http://wiki.fast.ai/index.php/Main_Page)

[Ensemble](./examples/dogscats-ensemble.ipynb)
1. Create a model that retrains just the last layer
2. Add this to a model containing all VGG layers except the last layer
3. Fine-tune just the dense layers of this model (pre-computing the convolutional layers)
4. Add data augmentation, fine-tuning the dense layers without pre-computation.


Deal w/ underfitting:
1. Complex model
2. Remove dropout
Eg, trn_acc >> val_acc, 
removing dropout entirely

Deal w/ overfitting
1. Add more data
2. Use data augmentation
3. Use architectures that generalize well
4. Add regularization
5. Reduce architecture complexity.
6. Batch Norm
  - Adding batchnorm to a model can result in 10x or more improvements in training speed
  - Because normalization greatly reduces the ability of a small number of outlying inputs to over-influence the training, it also tends to reduce overfitting.

```
gen = image.ImageDataGenerator(rotation_range=10, width_shift_range=0.1, 
       height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, 
       channel_shift_range=10., horizontal_flip=True, dim_ordering='tf')

# Create a 'batch' of a single image
img = np.expand_dims(ndimage.imread('data/dogscats/test/7.jpg'),0)
# Request the generator to create batches from this image
aug_iter = gen.flow(img)
aug_imgs = [next(aug_iter)[0].astype(np.uint8) for i in range(8)]
plt.imshow(img[0])
mlutil.viz.plots(aug_imgs, (20,7), 2)
```