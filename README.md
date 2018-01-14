# ml_utils
Machine learning utils to use

Stealing away for [fast.ai](http://wiki.fast.ai/index.php/Main_Page)

[Ensemble](./examples/dogscats-ensemble.ipynb)
1. Create a model that retrains just the last layer
2. Add this to a model containing all VGG layers except the last layer
3. Fine-tune just the dense layers of this model (pre-computing the convolutional layers)
4. Add data augmentation, fine-tuning the dense layers without pre-computation.
Also check [mnist ensemble](./examples/mnist.ipynb)

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

From possiblity to char:
[chars[np.argmax(o)] for o in p]

Train RNN:
Use identity to init GRU/ LSTM

[Char level RNN](./examples/char-rnn.ipynb)
Use two layers LSTM to predict next words or letters.


Take advantage of data leakage:
Add bounding box to enhance learning, please [check](./examples/lesson7.ipynb)

For Fully convolutional net, it is possible to viz weight distribution:
```
l = lrg_model.layers
conv_fn = K.function([l[0].input, K.learning_phase()], l[-4].output)
def get_cm(inp, label):
    conv = conv_fn([inp,0])[0, label]
    return scipy.misc.imresize(conv, (360,640), interp='nearest')
inp = np.expand_dims(conv_val_feat[0], 0)
np.round(lrg_model.predict(inp)[0],2)
plt.imshow(to_plot(val[0]))
cm = get_cm(inp, 0)
plt.imshow(cm, cmap="cool")
```

Obtain the output of an intermediate layer?
```
1. 
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer("layer_name").output)
intermediate_output = intermediate_layer_model.predict(data)

2. 
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output]) # learning_phase is to tell is trn or test

# output in test mode = 0
layer_output = get_3rd_layer_output([x, 0])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([x, 1])[0]
```

Pseudo-labeling is able to enhance perf in game:
```
preds = model.predict([conv_test_feat, test_sizes], batch_size=batch_size*2)
gen = image.ImageDataGenerator()
test_batches = gen.flow(conv_test_feat, preds, batch_size=16)
val_batches = gen.flow(conv_val_feat, val_labels, batch_size=4)
batches = gen.flow(conv_feat, trn_labels, batch_size=44)
mi = MixIterator([batches, test_batches, val_batches])
bn_model.fit_generator(mi, mi.N, nb_epoch=8, validation_data=(conv_val_feat, val_labels))
```