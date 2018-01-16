## Hyperparameter tunning
### Learning Rate
To randomly try lr, make sure sample lr in a log scale, the python implementation is:
```
r = -4 * np.random.rand()      // from 0.0001 to 1.
lr = np.pow(10, r)
```
## Losses
### How to customize loss function in Keras
```
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

model.compile(optimizer = "rmsprop", loss = root_mean_squared_error, 
              metrics =["accuracy"])
```
