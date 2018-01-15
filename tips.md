## Hyperparameter tunning
### Learning Rate
To randomly try lr, make sure sample lr in a log scale, the python implementation is:
```
r = -4 * np.random.rand()      // from 0.0001 to 1.
lr = np.pow(10, r)
```
