import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random

fashion_test_df = pd.read_csv('fashion-mnist_test.csv',sep=',' )
fashion_train_df = pd.read_csv('fashion-mnist_train.csv',sep=',' )


fashion_test_df.head()

training = np.array(fashion_train_df, dtype= 'float32')

testing = np.array(fashion_test_df, dtype= 'float32')

i= random.randint(1, 60000)

plt.imshow(training[i, 1:].reshape(28,28))

label = training[i, 0]
label

w_grid = 15
L_grid = 15

fig, axes = plt.subplot(L_grid, w_grid, figsize=(17,17))

axes = axes.ravel()

n_training= len(training)

for i in np.arange(0, w_grid*L_grid):
    index =  np.random.radint(0,  n_training)
    axes[i].imshow(training[index,1:].reshape(28,28))
    axes[i].set_title(training[index,0], fontsize=0)
    axes[i].axis('off')
    
plt.subplot_adjust(hspace=0.4)




















