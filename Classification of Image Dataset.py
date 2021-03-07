#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import random
from sklearn import metrics
from features import *

get_ipython().run_line_magic('matplotlib', 'inline')
#plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
from features import *

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


### Get data
import os
import re
currentPath = os.getcwd()   
datapath = os.path.join(currentPath,'weather')
images = os.listdir(datapath)  ## list the file names in the folder
pixel = 128
data = []
#data = np.empty((len(images), pixel, pixel, 3))
classes = []
dict_classes = {'cloudy': 0, 'rain': 1, 'shine': 2, 'sunrise': 3}

for img in images:
    if not img.endswith('.jpg'):
        continue
    imagePath = os.path.join(datapath, img)
    thisimage = np.array(Image.open(imagePath).resize((pixel,pixel)))
    weather = re.search('([^0-9]+)', img).group()
    if thisimage.shape != (pixel, pixel, 3):
        continue
    assert img.startswith(weather) == True
    data.append(thisimage)
    classes.append(weather)
classes = np.array(classes)


# # EDA

# In[3]:


print("The number of instances is " + str(len(data)))
print("The number of classes is " + str(len(dict_classes)) +" Which are " + str(dict_classes.keys()))

size_list =[]
for i in range (len(data)):
    size = data[i].shape
    size_list.append(size)
unique = list(set(size_list))
print("the shape of each image instance is "+str(unique))

#Checking balance of dataset
pd.Series(classes).value_counts()


# # Visualize the Data

# In[4]:


def visualize_images(examples_per_class):
    examples_per_class = examples_per_class
    im_list = []
    for weatherclass in list(dict_classes.keys()): 
        for i in range(examples_per_class): 
            n = random.choice(np.where(classes == weatherclass)[0]) 
            im_list.append(n)

    fig=plt.figure(figsize=(8, 8))
    columns = examples_per_class
    rows = len(list(dict_classes.keys()))
    for m in im_list:
        i = im_list.index(m)
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(data[m])
visualize_images(5)


# # Model Building

# ## Split your data into Train, Test and Validation set

# In[5]:


from sklearn.model_selection import train_test_split

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
    del X_train, y_train
    del X_test, y_test
    del X_val, y_val
    print('Clear previously loaded data.')
except:
    pass
X = data
y = classes
random_state = 4

def get_train_val_test(train_ratio, test_ratio, val_ratio):
    train_ratio = train_ratio
    test_ratio = test_ratio
    val_ratio = val_ratio
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,test_size=(val_ratio*len(X))/len(X_train_val))
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test(0.8, 0.1, 0.1)


# ## Extract Features

# In[6]:


# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
    del X_train_feats
    del X_val_feats
    del X_test_feats
    print('Clear previously loaded data.')
except:
    pass

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)
print("Shape of X_train_feats features", X_train_feats.shape)
print("Shape of X_val_feats features", X_val_feats.shape)
print("Shape of X_test_feats features", X_test_feats.shape)


# ## Normalize features using sklearn standardscaler 

# In[7]:


from sklearn.preprocessing import StandardScaler
# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
    del X_train_feat_norm
    del X_val_feat_norm
    del X_feat_test_norm
    print('Clear previously loaded data.')
except:
    pass

def get_normalized_features(X_train, X_val, X_test):
    X_train = X_train
    X_val = X_val
    X_test = X_test
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.fit_transform(X_val)
    X_test_norm = scaler.fit_transform(X_test)
    return X_train_norm, X_val_norm, X_test_norm
    
X_train_feat_norm, X_val_feat_norm, X_feat_test_norm = get_normalized_features(X_train_feats, X_val_feats, X_test_feats)
print("Shape of X_train_feat_norm features", X_train_feat_norm.shape)


# ## KNN model to classify the images

# In[8]:


get_ipython().run_cell_magic('time', '', 'from sklearn.neighbors import KNeighborsClassifier\nnp.random.seed(0)\nbest_knn = None\nknn_best_val = -1\n\nresults = {}\nn_neighbors = [1, 3, 5, 10]\nweights = [\'uniform\',\'distance\']\ngrid_search = [ (nn,weight) for nn in n_neighbors for weight in weights ]\n\nfor nn, weight in grid_search:\n    # Create knn model\n    knn = KNeighborsClassifier(n_neighbors = nn, weights = weight)\n    \n    # Train phase\n    knn_model_2 = knn.fit(X_train_feat_norm, y_train)\n    \n    \n    # Accuracy\n    train_accuracy = knn_model_2.score(X_train_feat_norm, y_train)\n    val_accuracy = knn_model_2.score(X_val_feat_norm, y_val)\n    \n    results[nn,weight] = (train_accuracy,val_accuracy)\n    \n    \n    # Save best model\n    if val_accuracy > knn_best_val:\n        knn_best_val = val_accuracy\n        best_knn = knn_model_2\n\nprint(\'Best validation accuracy achieved during cross-validation: {:.2%}\'.format(knn_best_val))\nprint("Paramters of best trained KNN", best_knn.get_params())')


# ## Test results of best trained KNN

# In[9]:


# Define and train the KNN with best fitted parameters
knn = KNeighborsClassifier(n_neighbors=1,weights='uniform')
knn_model_3 = knn.fit(X_train_feat_norm, y_train)

from sklearn.metrics import classification_report
# Print Accuracy and Classification Report on test data
print(knn_model_3.score(X_feat_test_norm, y_test))
y_true, y_pred = y_test, knn_model_3.predict(X_feat_test_norm)
print(classification_report(y_true, y_pred))


# ## SVM to classify the images

# In[10]:


from sklearn.svm import SVC
np.random.seed(0)
svm_best_val = -1
best_svm = None

result ={}
kernel_value = ['poly', 'linear', 'rbf']
gamma_value = ['scale', 'auto']
C_value = [1, 10, 100]
grid_search = [ (ker,gam,c) for ker in kernel_value for gam in gamma_value for c in C_value ]

for ker, gam, c in grid_search:
    # Create knn model
    svm = SVC(kernel = ker, gamma = gam, C = c)

    # Train phase
    svm_model_2 = svm.fit(X_train_feat_norm, y_train)
    
    # Accuracy
    train_accuracy = svm_model_2.score(X_train_feat_norm, y_train)
    val_accuracy = svm_model_2.score(X_val_feat_norm, y_val)
    
    results[ker,gam,c] = (train_accuracy,val_accuracy)
    
    # Save best model
    if val_accuracy > knn_best_val:
        svm_best_val = val_accuracy
        best_svm = svm_model_2

print('best validation accuracy achieved during cross-validation: {:.2%}'.format(svm_best_val))
print("Paramters of best trained SVM", best_svm.get_params())


# ## Test results of best trained SVM

# In[11]:


# define the SVM with best fitted parameters found above and fit the train data
svm = SVC(C = 100, gamma = 'auto', kernel = 'rbf')
svm_model_3 = knn.fit(X_train_feat_norm, y_train)

# Using the same SVM, visualize the confusion matrix on the test data
y_pred = svm_model_3.predict(X_feat_test_norm)
cm = metrics.confusion_matrix(y_test, y_pred)
cm[:,:]


# In[12]:


# Visualize 25 test images randomly along with their predicted values
plt.figure(figsize=(20,10))
columns = 5
rows = 5
k = 25
for i in range(1,k+1):
    n = random.choice(range(len(X_test)))
    plt.subplot(rows,columns,i)
    plt.imshow(X_test[n])
    plt.axis('off')
    plt.title(y_pred[n])


# # Visualize wrong predictions

# In[14]:


import matplotlib.gridspec as gridspec
def visualize_negative_classified_images(examples_per_class):
    cloudy_index_list = []
    rain_index_list = []
    shine_index_list = []
    sunrise_index_list = []
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            if y_test[i] == 'cloudy':
                cloudy_index_list.append(i)
            if y_test[i] == 'rain':
                rain_index_list.append(i)
            if y_test[i] == 'shine':
                shine_index_list.append(i)
            if y_test[i] == 'sunrise':
                sunrise_index_list.append(i)
    
    k = examples_per_class            
    fig2 = plt.figure(constrained_layout=True,figsize=(5, 5))
    spec2 = gridspec.GridSpec(ncols=4, nrows=k, figure=fig2)
 

    #CLOUDY
    for i in range(k):
        try:
            n = cloudy_index_list[i]
            fig2.add_subplot(spec2[i, 0])
            plt.imshow(X_test[n])
            plt.axis('off')
        except:
            pass

    #RAIN    
    for i in range(k):
        try:
            n = rain_index_list[i]
            fig2.add_subplot(spec2[i, 1])
            plt.imshow(X_test[n])
            plt.axis('off')
        except:
            pass

    #SHINE
    for i in range(k):
        try:
            n = shine_index_list[i]
            fig2.add_subplot(spec2[i, 2])
            plt.imshow(X_test[n])
            plt.axis('off')
        except:
            pass

    #SUNRISE

    for i in range(k):
        try:
            n = sunrise_index_list[i]
            fig2.add_subplot(spec2[i, 3])
            plt.imshow(X_test[n])
            plt.axis('off')
        except:
            pass
    
visualize_negative_classified_images(5)


# In[ ]:




