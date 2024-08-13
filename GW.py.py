
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

paths={
    'train_path':'/content/train/train/',
    'test_path':'/content/test/test/',
    'val_path':'/content/validation/validation/'
}
category=sorted(os.listdir(paths['train_path']))
print(category)

dataset_path = []
location_path=[]

for path in paths:
  for cat in category:
    location_path = os.listdir(paths[path]+cat)
    dataset_path.extend([paths[path]+cat+'/'+image for image in location_path])

len(dataset_path)

def proc_dataset(filepath):
    labels = [os.path.split(os.path.split(x)[0])[1] for x in filepath]
    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    df = pd.concat([filepath, labels], axis=1)
    df = df.sample(frac=1).reset_index(drop = True)
    return df

dataset = proc_dataset(dataset_path)
dataset.head(5)

#dataset = dataset[dataset['Label'] != 'None_of_the_Above']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# Encode the labels in the 'label' column and create a new column 'label_encoded'
dataset['label_encoded'] = le.fit_transform(dataset['Label'])

dataset.head()

# Assuming your DataFrame is named 'df'
unique_df = dataset.drop_duplicates(subset=['Label', 'label_encoded'])

unique_df.head(20)

# Count the total number of samples per category
category_counts = dataset['Label'].value_counts()
print("Total samples per category:")
print(category_counts)

IMAGE_SHAPE = 150

label=(os.path.split(os.path.split(x)[0])[1] for x in dataset_path)

from keras.preprocessing.image import ImageDataGenerator
def get_dataset(train_set, test_set, val_set):
  train_gen = ImageDataGenerator(
    rescale=1/255, horizontal_flip=True
  )
  train_data = train_gen.flow_from_dataframe(
      dataframe=train_set,
      x_col='Filepath',
      y_col='Label',
      target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
      color_mode='rgb',
      class_mode='categorical',
      batch_size=batch_size,
      shuffle=True,
      seed=123
  )
  test_gen = ImageDataGenerator(
    rescale=1/255
  )
  test_data = test_gen.flow_from_dataframe(
      dataframe=test_set,
      x_col='Filepath',
      y_col='Label',
      target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
      color_mode='rgb',
      class_mode='categorical',
      batch_size=batch_size,
      shuffle=True,
      seed=123
  )
  val_gen = ImageDataGenerator(
    rescale=1/255
  )
  val_data = val_gen.flow_from_dataframe(
    dataframe=val_set,
    x_col='Filepath',
    y_col='Label',
    target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=123
  )
  return train_data,test_data,val_data

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset,test_size=0.2, stratify=dataset.Label)
val_set, test_set = train_test_split(test_set, test_size=0.5)
len(train_set), len(test_set), len(val_set)

import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
def get_model(IMAGE_SHAPE):
    model= Sequential()

    model.add(Conv2D(filters=16, kernel_size=(3,3),activation='relu',input_shape=(IMAGE_SHAPE,IMAGE_SHAPE,3)))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.05))

    model.add(Conv2D(filters=32, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D((3,3)))
    model.add(Dropout(0.1))

    model.add(Conv2D(filters=64, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=256, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=512, kernel_size=(3,3),activation='relu'))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(22,activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer ='adam', metrics= ['accuracy'])

    return model

!pip install visualkeras
from PIL import ImageFont
import visualkeras
visualkeras.layered_view(get_model(IMAGE_SHAPE),legend=True,scale_xy=2, scale_z=2, max_z=20)

get_model(IMAGE_SHAPE).summary()

"""#Finding FLOPs"""

import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, AveragePooling2D, Flatten, Dropout, BatchNormalization

def calculate_flops(model):
    def conv2d_flops(layer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        kernel_shape = layer.kernel_size
        filters = layer.filters

        # Conv2D FLOPs: 2 * (kernel height * kernel width * input channels) * (output height * output width) * output channels
        flops = 2 * np.prod(kernel_shape) * input_shape[-1] * np.prod(output_shape[1:-1]) * filters
        return flops

    def dense_flops(layer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape

        # Dense FLOPs: 2 * (input units * output units)
        flops = 2 * np.prod(input_shape[1:]) * np.prod(output_shape[1:])
        return flops

    def maxpool_flops(layer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        pool_size = layer.pool_size

        # MaxPooling FLOPs: (output height * output width * output channels) * (pool height * pool width - 1)
        flops = np.prod(output_shape[1:]) * (np.prod(pool_size) - 1)
        return flops

    def avgpool_flops(layer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        pool_size = layer.pool_size

        # AveragePooling FLOPs: (output height * output width * output channels) * pool height * pool width
        flops = np.prod(output_shape[1:]) * np.prod(pool_size)
        return flops

    def batchnorm_flops(layer):
        input_shape = layer.input_shape

        # BatchNormalization FLOPs: 2 * input size (mean and variance calculation) + 2 * input size (scaling and shifting)
        flops = 4 * np.prod(input_shape[1:])
        return flops

    def flatten_flops(layer):
        # Flatten layer has no FLOPs
        return 0

    def dropout_flops(layer):
        input_shape = layer.input_shape

        # Dropout FLOPs: number of elements in input (comparison)
        flops = np.prod(input_shape[1:])
        return flops

    flops_dict = {}
    total_flops = 0

    for layer in model.layers:
        if isinstance(layer, Conv2D):
            flops = conv2d_flops(layer)
        elif isinstance(layer, Dense):
            flops = dense_flops(layer)
        elif isinstance(layer, MaxPooling2D):
            flops = maxpool_flops(layer)
        elif isinstance(layer, AveragePooling2D):
            flops = avgpool_flops(layer)
        elif isinstance(layer, BatchNormalization):
            flops = batchnorm_flops(layer)
        elif isinstance(layer, Flatten):
            flops = flatten_flops(layer)
        elif isinstance(layer, Dropout):
            flops = dropout_flops(layer)
        else:
            flops = 0
        flops_dict[layer.name] = flops
        total_flops += flops

    return flops_dict, total_flops

# Assuming `model` is your Keras Sequential model
flops_dict, total_flops = calculate_flops(get_model(IMAGE_SHAPE))
for layer_name, flops_value in flops_dict.items():
    print(f"Layer: {layer_name}, FLOPs: {flops_value}")
print(f"Total FLOPs: {total_flops}")

import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Dense

def calculate_flops(model):
    def conv2d_flops(layer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        kernel_shape = layer.kernel_size
        filters = layer.filters

        # Conv2D FLOPs: 2 * (kernel height * kernel width * input channels) * (output height * output width) * output channels
        flops = 2 * np.prod(kernel_shape) * input_shape[-1] * np.prod(output_shape[1:-1]) * filters
        return flops

    def dense_flops(layer):
        input_shape = layer.input_shape
        output_shape = layer.output_shape

        # Dense FLOPs: 2 * (input units * output units)
        flops = 2 * np.prod(input_shape[1:]) * np.prod(output_shape[1:])
        return flops

    flops_dict = {}
    total_flops = 0

    for layer in model.layers:
        if isinstance(layer, Conv2D):
            flops = conv2d_flops(layer)
        elif isinstance(layer, Dense):
            flops = dense_flops(layer)
        else:
            flops = 0
        flops_dict[layer.name] = flops
        total_flops += flops

    return flops_dict, total_flops

# Assuming `model` is your Keras Sequential model
flops_dict, total_flops = calculate_flops(get_model(IMAGE_SHAPE))
for layer_name, flops_value in flops_dict.items():
    print(f"Layer: {layer_name}, FLOPs: {flops_value}")
print(f"Total FLOPs: {total_flops}")


def pltaccuracy(accuracy, val_accuracy, figname):
  plt.clf()
  epochs = range(1, len(accuracy) + 1)
  plt.plot(epochs, accuracy, label='Training accuracy')
  plt.plot(epochs, val_accuracy, label='Validation accuracy')
  plt.title('Training and Validation accuracy', fontweight='bold',fontsize=15)
  plt.xlabel('Epoch', fontweight='bold',fontsize=15)
  plt.ylabel('Accuracy', fontweight='bold',fontsize=15)
  plt.legend(prop={'size':13}, loc='lower right')
  plt.savefig(figname, dpi=1200)
  plt.show()

def pltloss(loss, val_loss,figname):
  plt.clf()
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, label='Training Loss')
  plt.plot(epochs, val_loss, label='Validation Loss')
  plt.title('Training and Validation Loss', fontweight='bold',fontsize=15)
  plt.xlabel('Epoch', fontweight='bold',fontsize=15)
  plt.ylabel('Loss', fontweight='bold',fontsize=15)
  plt.legend()
  plt.savefig(figname, dpi=1200)
  plt.show()

import math
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
category=['LI','RI','AC','BL','CH','EL','HE','KF','LM','LB','LL','NG','NA','PD','PL','RB','SL','SC','TM','VM','WL','WH']
def DisplayConfusionMatrix(test_data,batch_size,model,figname):
  total = len(test_data.filenames)
  total_generator = math.ceil(total / (1.0 * batch_size))
  test_labels = []
  for_test = []
  for i in range(0,int(total_generator)):
    for_test.extend(np.array(test_data[i][0]))
    test_labels.extend(np.array(test_data[i][1]))
  int_labels = np.argmax(test_labels,axis=-1)
  test_img_1d = np.atleast_1d(for_test)
  test_pred = model.predict(test_img_1d)
  test_pred = np.argmax(test_pred,axis=-1)
  cm = confusion_matrix(int_labels, test_pred)
  # Set the size of the figure
  # Create a ConfusionMatrixDisplay object with the specified size
  #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category)
  # Plot confusion matrix
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=category, yticklabels=category, cbar=False)
  plt.xlabel('Predicted labels')
  plt.ylabel('True labels')
  plt.title('Confusion Matrix')
  # Show the plot
  plt.savefig(figname, dpi=800)
  plt.show()
  print(classification_report(y_true=int_labels,y_pred=test_pred,target_names=category))
  print("======="*15, end="\n")
  from sklearn.metrics import precision_recall_fscore_support
  res = []
  for l in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]:
     prec,recall,_,_ = precision_recall_fscore_support(np.array(int_labels)==l,
                                                  np.array(test_pred)==l,
                                                  pos_label=True,average=None)
     res.append([l,recall[0],recall[1]])
  display(pd.DataFrame(res,columns = ['class','sensitivity','specificity']))
  print("======="*15, end="\n")

  return for_test,test_labels

from sklearn.metrics import roc_curve, auc
def aucroc(for_test,test_labels,model,figname):
  y_score=model.predict(np.array(for_test))
  y_true=np.array(test_labels).astype('int')

  # Compute ROC curve and ROC area for each class
  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  for i in range(22):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  plt.figure(figsize=(10,10))
  for i in range(22):
    plt.plot(fpr[i],
             tpr[i],
             label=category[i]+" : %.2f" %(roc_auc[i]))
    plt.plot([0,1], [0,1], color='blue', linestyle='--')
    #plt.figtext()
  plt.xticks(np.arange(0.0, 1.1, step=0.1))
  plt.xlabel("False Positive Rate", fontsize=12, fontweight='bold')
  plt.yticks(np.arange(0.0, 1.1, step=0.1))
  plt.ylabel("True Positive Rate", fontsize=12, fontweight='bold')
  plt.title('ROC Curve Analysis', fontweight='bold', fontsize=12)
  plt.legend(prop={'size':13}, loc='lower right')
  plt.savefig(figname, dpi=800)
  plt.show()

n_folds=1
n_epochs=10
batch_size=16
from sklearn.model_selection import train_test_split
import tensorflow as tf
for i in range(n_folds):
  print("======="*15, end="\n")
  print("Training on Fold: ",i+1)
  train_set, test_set = train_test_split(dataset,test_size=0.2, stratify=dataset.Label)
  val_set, test_set = train_test_split(test_set, test_size=0.5)
  len(train_set), len(test_set), len(val_set)
  train_data, test_data, val_data = get_dataset(train_set, test_set, val_set)
  model=get_model(IMAGE_SHAPE)
  n=(images_dir+"model"+str(i)+".h5")
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
  history=model.fit(train_data,epochs=n_epochs,validation_data=val_data,verbose= 1, callbacks=[callback])
  model.save('model.h5')
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  print("======="*15, end="\n\n")
  n=(images_dir+"accuracy"+str(i)+".png")
  pltaccuracy(accuracy, val_accuracy,n)

  n=(images_dir+"loss"+str(i)+".png")
  pltloss(loss, val_loss,n)
  print("======="*15, end="\n")
  print("Validation Accuracy")
  model.evaluate(test_data)
  print("======="*15, end="\n")
  print("Confusion Matrix")
  n=(images_dir+"confusionmatrix"+str(i)+".png")
  for_test,test_labels = DisplayConfusionMatrix(test_data,batch_size,model,n)

  print("======="*15, end="\n")
  print("AUC ROC")
  n=(images_dir+"aucroc"+str(i)+".png")
  aucroc(for_test,test_labels,model,n)

y_score=model.predict(np.array(for_test))
y_true=np.array(test_labels).astype('int')

print(y_score)

print(y_true)

a=y_true.argmax(axis=1)
b=y_score.argmax(axis=1)

"""Mann-Whitney U Test Statistic"""

from scipy.stats import mannwhitneyu

statistic, p_val = mannwhitneyu(a, b)
print("Mann-Whitney U Test Statistic:", statistic)
print("p-value:", p_val)

"""kruskal"""

from scipy.stats import kruskal

result = kruskal(a, b)
print("Kruskal-Wallis Test Statistic:", result.statistic)
print("p-value:", result.pvalue)

from sklearn.metrics import cohen_kappa_score

# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(a, b)

print("Cohen's Kappa Score:", kappa_score)





model.save(n)

from google.colab import drive
drive.mount('/content/drive/')

model_dir='/content/drive/MyDrive/Research/Gravitational Waves/'

batch_size=16
train_data, test_data, val_data = get_dataset(train_set, test_set, val_set)

import tensorflow as tf
n=(model_dir+"model.h5")
model = tf.keras.models.load_model(n)

images,labels=next(test_data)
images.shape

imgn=10
single= np.expand_dims(images[imgn], axis=0)
pred=model.predict(single)
print(pred)
pred_single=np.argmax(pred, axis=-1)
print(pred_single)
category[pred_single[0]]
#Scattered_Light --> SC

plt.imshow(images[imgn])
plt.axis('off')  # Turn off axis labels
plt.show()

def fgsm_attack(image, epsilon):
    # Define the loss function (e.g., cross-entropy) and compute the gradient
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(1, prediction)

    # Calculate the gradient of the loss with respect to the input image
    gradient = tape.gradient(loss, image)

    # Create the perturbation (adversarial noise) using the sign of the gradient
    perturbation = epsilon * tf.sign(gradient)

    # Generate the adversarial example by adding the perturbation to the input image
    adversarial_image = image + perturbation

    # Clip the pixel values to ensure they are within the valid range (0-1)
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

    return adversarial_image

# Choose an input image from your test dataset
input_image = single  # Replace 'index' with the desired index

# Set the epsilon value to control the magnitude of the perturbation
epsilon = 0.18  # You can adjust this value

# Generate the adversarial example
adversarial_example = fgsm_attack(tf.convert_to_tensor(input_image), epsilon)

img = adversarial_example[0]

# Plot the image
plt.imshow(img)
plt.axis('off')  # Turn off axis labels
plt.show()

# Check the model's prediction on the original and adversarial examples
original_prediction = model.predict(input_image)
adversarial_prediction = model.predict(adversarial_example)
adversarial_prediction = np.array(adversarial_prediction)
# Set print options to print numbers in normal digits
np.set_printoptions(suppress=True)
# Print the array
print(adversarial_prediction*100)
print("Original Prediction:", np.argmax(original_prediction))
print("Adversarial Prediction:", np.argmax(adversarial_prediction))

import numpy as np
import matplotlib.pyplot as plt

# Assign the values to arrays
epsilon = np.array([0.01, 0.05, 0.09, 0.1, 0.15, 0.16, 0.17, 0.18, 0.19,])
accuracy = np.array([99.99924, 98.91003, 90.96633, 85.153456, 73.898445, 63.848763, 10.933959, 1.4410704, 0.3285209])

# Create a normal plot
plt.figure(figsize=(8, 6))
plt.plot(epsilon, accuracy, marker='o', linestyle='-')
plt.title('Accuracy vs Epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.xlim(0, 0.2)
plt.ylim(0, 101)
plt.show()










#!pip install lime

import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import random

explainer = lime_image.LimeImageExplainer(random_state=42)
#single= np.expand_dims(images[image_number], axis=0)
#explanation = explainer.explain_instance(images[image_number].astype('double'), model.predict)
explanation = explainer.explain_instance(ri.astype('double'), model.predict)
plt.imshow(ri)
#plt.imshow(images[image_number])
plt.axis(False)
n=(images_dir+"LIMEOriginal1.png")
plt.savefig(n, dpi=1200)

from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                            positive_only=True,
                                            num_features=2,
                                            hide_rest=True)

print(explanation.top_labels[0])
plt.imshow(mark_boundaries(temp, mask))
plt.axis(False)
n=(images_dir+"LIMEMask.png")
plt.savefig(n, dpi=1200)

from skimage.segmentation import mark_boundaries

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                            positive_only=True,
                                            num_features=20,
                                            hide_rest=True)

plt.imshow(mark_boundaries(temp, mask))
plt.axis(False)
n=(images_dir+"LIMEWithBoundary.png")
plt.savefig(n, dpi=1200)



