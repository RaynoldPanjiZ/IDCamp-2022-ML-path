#!/usr/bin/env python
# coding: utf-8

# # Image Classification Model Deployment - CNN

# **Raynold Panji Zulfiandi**
# 
# > Emotion Detection
# 
# > Dataset: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

# In[2]:


# %tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)


# In[3]:


# cek penggunaan GPU
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found')
else:
  print('Found GPU at: {}'.format(device_name))


# # Data Preparation

# ### Download Dataset (Google collbs)

# In[6]:


get_ipython().system('pip install -q kaggle')
from google.colab import files 

# upload kaggle.json
from google.colab import files
files.upload()


# In[7]:


get_ipython().system('mkdir ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[8]:


get_ipython().system('kaggle datasets download -d ananthu017/emotion-detection-fer')


# In[9]:


# !rm -rf datasets
get_ipython().system('ls')


# In[10]:


get_ipython().system('mkdir datasets')
get_ipython().system('unzip -q emotion-detection-fer.zip -d datasets')
get_ipython().system('ls datasets')


# ### Download Datasets (kaggle notebook)

# In[4]:


# !rm -rf datasets
get_ipython().system('ls -al')


# In[5]:


get_ipython().system('mkdir ./datasets')
get_ipython().system('cp -r ../input/emotion-detection-fer/* ./datasets')
get_ipython().system('ls -al datasets')


# In[6]:


get_ipython().system('chmod 777 ./datasets/train')
get_ipython().system('chmod 777 ./datasets/test')

get_ipython().system('ls datasets/test')


# # Data Cleansing

# In[7]:


import os

TRAINING_DIR = 'datasets/train/'
VALIDATION_DIR = 'datasets/test/'

os.listdir(TRAINING_DIR), os.listdir(VALIDATION_DIR)


# In[8]:


## cek jumlah dataset
def jum_data():
  train = []
  val = []
  lists = os.listdir(TRAINING_DIR)
  for cls in lists:
    train.append(len(os.listdir(os.path.join(TRAINING_DIR, cls))))
  for cls in lists:
    val.append(len(os.listdir(os.path.join(VALIDATION_DIR, cls))))
  return train, val, lists

def cek_data():
  chs = []
  train, val, lists = jum_data()
  
  msg="\n======================================"
  for i, cls in enumerate(lists):
    tot = train[i]+val[i]
    ch = round(tot*0.8)
    chs.append(ch-train[i])
    percen = ": "+str(80)+"% dari total ✓✓" if (ch==train[i]) else "-"
    msg=msg+f"\ntotal {cls}\t: {train[i]} ===> {ch} {percen}"
  msg=msg+"\ntotal : "+str(sum(train))
  
  msg=msg+"\n======================================"
  for i, cls in enumerate(lists):
    tot = train[i]+val[i]
    percen = ": "+str(20)+"% dari total ✓✓" if (round(tot*0.2)==val[i]) else "-"
    msg=msg+f"\ntotal {cls}\t: {val[i]} ===> {round(tot*0.2)} {percen}"
  msg=msg+"\ntotal : "+str(sum(val))

  msg=msg+"\n======================================"
  return msg, chs

print(cek_data()[0])
print(cek_data()[1])


# In[9]:


## hapus folder disgusted
import shutil
shutil.rmtree(os.path.join(TRAINING_DIR, 'disgusted') )
shutil.rmtree(os.path.join(VALIDATION_DIR, 'disgusted'))

os.listdir(TRAINING_DIR), os.listdir(VALIDATION_DIR)
print(cek_data()[0])
print(cek_data()[1])


# In[10]:


## pindahkan train test gambar agar sesuai kriteria 80/20

import random 

for i, dir in enumerate(jum_data()[2]):
  source = None
  dest = None

  if (cek_data()[1][i])<0:
    source = os.path.join(TRAINING_DIR, dir)
    dest = os.path.join(VALIDATION_DIR, dir)
  elif cek_data()[1][i]>0:
    source = os.path.join(VALIDATION_DIR, dir)
    dest = os.path.join(TRAINING_DIR, dir)
  
  if source==None:
    continue
  print("\n"+source+" ====> "+dest)
  files = os.listdir(source)

  for file_name in random.sample(files, abs(cek_data()[1][i])):
    shutil.move(os.path.join(source, file_name), os.path.join(dest, "mov_"+file_name))
    print(file_name+" moved")


# In[11]:


print(cek_data()[0])
print(cek_data()[1])


# In[12]:


## Undersampling class happy

avg_undersampling = (sum(jum_data()[0]) / len(jum_data()[0])) / 7191  # persentase 0.6553

train_happy = 7191 - round(7191 * avg_undersampling)  # total train - 65,53% dari total train data: 4712
val_happy = 1798 - round(1798 * avg_undersampling)    # total val - 65,53% dari total val data: 1178

train_dir = os.path.join(TRAINING_DIR, 'happy')
val_dir = os.path.join(VALIDATION_DIR, 'happy')

for i, file_name in enumerate(random.sample(os.listdir(train_dir), train_happy)):
  os.remove(os.path.join(train_dir, file_name))
print(str(train_happy)+" files removed ")

for i, file_name in enumerate(random.sample(os.listdir(val_dir), val_happy)):
  os.remove(os.path.join(val_dir, file_name))
print(str(val_happy)+" files removed ")


# In[13]:


print(cek_data()[0])
print(cek_data()[1])


# In[14]:


## Undersampling class sad, happy, dan neutral
train_sampling = ((4958 + 4862 + 4712) // 3) - (3962 + 3202 + 4097) // 3  # rata2 data train yg besar - rata2 train yg kecil: 1091
val_sampling = ((1240 + 1215 + 1178) // 3) - (991 + 800 + 1024) // 3    # rata2 data val yg besar - rata2 val yg kecil: 273

# remove sebagian data pada data happy, neutral, dan sad
for u_dir in ['happy', 'neutral', 'sad']:
  train_dir = os.path.join(TRAINING_DIR, u_dir)
  val_dir = os.path.join(VALIDATION_DIR, u_dir)
  for i, file_name in enumerate(random.sample(os.listdir(train_dir), round(train_sampling))):
    os.remove(os.path.join(train_dir, file_name))
    continue
  print(f"{i+1} train {u_dir} files removed ")
  for i, file_name in enumerate(random.sample(os.listdir(val_dir), round(val_sampling))):
    os.remove(os.path.join(val_dir, file_name))
    continue
  print(f"{i+1} val {u_dir} files removed ")


# In[15]:


print(cek_data()[0])
print(cek_data()[1])


# In[16]:


source = os.path.join(TRAINING_DIR, 'sad')
dest = os.path.join(VALIDATION_DIR, 'sad')
file = random.sample(os.listdir(source), 1)[0]

shutil.move(os.path.join(source, file), os.path.join(dest, "mov_"+file))
# print(os.path.join(source, file))

print(cek_data()[0])
print(cek_data()[1])


# In[17]:


## backup datasets
get_ipython().system('mkdir ./datasets/backup')
get_ipython().system('cp -r ./datasets/{train,test} ./datasets/backup')
get_ipython().system('ls -al ./datasets/backup')


# In[18]:


## hapus class surprised dan fearful
shutil.rmtree(os.path.join(TRAINING_DIR, 'surprised') )
shutil.rmtree(os.path.join(VALIDATION_DIR, 'surprised'))

shutil.rmtree(os.path.join(TRAINING_DIR, 'neutral'))
shutil.rmtree(os.path.join(VALIDATION_DIR, 'neutral'))

shutil.rmtree(os.path.join(TRAINING_DIR, 'fearful'))
shutil.rmtree(os.path.join(VALIDATION_DIR, 'fearful'))

print(cek_data()[0])
print(cek_data()[1])


# In[19]:


## Undersampling class angry
train_sampling = (sum(jum_data()[0]) / len(jum_data()[0]))*0.25
val_sampling = (sum(jum_data()[1]) / len(jum_data()[1]))*0.25

for u_dir in ['angry', 'sad', 'happy']:
  train_dir = os.path.join(TRAINING_DIR, u_dir)
  val_dir = os.path.join(VALIDATION_DIR, u_dir)
  for i, file_name in enumerate(random.sample(os.listdir(train_dir), round(train_sampling))):
    os.remove(os.path.join(train_dir, file_name))
    continue
  print(f"{round(train_sampling)} train {u_dir} files removed ")
  for i, file_name in enumerate(random.sample(os.listdir(val_dir), round(val_sampling))):
    os.remove(os.path.join(val_dir, file_name))
    continue
  print(f"{round(val_sampling)} val {u_dir} files removed ")

print(cek_data()[0])
print(cek_data()[1])


# In[20]:


source = os.path.join(TRAINING_DIR, 'happy')
dest = os.path.join(VALIDATION_DIR, 'happy')
file = random.sample(os.listdir(source), 1)[0]

shutil.move(os.path.join(source, file), os.path.join(dest, "mov_"+file))
# print(os.path.join(source, file))


print(cek_data()[0])
print(cek_data()[1])


# # Data Preprocessing

# In[21]:


## Augmentasi data

from keras.preprocessing.image import ImageDataGenerator
training_datagen = ImageDataGenerator(
    rescale = 1./255.,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip = True,
    fill_mode='nearest',
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,  
)

img_size = 128
batch_size = 45

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR, 
    class_mode='categorical', 
    target_size=(img_size, img_size), 
    batch_size=batch_size,
#     color_mode='grayscale'
)

validation_generator = validation_datagen.flow_from_directory(      
    VALIDATION_DIR,    
    class_mode='categorical',
    target_size=(img_size, img_size),
    batch_size=batch_size,
#     color_mode='grayscale'
)


# In[22]:


train_generator.class_indices


# In[23]:


## Plot gambar

import numpy as np
import matplotlib.pylab as plt

plt.figure(figsize=[15,15])
for i in range(12):
    x, y = random.choice(train_generator)
    plt.subplot(3, 4, i+1)
    for im, lb in zip(x, y):
        plt.title(str(lb))
#         plt.imshow(im.reshape(img_size, img_size))
        plt.imshow(im)
        plt.axis('off')
plt.show()


# # Training

# In[38]:


# conv_base = tf.keras.applications.ResNet152V2(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
conv_base = tf.keras.applications.InceptionResNetV2(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')

conv_base.summary()


# In[49]:


## Frezze sebagian layer
# for layer in conv_base.layers[:-1]:
#   layer.trainable = False


## build architecture
num_cls = len(train_generator.class_indices)

model = tf.keras.models.Sequential([
  conv_base,

#   tf.keras.layers.Conv2D(512,(3,3), padding="same", activation="relu"),
  tf.keras.layers.Conv2D(512,(2,2), padding="same", activation="relu"),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.MaxPooling2D(2,2),
  
  tf.keras.layers.Flatten(),
  
#   tf.keras.layers.Dense(512, activation="relu", use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(l=0.01)),
  tf.keras.layers.Dense(512, activation="relu", use_bias=True),
  tf.keras.layers.Dropout(0.5),
    
  tf.keras.layers.Dense(128, activation="relu", use_bias=True),
  tf.keras.layers.Dropout(0.5),

  # tf.keras.layers.Dense(128, activation="relu", use_bias=True),
  # tf.keras.layers.Dropout(0.2),
  
  tf.keras.layers.Dense(num_cls, activation="softmax")
])

model.summary()


# In[50]:


## define callbacks

early_stopping = tf.keras.callbacks.EarlyStopping(
  monitor = "val_accuracy",
  patience = 10,
  verbose = 0,
  mode = "auto",
  restore_best_weights=True
)

callbacks = [early_stopping]


# In[51]:


## compile model
epochs = 100
lr = 0.01
decay_rate = lr / epochs
momentum=0.6

opt_adam = tf.optimizers.Adam(learning_rate=lr)
opt_rms = tf.optimizers.RMSprop(learning_rate=lr)
opt_sgd_par = tf.optimizers.SGD(learning_rate=lr, decay=decay_rate, momentum=momentum)
opt_sgd = tf.optimizers.SGD(learning_rate=lr)

model.compile(
  loss = 'categorical_crossentropy',
  optimizer = opt_sgd_par,
  metrics = ['accuracy']
)


## Train model

STEP_PER_EPOCH = train_generator.n // train_generator.batch_size
VALIDATION_STEPS = validation_generator.n // validation_generator.batch_size

with tf.device(device_name):
  history = model.fit(
      train_generator,
      steps_per_epoch = STEP_PER_EPOCH,
      epochs = epochs,
      validation_data = validation_generator,
      validation_steps = VALIDATION_STEPS,
      verbose = 1,
      callbacks = callbacks
  )


# In[52]:


## model evaluate

loss, acc = model.evaluate(validation_generator)
print(f"valid accuracy: {acc} \nvalid loss: {loss}")


# # Evaluation Model

# In[53]:


## Plot accuracy dan Loss

loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss, label='Training set')
plt.plot(val_loss, label='Validation set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)

plt.subplot(1, 2, 2)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(acc, label='Training set')
plt.plot(val_acc, label='Validation set', linestyle='--')
plt.legend()
plt.grid(linestyle='--', linewidth=1, alpha=0.5)

plt.show()


# In[54]:


## Plot Confusion Matrix dan Classification Report

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

labels_list = list(train_generator.class_indices.keys())


fig, ax = plt.subplots(figsize=(18, 6))
cm = confusion_matrix(validation_generator.classes, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_list)
disp.plot(cmap=plt.cm.Greens, ax=ax)
plt.title("==== Confusion Matrix ===== \n")
plt.show()


print("\n=============== Classification Report ================")
print(classification_report(validation_generator.classes, y_pred, target_names=labels_list))


# # Deployment

# In[55]:


## save model keras *.h5

if os.path.exists('model')==False:
  os.mkdir('model')

model.save_weights("model/model_weights.h5")
model.save("model/model.h5")


# In[56]:


import warnings
warnings.filterwarnings('ignore')

## Convert Model keras ke tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

## save model *.tflite
with open('model/model.tflite', 'wb') as f:
  f.write(tflite_model)


# In[ ]:




