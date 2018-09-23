###This is a script by Bradford Tuckfield.
###It does chest x-ray diagnosis for 13 diseases.
###It uses a convolutional neural network.
###It is designed to operate on a machine with some GPU RAM.
###It takes png files as input.
###It does not currently use metadata inputs.
###It uses the data from https://www.kaggle.com/nih-chest-xrays/data

# coding: utf-8

#imports first
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


de_2017 = pd.read_csv('Data_Entry_2017.csv') #This is our main dataframe

de_2017.columns = ['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID',
       'Patient Age', 'Patient Gender', 'View Position', 'OriginalImageWidth',
       'OriginalImageHeight', 'OriginalImagePixelSpacingx',
       'OriginalImagePixelSpacingy','ToDrop']

#specify the loss function to minimize.
chosen_loss_function='binary_crossentropy'

#checking that the data loaded correctly
#not necessary for main functionality
de_2017.head()
len(de_2017.columns)
de_2017.columns
de_2017.describe()
de_2017.dtypes

new_de_2017 = de_2017.copy() #Made a copy of the dataframe just to show some data analysis and the distribution of different diseases

new_de_2017.isnull().sum() #There are no null values in the dataset

#take away columns that we're not using
new_de_2017.drop(columns=['View Position','OriginalImageWidth','OriginalImageHeight','OriginalImagePixelSpacingx','OriginalImagePixelSpacingy','ToDrop'],inplace=True)

new_de_2017.columns


# Class descriptions
# There are 15 classes (14 diseases, and one for "No findings"). Images can be classified as "No findings" or one or more disease classes:
# 
# Atelectasis
# Consolidation
# Infiltration
# Pneumothorax
# Edema
# Emphysema
# Fibrosis
# Effusion
# Pneumonia
# Pleural_thickening
# Cardiomegaly
# Nodule Mass
# Hernia

#So, create a vector of every type of disease we classify
disease = ['No Finding','Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema','Fibrosis','Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule','Mass','Hernia']

len(disease)

#create 1-0 variables for each of the diseases
for i in disease:
    new_de_2017[i] = new_de_2017['Finding Labels'].apply(lambda x: 1 if i in x else 0)


#Out of ~110k data rows, we have about 60k data for which the corresponding value is "No Findings".This is the distribution
#of the rest of the diseases
for i in disease:
    print(str(i)+" "+str(sum(new_de_2017[i])))

new_de_2017.dtypes


#for plots. again, not necessary for main functionality
import matplotlib.gridspec as gs
import matplotlib.ticker as tck

#get counts of the various diseases, and other data exploration
data_ax1 = pd.melt(new_de_2017,id_vars=['Patient Gender'],value_vars=disease,var_name='Category',value_name='Count')

data1 = data_ax1.loc[data_ax1['Count']>0]

new_de_2017['No disease'] = new_de_2017['Finding Labels'].apply(lambda x:1 if 'No Finding' in x else 0)

data_ax2 = pd.melt(new_de_2017,id_vars=['Patient Gender'],value_vars=list(['No disease']),var_name='Category',value_name='Count')

data2 = data_ax2.loc[data_ax2.Count>0]


#Hernia has just has about 20-30 cases which are not good enough to predict. And about 60k cases have no disease



##Need to compare between single and multiple diseases. A patient might have more than one disease

disease_df = new_de_2017.groupby('Finding Labels').count().sort_values('Patient ID',ascending=False)

multipledisease = disease_df[['|' in i for i in disease_df.index]].copy()

multipledisease.head()

singledisease  = disease_df[['|' not in i for i in disease_df.index]]

singledisease = singledisease[['No Finding' not in i for i in singledisease.index]]

singledisease['Finding Labels'] = singledisease.index.values

multipledisease['Finding Labels'] = multipledisease.index.values

multipledisease.head()

x = multipledisease.loc[multipledisease['Patient ID']>30,['Patient ID','Finding Labels']]

for i in disease:
    x[i] = x.apply(lambda x: x['Patient ID']  if i in x['Finding Labels'] else 0, axis=1)

x.head()

y = x[x['Hernia']>0]

y

#data set without hernia cases, since we are ignoring those
disease_wh = [i for i in disease if 'Hernia' not in i]

disease_wh

#we need these imports to find the locations of the image files that we will load
import os
from glob import glob

x = os.path.join('images*','*.png')

x

image_paths = {os.path.basename(i): i for i in glob(x)}

image_paths #This is a directory where the key is the name of the image file and value is the path of the image in the directory

len(image_paths)

de_2017.shape[0]

#You can see the the number of records of data in the dataframe is equal to the number of images present in the directory

de_2017['Image Paths'] = de_2017['Image Index'].map(image_paths.get) #Mapping the paths in the dataframe

de_2017.head()

#this is one particular image that we will use for a demo
de_2017.loc[de_2017['Image Index']=='00005141_000.png',:]

disease = ['No Finding','Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema','Fibrosis','Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule','Mass','Hernia']

for i in disease:
    de_2017[i] = de_2017['Finding Labels'].apply(lambda x: 1 if i in x else 0)

de_2017.head()

len(de_2017.columns)

# We consider only those diseases which have more than 500 patients. Only Hernia has about 20-30. The rest are all above 1000

disease500 = [i for i in disease if de_2017[i].sum()>500]

len(disease500) #14 categories of diseases. Hernia not considered. Hence 13.

print('Diseases with more than 500 patients({})'.format(len(disease500)),
       [(i,int(de_2017[i].sum())) for i in disease500])

de_2017['Finding Labels'] = de_2017['Finding Labels'].map(lambda i : i.replace('No Finding', 'No Finding') ) #This is the line which replaces "No findings" with empty space

de_2017.head()

w = de_2017['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 5e-2


len(w)

w /= w.sum() 

w 

#optionally, we can weight our dataset according to the prevalence of various diseases
sample_de_2017 = de_2017.copy()#.sample(100000,weights=w)
#sample_de_2017 = de_2017.sample(50000,weights=w)


sample_de_2017.shape

sample_de_2017['Finding Labels'].value_counts().to_csv('Diseases.csv') #Working on the main dataset. The blank value is "No findings" which is replaced with empty space for data manipulation


# Splitting into Train and Test set

#Disease vector is a one-hot encoding. For example, suppose a patient is sufferring from Atelectasis, Effusion. 
#The vector will be of length 13 one each for every disease. 0 indicates the absence and 1 the presence.

sample_de_2017['disease_vector'] = sample_de_2017.apply(lambda x: [x[disease500].values],1).map(lambda x: x[0])

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(sample_de_2017,test_size=0.30,random_state=42,stratify = sample_de_2017['Finding Labels'].map(lambda x:x[:4]))

train_df.shape

test_df.shape

#Out of 110k values I am training my model on 78k and testing it on the rest
#We chose 50k records: Train - 35k, Test - 15k

# Functions to generate and transform images. Using keras with tensorflow backend

from keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (150,150) #Image size we want. You can have 128*128 too if you want or 256*256 too if you want.

idg  = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True,
                         horizontal_flip = False, vertical_flip = False,
                         height_shift_range=0.05,width_shift_range=0.1,
                         rotation_range=5,shear_range=0.1,fill_mode='reflect',zoom_range=0.2)

#ImageDataGenerator of Keras transforms, rotates the images so the model can learn the image from every angle

de_2017.columns

def flow_df(idg,df,path,y,**kwargs):
    base_dir = os.path.dirname(df[path].values[0])
    dfg = idg.flow_from_directory(base_dir,class_mode='sparse',**kwargs)
    dfg.filenames = df[path].values
    dfg.classes = np.stack(df[y].values)
    dfg.samples = df.shape[0]
    dfg.n = df.shape[0]
    dfg._set_index_array()
    dfg.directory = ''
    print('Reinserting df: {} images'.format(df.shape[0]))
    return dfg

#set training batch size. originally was 32.
train_batch_size=32

train_gen =flow_df(idg, train_df, path = 'Image Paths', y = 'disease_vector',
                   target_size = IMG_SIZE, color_mode = 'rgb', batch_size = train_batch_size)


test_gen =flow_df(idg, test_df, path = 'Image Paths', y = 'disease_vector',
                   target_size = IMG_SIZE, color_mode = 'rgb', batch_size = 1024)


test_X, test_Y = next(flow_df(idg, test_df, path = 'Image Paths', y = 'disease_vector',
                   target_size = IMG_SIZE, color_mode = 'rgb', batch_size = 1024))

train_X, train_Y = next(train_gen)


#flow_df function uses flow_from_directory() of keras which takes all the images and organizes them into batches to feed into the model


# Creating the neural network model

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU

input_shape = train_X.shape[1:]
input_shape

#We build our model on top of InceptionResNetV2 model. It will download the weights for imagenet when you will use it
#for the first time.

base_InceptionResNetV2_model = InceptionResNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

train_X.shape[1:]

dl_model = Sequential()

train_X.shape


#commented out below: previously attempted architectures

#dl_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
#dl_model.add(Activation('relu'))

#dl_model.add(MaxPooling2D(pool_size=(2,2)))

#dl_model.add(Conv2D(32,(3,3), activation='relu'))
#dl_model.add(MaxPooling2D(pool_size=(2,2)))

#dl_model.add(Conv2D(64,(3,3), activation='relu'))
#dl_model.add(MaxPooling2D(pool_size=(2,2)))

dl_model.add(base_InceptionResNetV2_model)

dl_model.add(GlobalAveragePooling2D())

dl_model.add(Dropout(0.5))

dl_model.add(Dense(512))

#there were some errors previously when using relu activation
#dl_model.add(Activation('relu'))
#dl_model.add(LeakyReLU(alpha=0.3))
#dl_model.add(Activation('LeakyReLU'))
dl_model.add(Activation('tanh'))

dl_model.add(Dropout(0.5))

dl_model.add(Dense(len(disease500), activation = 'sigmoid'))

dl_model

dl_model.summary() #This is our NN model with the following architecture



#dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['binary_accuracy','mae']) #We need to compile the model to use it. The optimizer that we are using is adam and loss is binarycrossentropy which is the standard in image classification tasks
dl_model.compile(optimizer='adam', loss=chosen_loss_function, metrics = ['binary_accuracy','mae']) #We need to compile the model to use it. The optimizer that we are using is adam and loss is binarycrossentropy which is the standard in image classification tasks

#that's all it took to build the NN model

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

#a path where we will save the optimized weights
#weight_path = "{}_weights.best.hdf5".format('xray_class')
weight_path = "{}_weights_nonb.best.hdf5".format('xray_class')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose = 1, save_best_only = True,
                            mode='min', save_weights_only = True)

#we willstop when we have hit an apparent minimum of the loss function
#early = EarlyStopping(monitor='val_loss',
#                     mode='min',
#                     patience=3)
early = EarlyStopping(monitor='val_loss',
                     mode='min',
                     patience=5)

callbacks_list = [checkpoint,early] #callbacks list is contains checkpoint which will basically show you the progress of the model in each epoch and early will stop the model if it finds out that the validation loss is not improving after X rounds

import h5py

from sklearn.metrics import roc_curve,auc

dl_model.fit_generator(train_gen,steps_per_epoch=300,validation_data=(test_X,test_Y),epochs=100,callbacks=callbacks_list)

from keras.models import model_from_json
model_json = dl_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


dl_model.load_weights(weight_path) #loading the previously saved weights

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#loaded_model.load_weights("xray_class_weights.best.hdf5")
loaded_model.load_weights(weight_path)#"xray_class_weights.best.hdf5")

loaded_model.compile(optimizer='adam', loss=chosen_loss_function, metrics = ['binary_accuracy','mae']) #We need to compile the model to use it. The optimizer that we are using is adam and loss is binarycrossentropy which is the standard in image classification tasks

de_2017[['Image Index','Image Paths']].head(50)

de_2017.head(100)

from keras.preprocessing import image

img_width, img_height = 150,150

img = image.load_img('images_001/00001265_000.png', target_size=(img_width, img_height))

np.random.seed(8675309)

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

images = np.vstack([x])

p = loaded_model.predict(images, batch_size=32)
p



#############more testing on a tiny subset of the images
tiny_test_df=sample_de_2017.head()
new_image='00001265_000.png'
#tiny_test_df['Image Index']=new_image
tiny_gen =flow_df(idg, tiny_test_df, path = 'Image Paths', y = 'disease_vector',
                   target_size = IMG_SIZE, color_mode = 'rgb', batch_size = 256)

tiny_X,tiny_Y=next(tiny_gen)

tiny_pred = dl_model.predict(tiny_X, batch_size=32, verbose=True)

print(tiny_pred)
#####################end testing on tiny subset

for i,j in zip(disease500,p[0]):
    print('%s: %2.2f%%' %(i,j*100))

disease500

#batch for final check, originally 1024
check_batch_size=100000
test_X, test_Y = next(flow_df(idg, test_df, path = 'Image Paths', y = 'disease_vector',
                   target_size = IMG_SIZE, color_mode = 'rgb', batch_size = check_batch_size))

pred_Y = dl_model.predict(test_X, batch_size=32, verbose=True)

print('printing mean and std')
pred_Y = pred_Y.astype('float')
print(np.mean(pred_Y,0))
print(np.std(pred_Y,0))

print(np.mean(test_Y,0))

#Dx is truth and PDx is predicted
for i,j,k in zip(disease500,100*np.mean(pred_Y,0),100*np.mean(test_Y,0)):
    print('%s: Dx: %2.2f%%, PDx:%2.2f%%' %(i,k,j))

fig, ax =  plt.subplots(1,1,figsize=(10,10))
for (i,j) in enumerate(disease500):
    fpr, tpr, thresholds = roc_curve(test_Y[:,i].astype(int),pred_Y[:,i])
    ax.plot(fpr,tpr,label= '%s (AUC:%0.2f)' %(j,auc(fpr,tpr)))
ax.legend()
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
fig.savefig('XRaynet_nn.png')

np.mean(pred_Y,0)

test_Y = test_Y.astype('float')
print(np.std(pred_Y,0))
print(np.std(test_Y,0))

pred_percentiles=np.percentile(pred_Y,q=range(0,100),axis=0)
range(0,100)

pred_Y[:6,:]

percentiles=pd.DataFrame(pred_percentiles)

percentiles.head()

len(percentiles.index)

percentiles.to_csv('predpercentiles.csv')

pct2=pd.read_csv('predpercentiles.csv')
pct2=pct2.iloc[:,1:]
pct2.columns=disease500

pct2.head()

colthis=disease500[2]
lookup_percentile=0.06
pct3 = pct2.loc[(pct2[colthis]-lookup_percentile).abs().argsort()[:2],colthis].index[0]
pct3

for i in range(0,len(disease500)):
    colthis=disease500[i]
    lookup_percentile=p[0][i]
    toprint=pct2.loc[(pct2[colthis]-lookup_percentile).abs().argsort()[:2],colthis].index[0]
    print(disease500[i],' ',toprint)

p

pred_Y[:6]

predypd=pd.DataFrame(pred_Y)
predypd.to_csv('predy.csv')


