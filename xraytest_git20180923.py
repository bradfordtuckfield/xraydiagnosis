#This is a script by Bradford Tuckfield.
#It does demos for a previously generated neural net for x-ray diagnosis

#imports
import keras.backend.tensorflow_backend
if keras.backend.tensorflow_backend._SESSION:
   import tensorflow as tf
   tf.reset_default_graph() 
   keras.backend.tensorflow_backend._SESSION.close()
   keras.backend.tensorflow_backend._SESSION = None
import numpy as np
import h5py
from keras.models import model_from_json
from keras.preprocessing import image
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

import os
from glob import glob

np.random.seed(8675309)

#images to look at
new_image='00003923_006.png'
new_image='00001265_000.png'

#open the NN model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
print('here1')

#load the model weights
loaded_model.load_weights("xray_class_weights_nonb.best.hdf5")
loaded_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['binary_accuracy','mae'])

img_width, img_height = 150, 150
IMG_SIZE = (150,150)

#the diseases we are testing for
diseases = ['No Finding','Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
            'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
            'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass']

disease500=diseases
disease=diseases


#loading the images
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

de_2017 = pd.read_csv('Data_Entry_2017.csv') #This is the dataframe from https://www.kaggle.com/nih-chest-xrays/data
de_2017.columns = ['Image Index', 'Finding Labels', 'Follow-up #', 'Patient ID',
       'Patient Age', 'Patient Gender', 'View Position', 'OriginalImageWidth',
       'OriginalImageHeight', 'OriginalImagePixelSpacingx',
       'OriginalImagePixelSpacingy','ToDrop']

#loop for diagnosing images, one at a time
while new_image!='stop':
    print("Enter the image path:")
    new_image = input()
    #samplede2017 preparation

    de_2017['Image Index']=new_image

    x = os.path.join('images*','*.png')
    image_paths = {os.path.basename(i): i for i in glob(x)}
    de_2017['Image Paths'] = de_2017['Image Index'].map(image_paths.get) #Mapping the paths in the dataframe

    for i in disease:
        de_2017[i] = de_2017['Finding Labels'].apply(lambda x: 1 if i in x else 0)
    sample_de_2017 = de_2017.copy()
    sample_de_2017['disease_vector'] = sample_de_2017.apply(lambda x: [x[disease500].values],1).map(lambda x: x[0])

    #end samplede2017 preparation

    idg  = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True,
                             horizontal_flip = False, vertical_flip = False,
                             height_shift_range=0.05,width_shift_range=0.1,
                             rotation_range=5,shear_range=0.1,fill_mode='reflect',zoom_range=0.2)


    #putting the image into a tiny dataset for prediction
    tiny_test_df=sample_de_2017.head(2).reset_index(drop=True)
    tiny_test_df.loc[0,'Image Index']=''
    #tiny_test_df=tiny_test_df.loc[0,:]
    tiny_test_df.to_csv('tiny_test_df.csv')
    tiny_gen =flow_df(idg, tiny_test_df, path = 'Image Paths', y = 'disease_vector',
                       target_size = IMG_SIZE, color_mode = 'rgb', batch_size = 256)

    tiny_X,tiny_Y=next(tiny_gen)

    tiny_pred = loaded_model.predict(tiny_X, batch_size=32, verbose=True)
    tiny_pred=tiny_pred[0]
    print(tiny_pred)

    #for p in tiny_pred:
    for i, j in zip(diseases, tiny_pred):
        print('%s: %2.2f%%' % (i, j*100))

    #doing percentile adjustment to deal with bias from sample sizes
    pct2=pd.read_csv('predpercentiles.csv')
    pct2=pct2.iloc[:,1:]

    pct2.columns=disease500

    for i in range(0,len(disease500)):
        colthis=disease500[i]
        lookup_percentile=tiny_pred[i]
        toprint=pct2.loc[(pct2[colthis]-lookup_percentile).abs().argsort()[:2],colthis].index[0]
        print(disease500[i],' ',toprint)
