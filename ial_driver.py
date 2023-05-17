import ial_class
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import os
import build_unet
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import *
import ants
from tqdm.auto import tqdm
from ial_class import InterActiveLearning

gpus = tf.config.list_physical_devices('GPU')

#Select a single GPU to use
tf.config.set_visible_devices(gpus[2],'GPU')
tf.config.experimental.set_memory_growth(gpus[2],True)

logical_gpus = tf.config.list_logical_devices('GPU')
print(f'logical gpus: {logical_gpus}')
print(f'Phys: {len(gpus)}\nLogical: {len(logical_gpus)}')

print(f'Print: logical_gpus: {logical_gpus}')

#with tf.device('/device:GPU:0'):
#    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#    c = tf.matmul(a, b)

#print(f'c = {c}')
path_to_files = '/rsrch1/ip/abalsells/Liver_Click_Experiment/AL_exp/'
train_pid_list = list(np.load(os.path.join(path_to_files,'train_pid_list.npy')))

#Load a small example and save results in sandbox 
pts = '/rsrch1/ip/abalsells/Liver_Click_Experiment/AL_exp/sandbox'
initial_train = train_pid_list[:100]
unlabel_pool = [p for p in train_pid_list if p not in initial_train]
unlabel_pool = unlabel_pool[:100]
ob = InterActiveLearning(path_to_files,initial_train,unlabel_pool,path_to_save=pts)


for i in range(2):
    print('.........................')
    with tf.device(logical_gpus[0]):
        ob.initialize_ML(batch_size=10,num_epochs=2)
        ob.run_ML()
    #ob.compute_dice_uncertainty()
    #ob.draw_most_uncertain(10)
    #ob.update_pools()
