import os
import numpy as np
dir = r'H:\Dataset'

direcs = ['Batch_1','Batch_2','Batch_3']


directory = [os.path.join(dir,direc) for direc in direcs]
maxval = 0
for direct in directory:
    for folder in os.listdir(direct):
        if(not os.path.exists(os.path.join(direct,folder,'cleaned_points.npy'))):
            print(os.path.join(direct,folder,'cleaned_points.npy'))
            print('issue')

        else:
            maxval = np.max(np.load(os.path.join(direct,folder,'cleaned_points.npy')))


print(maxval)