#import librari numpy
import numpy as np

#deklarasi variabel dengan matriks ordo 6x10 
inputs = [[1.1, 1.2, 2.2, 2.3, 3.3, 3.4, 4.4, 4.5, 5.5, 5.6],
          [2.6, 7.5, 1.8, 0.9, 6.0, 7.6, 8.0, 9.9, 0.9, 6.9],
          [2.5, 3.0, 4.0, 8.0, 2.8, 2.9, 8.3, 4.4, 1.5, 1.1],
          [3.3, 1.7, 1.9, 1.5, 2.7, 2.0, 7.0, 3.0, 0.7, 2.8],
          [4.2, 0.9, 1.9, 1.4, 1.1, 0.2, 0.7, 0.3, 9.6, 1.7],
          [8.8, 0.3, 0.9, 0.3, 0.7, 2.0, 7.0, 3.0, 5.0, 1.3]]

#deklarasi bobot per neuron pada layer 1
weightsSatu = [[1.0, 2.0, 3.3, 4.1, 5.2, 6.9, 7.0, 9.0, 1.1, 8.7],
           [5.0, 8.0, 1.0, 6.0, 7.0, 0.1, 2.0, 3.7, 4.1, 2.0],
           [2.0, 6.0, 8.5, 0.5, 1.9, 1.8, 0.2, 2.6, 2.4, 1.8],
           [4.0, 9.0, 1.3, 1.9, 2.4, 8.0, 2.0, 4.0, 8.0, 1.6],
           [1.0, 6.7, 2.1, 1.8, 8.9, 9.9, 2.7, 0.2, 1.8, 3.0]]

#bias per neuron layer 1
biasSatu = [2.2, 3.3, 4.4, 5.5, 0.1]

#deklarasi bobot per neuron layer 2
weightsDua = [[3.5, 2.4, 0.1, 4.8, 2.9,],
            [3.6, 2.5, 0.2, 4.9, 6.0],
            [3.7, 2.6, 0.3, 5.0, 0.9]]

#bias per neuron layer 2
biasDua = [3.0, 6.0, 9.0]

#ouputs dengan menggunakan metode numpy
layerSatu_outputs = np.dot(inputs, np.array(weightsSatu).T) + biasSatu
layerDua_outputs = np.dot (layerSatu_outputs, np.array(weightsDua).T) + biasDua

#print ouputs
print(layerDua_outputs)

#develop @mahendrarendi
