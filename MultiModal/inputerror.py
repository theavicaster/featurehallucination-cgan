import scipy.io as sio
import numpy as np

np.random.seed(666)

def loadData():
   
    msdata = np.load('/home/SharedData/saurabh/HS/multispectralData.npy')
    pcdata = np.load('/home/SharedData/saurabh/HS/panchromaticData.npy')
    labels = np.load('/home/SharedData/saurabh/HS/labelsData.npy')
    
    return msdata, pcdata, labels


msdata, pcdata, gt = loadData()

print("ORIGINAL SHAPES - ")
print(msdata.shape)
print(pcdata.shape)
print(gt.shape)

msdata=np.transpose(msdata,(0,2,3,1))
pcdata=np.transpose(pcdata,(0,2,3,1))
gt=np.transpose(gt,(1,0))
gt=np.ravel(gt)

print("TRANSPOSED SHAPES - ")
print(msdata.shape)
print(pcdata.shape)
print(gt.shape)

'''
for i in range(msdata.shape[0]):

	print("reached - ", i)
	msdata[i] = (msdata[i] - msdata.mean(axis=(0, 1), keepdims=True))/(msdata.std(axis=(0,1), keepdims=True))
	#msdata[i] = msdata[i].astype(np.float32)
	pcdata[i] = (pcdata[i] - pcdata[i].mean(axis=(0,1), keepdims=True))/(pcdata.std(axis=(0,1), keepdims=True))
	#pcdata[i] = pcdata[i].astype(np.float32)

#print(msdata.dtype, pcdata.dtype)
'''

msdata = msdata.astype(np.float32)
pcdata = pcdata.astype(np.float32)

it2 = 5000
for it1 in range(0,80000,5000):

	print(it1,it2)
	msdata[it1:it2,:,:,:] = (msdata[it1:it2,:,:,:] - msdata[it1:it2,:,:,:].mean(axis=(0, 1, 2), keepdims=True))/(msdata[it1:it2,:,:,:].std(axis=(0, 1,2), keepdims=True))
	pcdata[it1:it2,:,:,:] = (pcdata[it1:it2,:,:,:] - pcdata[it1:it2,:,:,:].mean(axis=(0,1, 2), keepdims=True))/(pcdata[it1:it2,:,:,:].std(axis=(0,1,2), keepdims=True))
	it2 += 5000

print(msdata.shape, pcdata.shape)