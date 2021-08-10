# Librosa_embedded_mfcc
procedures from Librosa library for mfcc embeddings calculation of sound files for Raspberry Pi 64bit OS

# Problem why this part of Librosa have made
Librosa can't be installed (now 2021-08-10) for Raspberry 64-bit OS (numba and lliblvm problems).
But with pytorch model  we need to made MFCC embeddings which is used for inference and sound classification.

# Dependencies need to be installed
```
#dependencies for custom librosa procedures
pip3 install soundfile
pip3 install audioread
pip3 install samplerate
pip3 install soxr
```

# Example of use
```
'''make a class prediction for one row of data'''
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

model = torch.load('torch_MFCC.pt')

'''make a single prediction'''
name_arr = ['event', 'hairdryer', 'sinkwater', 'event', 'hairdryer', 'sinkwater', 'event', 'hairdryer', 'sinkwater', 'event', 'hairdryer', 'sinkwater',]
for i in range(10):
    time_start =  time.time()
    y,sr=load('/home/pi/torch_python/' + str(name_arr[i]) + '.wav', sr=None)
    print(sr)
    mfccs = mfcc(y, sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T,axis=0)
    mfccs_mean = np.reshape(mfccs_mean, (1,40,1))
    #print(mfccs_mean)
    yhat = predict(mfccs_mean, model)
    print(time.time()-time_start)
    print('Predicted: '+str(yhat))
```
    
