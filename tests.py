import numpy as np
import os
file_paths = []
labels = []
spoken_word = []
for f in os.listdir('audio'):
    for w in os.listdir('audio/' + f):  
        file_paths.append('audio/' + f + '/' + w)
        labels.append(f)
        if f not in spoken_word:
            spoken_word.append(f)
print('List of spoken words:', spoken_word)
print set(labels)

from scipy.io import wavfile

data = np.zeros((len(file_paths), 32000))
maxsize = -1
for n,file in enumerate(file_paths):
    _, d = wavfile.read(file)
    data[n, :d.shape[0]] = d
    if d.shape[0] > maxsize:
        maxsize = d.shape[0]
data = data[:, :maxsize]
print('Number of total files:', data.shape[0])
all_labels = np.zeros(data.shape[0])
for n, l in enumerate(set(labels)):
    all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n
    
print('Labels and label indices', all_labels)
print data.shape

import python_speech_features as speech
all_obs = np.zeros([500, 13, 43])
for n,file in enumerate(file_paths):
    all_obs[n,]=speech.mfcc(data[n,:]).T
print all_obs.shape

'''import scipy

def stft(x, fftsize=64, overlap_pct=.5):
    hop = int(fftsize * (1 - overlap_pct))
    w = scipy.hanning(fftsize + 1)[:-1]    
    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
    return raw[:, :(fftsize // 2)]

from numpy.lib.stride_tricks import as_strided


def peakfind(stft_data, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
    window_size = l_size + r_size + c_size
    shape = stft_data.shape[:-1] + (stft_data.shape[-1] - window_size + 1, window_size)
    strides = stft_data.strides + (stft_data.strides[-1],)
    xs = as_strided(stft_data, shape=shape, strides=strides)
    def is_peak(stft_data):
        centered = (np.argmax(data) == l_size + int(c_size/2))
        l = stft_data[:l_size]
        c = stft_data[l_size:l_size + c_size]
        r = stft_data[-r_size:]
        passes = np.max(c) > np.max([f(l), f(r)])
        if centered and passes:
            return np.max(c)
        else:
            return -1
    r = np.apply_along_axis(is_peak, 1, xs)
    top = np.argsort(r, None)[::-1]
    heights = r[top[:n_peaks]]
    top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)
    return heights, top[:n_peaks]

all_obs = []
for i in range(data.shape[0]):
    d = np.abs(stft(data[i, :]))
    n_dim = 5
    obs = np.zeros((n_dim, d.shape[0]))
    for r in range(d.shape[0]):
        _, t = peakfind(d[r, :], n_peaks=n_dim)
        obs[:, r] = t.copy()
    if i % 50 == 0:
        print("Processed observation %s" % i)
    all_obs.append(obs)
    
all_obs = np.atleast_3d(all_obs)
print all_obs.shape '''


import scipy.stats as st
import numpy as np
from hmm.continuous.GMHMM import GMHMM
#from hmm.discrete.DiscreteHMM import DiscreteHMM
import numpy

class Predict(GMHMM):
    def __init__(self,n,m,d=1,A=None,means=None,covars=None,w=None,pi=None,min_std=0.01,init_type='uniform',precision=numpy.double,verbose=False):
        GMHMM.__init__(self,n,m,d,A,means,covars,w,pi,min_std,init_type,precision,verbose)

    def trainModel(self, obs):
        pi = numpy.array([0.2, 0.2, 0.2, 0.2, 0.2])
        A = numpy.ones((self.n,self.n),dtype=numpy.double)/float(self.n)

        w = numpy.ones((self.n,self.m),dtype=numpy.double)
        means = numpy.ones((self.n,self.m,self.d),dtype=numpy.double)
        covars = [[ numpy.matrix(numpy.eye(self.d,self.d)) for j in xrange(self.m)] for i in xrange(self.n)]
        n_iter = 20
        '''w[0][0] = 0.5
        w[0][1] = 0.5
        w[1][0] = 0.5
        w[1][1] = 0.5    
        means[0][0][0] = 0.5
        means[0][0][1] = 0.5
        means[0][1][0] = 0.5    
        means[0][1][1] = 0.5
        means[1][0][0] = 0.5
        means[1][0][1] = 0.5
        means[1][1][0] = 0.5    
        means[1][1][1] = 0.5 '''

        gmmhmm = GMHMM(self.n,self.m,self.d,A,means,covars,w,pi,init_type='user',verbose=True)

        print "Doing Baum-welch"
        #gmmhmm.train(obs,10)
        if len(obs.shape) == 2:
            gmmhmm.train(obs)
            return self

        elif len(obs.shape) == 3:
            count = obs.shape[0]
            for n in range(count):
                gmmhmm.train(obs[n, :, :])
                return self

    def testModel(self, obs):
        if len(obs.shape) == 2:
            log_likelihood, _ = self.forwardbackward(obs)
            return log_likelihood
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            out = np.zeros((count,))
            for n in range(count):
                log_likelihood, _ = self.forwardbackward(obs[n, :, :])
                out[n] = log_likelihood
            return out

from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(all_labels, test_size=0.1, random_state=0)

for n,i in enumerate(all_obs):
    all_obs[n] /= all_obs[n].sum(axis=0)
    

for train_index, test_index in sss:
    X_train, X_test = all_obs[train_index, ...], all_obs[test_index, ...]
    y_train, y_test = all_labels[train_index], all_labels[test_index]
ys = set(all_labels)

'''We need to make a model for each word.'''

ms = [Predict(5, 5, 43) for y in ys]
_ = [model.trainModel(X_train[y_train == y, :, :]) for model, y in zip(ms, ys)]
ps = [model.testModel(X_test) for model in ms]
res = np.vstack(ps)
predicted_label = np.argmax(res, axis=0)
#dictionary = ['nine', 'seven', 'six', 'two', 'eight', 'five', 'three', 'zero', 'four', 'one']
dictionary = ['kiwi', 'apple', 'peach', 'pineapple', 'orange', 'banana', 'lime']
spoken_word = []
for i in predicted_label:
    spoken_word.append(dictionary[i])
print spoken_word
missed = (predicted_label != y_test)
print('Test accuracy: %.2f percent' % (100 * (1 - np.mean(missed))))
