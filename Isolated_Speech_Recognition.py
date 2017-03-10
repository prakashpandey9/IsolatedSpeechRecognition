import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline

import os
file_paths = []
labels = []
spoken_word = []
for f in os.listdir('Audio'):
    for w in os.listdir('Audio/' + f):  
        file_paths.append('Audio/' + f + '/' + w)
        labels.append(f)
        if f not in spoken_word:
            spoken_word.append(f)
print('List of spoken words:', spoken_word)
print set(labels)

from scipy.io import wavfile

data = np.zeros((len(file_paths), 75550))
maxsize = -1
for n,file in enumerate(file_paths):
    _, d = wavfile.read(file)
    data[n, :d.shape[0]] = d
    if d.shape[0] > maxsize:
        maxsize = d.shape[0]
        #print maxsize
data = data[:, :maxsize]
#Each sample file is one row in data, and has one entry in labels
print('Number of total files:', data.shape[0])
all_labels = np.zeros(data.shape[0])
for n, l in enumerate(set(labels)):
    all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n
    
print('Labels and label indices', all_labels)
print data.shape

import scipy

def stft(x, fftsize=512, overlap_pct=.8): # stft stands for "Short Time Fourier Transform"
    hop = int(fftsize * (1 - overlap_pct))
    w = scipy.hanning(fftsize + 1)[:-1]    
    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
    return raw[:, :(fftsize // 2)]

from numpy.lib.stride_tricks import as_strided

# Peaks in Amplitude vs Frequency graph and it is used as features of speech data. A better option is to use
# MFCC(Mel-Frequency Cepstral Coefficients) which has been used later in the notebook.
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
    #Add l_size and half - 1 of center size to get to actual peak location
    top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)
    return heights, top[:n_peaks]


'''plot_data = np.abs(stft(data[20, :]))[15, :]
values, locs = peakfind(plot_data, n_peaks=6)
fp = locs[values > -1]
fv = values[values > -1]
plt.plot(plot_data, color='blue')
plt.plot(fp, fv, 'x', color='darkred')
plt.title('Location of Peaks')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')'''

all_obs = []
for i in range(data.shape[0]):
    d = np.abs(stft(data[i, :]))
    n_dim = 7
    obs = np.zeros((n_dim, d.shape[0]))
    for r in range(d.shape[0]):
        _, t = peakfind(d[r, :], n_peaks=n_dim)
        obs[:, r] = t.copy()
    if i % 10 == 0:
        print("Processed observation %s" % i)
    all_obs.append(obs)
    
all_obs = np.atleast_3d(all_obs)
print all_obs.shape

# Feature detection using MFCC technique. This is much better choice than peak detection.

'''import python_speech_features as speech
all_obs = np.zeros([199, 13, 471])
for n,file in enumerate(file_paths):
    all_obs[n,]=speech.mfcc(data[n,:]).T
print all_obs.shape
#s = speech.mfcc(data[0])
#print s.shape'''

import scipy.stats as st
import numpy as np

class gmmhmm:
    def __init__(self, n_states):
        self.n_states = n_states
        self.random_state = np.random.RandomState(0)
        
        #Normalize random initial state
        self.prior = self._normalize(self.random_state.rand(self.n_states, 1))  # Initialize prior probability "Pi"
        self.A = self._stochasticize(self.random_state.rand(self.n_states, self.n_states)) # Initialize "A" probability
        
# Initializing mu, covariance and dimension matrix to calcuate "B" matrix       
        self.mu = None
        self.covs = None
        self.n_dims = None
           
    def _forward(self, B):# B is basically bj(o(t))
        log_likelihood = 0.
        T = B.shape[1]  # B.shape = (n_state x Total_time)
        alpha = np.zeros(B.shape) # n_states x Total_Time
        for t in range(T):
            if t == 0:
                alpha[:, t] = B[:, t] * self.prior.ravel()
            else:
                alpha[:, t] = B[:, t] * np.dot(self.A.T, alpha[:, t - 1])
         
            alpha_sum = np.sum(alpha[:, t])
            alpha[:, t] /= alpha_sum
            log_likelihood = log_likelihood + np.log(alpha_sum)
        return log_likelihood, alpha
    
    def _backward(self, B):
        T = B.shape[1]
        beta = np.zeros(B.shape);
           
        beta[:, -1] = np.ones(B.shape[0])
            
        for t in range(T - 1)[::-1]:
            beta[:, t] = np.dot(self.A, (B[:, t + 1] * beta[:, t + 1]))
            beta[:, t] /= np.sum(beta[:, t])
        return beta
# The product alpha_t(i)*beta_t(i) gives the probability of whole observation with the condition that at time t it was
# in ith state. Alone alpha upto time T can't give this probability.
       
    
    def _state_likelihood(self, obs):
        obs = np.atleast_2d(obs)
        B = np.zeros((self.n_states, obs.shape[1]))
        for s in range(self.n_states): # Probability of getting the observation (o1,o2,...oT) when it is in state "s"
            #Needs scipy 0.14
            np.random.seed(self.random_state.randint(1))
            B[s, :] = st.multivariate_normal.pdf(
                obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T)
        return B
    
    def _normalize(self, x):
        return (x + (x == 0)) / np.sum(x)
    
    def _stochasticize(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)
    
    def _em_init(self, obs):
        #Using this _em_init function allows for less required constructor args
        if self.n_dims is None:
            self.n_dims = obs.shape[0]
        if self.mu is None:
            subset = self.random_state.choice(np.arange(self.n_dims), size=self.n_states, replace=False)
            self.mu = obs[:, subset]
        if self.covs is None:
            self.covs = np.zeros((self.n_states, self.n_dims, self.n_dims))
            self.covs += np.diag(np.diag(np.cov(obs)))[:, :, None]
        return self
    
    def _em_step(self, obs): 
        obs = np.atleast_2d(obs)
        B = self._state_likelihood(obs)
        T = obs.shape[1]
        
        log_likelihood, alpha = self._forward(B)
        beta = self._backward(B)
        
        xi_sum = np.zeros((self.n_states, self.n_states))
        gamma = np.zeros((self.n_states, T)) # gamma is the probability that it is in ith state at time t, given the
        # observations and the model. gamma.shape = (n_state, T) 
        
        for t in range(T - 1):
            partial_sum = self.A * np.dot(alpha[:, t], (beta[:, t] * B[:, t + 1]).T)
            xi_sum += self._normalize(partial_sum)
            partial_g = alpha[:, t] * beta[:, t]
            gamma[:, t] = self._normalize(partial_g)
              
        partial_g = alpha[:, -1] * beta[:, -1]
        gamma[:, -1] = self._normalize(partial_g)
        
        expected_prior = gamma[:, 0]
        expected_A = self._stochasticize(xi_sum)
        
        expected_mu = np.zeros((self.n_dims, self.n_states))
        expected_covs = np.zeros((self.n_dims, self.n_dims, self.n_states))
        
        gamma_state_sum = np.sum(gamma, axis=1)
        #Set zeros to 1 before dividing
        gamma_state_sum = gamma_state_sum + (gamma_state_sum == 0)
        
        for s in range(self.n_states):
            gamma_obs = obs * gamma[s, :]
            expected_mu[:, s] = np.sum(gamma_obs, axis=1) / gamma_state_sum[s]
            partial_covs = np.dot(gamma_obs, obs.T) / gamma_state_sum[s] - np.dot(expected_mu[:, s], expected_mu[:, s].T)
            #Symmetrize
            partial_covs = np.triu(partial_covs) + np.triu(partial_covs).T - np.diag(partial_covs)
        
        #Ensure positive semidefinite by adding diagonal loading
        expected_covs += .01 * np.eye(self.n_dims)[:, :, None]
        
        self.prior = expected_prior
        self.mu = expected_mu
        self.covs = expected_covs
        self.A = expected_A
        return log_likelihood
    
    def train(self, obs, n_iter=15):
        if len(obs.shape) == 2:
            for i in range(n_iter):
                self._em_init(obs)
                log_likelihood = self._em_step(obs)
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            for n in range(count):
                for i in range(n_iter):
                    self._em_init(obs[n, :, :])
                    log_likelihood = self._em_step(obs[n, :, :])
        return self
    
    def test(self, obs):
        if len(obs.shape) == 2:
            B = self._state_likelihood(obs)
            log_likelihood, _ = self._forward(B)
            return log_likelihood
        elif len(obs.shape) == 3:
            count = obs.shape[0]
            out = np.zeros((count,))
            for n in range(count):
                B = self._state_likelihood(obs[n, :, :])
                log_likelihood, _ = self._forward(B)
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
ms = [gmmhmm(7) for y in ys]

_ = [model.train(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]
ps1 = [model.test(X_test) for m in ms]
res1 = np.vstack(ps1)
predicted_label1 = np.argmax(res1, axis=0)
dictionary = ['apple', 'banana', 'elephant', 'dog', 'frog', 'cat', 'jack', 'god', 'Intelligent', 'hello']
spoken_word = []
for i in predicted_label1:
    spoken_word.append(dictionary[i])
print spoken_word
missed = (predicted_label1 != y_test)
print('Test accuracy: %.2f percent' % (100 * (1 - np.mean(missed))))


# Recognizing a new word, first by recording the word using pyaudio.
from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

THRESHOLD = 100
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

def is_silent(snd_data):
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    snd_data = _trim(snd_data)

    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    r = array('h', [0 for i in xrange(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in xrange(int(seconds*RATE))])
    return r

def record():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.05)
    return sample_width, r

def record_to_file(path):
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

if __name__ == '__main__':
    print("please speak a word into the microphone")
    record_to_file('/home/prakash/Desktop/hmm-speech-recognition-0.1/Speech_Testing.wav')
    print("Done")


# Predicting the spoken word
import numpy as np
from scipy.io import wavfile

new_data = np.zeros(96000)
maxsize = 75526
_, new_d = wavfile.read("/home/prakash/Desktop/hmm-speech-recognition-0.1/Speech_Testing.wav")
new_data[:new_d.shape[0]] = new_d
new_data = new_data[:maxsize]

observation = []

new_d = np.abs(stft(new_data))
n_dim = 10
new_obs = np.zeros((n_dim, new_d.shape[0]))
for r in range(new_d.shape[0]):
    _, t = peakfind(new_d[r, :], n_peaks=n_dim)
    new_obs[:, r] = t.copy()
observation.append(new_obs)
    
observation = np.atleast_3d(observation)
observation /= observation.sum(axis=1)

ps_test = [model.test(observation) for m in ms]
res_test = np.vstack(ps_test)
test_predicted_label = np.argmax(res_test, axis=0)
word_spoken = dictionary[test_predicted_label]
print word_spoken
