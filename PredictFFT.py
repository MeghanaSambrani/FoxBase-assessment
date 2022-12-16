# import required librariers
import numpy as np

from numpy import fft 
import matplotlib.pyplot as plt



# Class to predict next sequence of values
# using FFT
class PredictFFT():
    """
    This initiate the FFT
    """
    
    # Class variables
    def __init__(self, x, n_predict):

        """
        Initiate the FFT with the values to predict and
        number of sequence
        Args:
            self.x (list): Initial/Raw data
            self.predict (int): number of days to predict the value
            
            
        """
        
        self.x = x
        self.predict = n_predict
      
    
    def fourierExtrapolation(self):
        
        n = self.x.size
        n_harm = 2048                    # number of harmonics in model
        t = np.arange(0, n)
        p = np.polyfit(t, self.x, 1)         # find linear trend in x
        x_notrend = self.x - p[0] * t        # detrended x
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        f = fft.fftfreq(n)              # frequencies
        indexes = list(range(n))

        # sort indexes by frequency, lower -> higher
        indexes.sort(key=lambda i: np.absolute(x_freqdom[i]))
        indexes.reverse()
        j=0
        t = np.arange(0, n + self.predict)
        restored_sig = np.zeros(t.size)
        col = ["darkorange","m","darkorange","g","k"]

        for i in indexes[:1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n   # amplitude
            phase = np.angle(x_freqdom[i])          # phase
            a = (ampli * np.cos(2 * np.pi * f[i] * t + phase))

            restored_sig += a


        return restored_sig + p[0] * t


