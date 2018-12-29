import numpy as np
import tisean
import matplotlib.pyplot as plt
import pandas as pd
import math
import pyyawt

def overall_analysis(x, f, sigma, em_features=[0.2]
                     , lf_features=[5.0], ws_features=['db8', 5]
                     , ts_features=[8, 1, 0.1, 314], n=80):
    
    # list of dataframes of length n
    output = []
    for i in range(n):
        filter1 = Filters(x, f, f+np.random.randn(len(f))*sigma, sigma)
        filter1.run_smoothing_methods(em_features=em_features, lf_features=lf_features, ws_features=ws_features
                                      , ts_features=ts_features)
        df = filter1.quality_measures()
        output += [df]
    
    # get the mean and deviation
    mean = sum(output)/n
    diff_list = [(x-mean).abs() for x in output]
    diff = pd.concat(diff_list).max(level=0)
    
    # keep 3 decimal places
    mean = mean.round(3)
    diff = diff.round(3)
    
    # initialize the output dataframe
    df = mean.copy()
    
    for col in mean.columns:
        df[col] = list(zip(mean.loc[:,col], diff.loc[:,col]))
    
    df = df.applymap(lambda x: str(x[0]) + ' ± ' + str(x[1]))
    df = df.style.apply(Filters.highlight_min)
    
    return df

class Filters:
    def __init__(self, x, f, series, sigma):
        self.x = x
        self.f = f
        self.series = series
        self.sigma = sigma
        self.moving_average = None
        self.exponential_smoothing = None
        self.linear_fourier = None
        self.wavelet = None
        self.tisean = None
    
    def __moving_average_filter(self):
        w = np.ones(5)
        w = w/w.sum()
        t = np.lib.pad(self.series, (2,2), 'symmetric') # symmetric padding
        self.moving_average = np.convolve(w, t, 'valid')
        return
    
    def __exponential_smoothing_filter(self, a=0.2):
        result = np.copy(self.series)
        for i in range(len(result)):
            if i!=0:
                result[i] = a*result[i] + (1-a)*result[i-1]
        self.exponential_smoothing = result
        return
    
    def __linear_fourier_smoothing(self, cutoff=10.0):
        '''
        https://www.scipy-lectures.org/intro/scipy/auto_examples/plot_fftpack.html
        '''
        t_fft = np.fft.fft(self.series)
        #power = np.abs(t_fft)
        sample_freq = np.fft.fftfreq(self.series.size, 1e-3)
        high_freq_fft = t_fft.copy()
        high_freq_fft[np.abs(sample_freq) > cutoff] = 0
        self.linear_fourier = np.real(np.fft.ifft(high_freq_fft))
        return
    
    def __wavelet_shrinkage(self, filt='db8', level=5):
        '''
        https://github.com/holgern/pyyawt
        http://matlab.izmiran.ru/help/toolbox/wavelet/wden.html
        https://pyyawt.readthedocs.io/
        '''
        denoised,_,_ = pyyawt.denoising.wden(self.series, 'sqtwolog', 's', 'mln', level, filt)
        self.wavelet = denoised
        return
    
    def __tisean_denoise(self, embedding_dimension=8, iteration=1, neighbourhood_size=0.1, delay_time=314):
        '''
        https://gist.github.com/benmaier/3d7a10a4492c19bd91ce270fa2321519
        https://www.pks.mpg.de/~tisean/TISEAN_2.1/docs/docs_c/nrlazy.html
        '''
        self.tisean = tisean.nrlazy(input=self.series, m=embedding_dimension, i=iteration
                                            , r=neighbourhood_size, d=delay_time)
        return
    
    @staticmethod
    def l1_norm(f, f_den, dt=1e-3):
        return dt * np.sum(np.abs(f-f_den))
    
    @staticmethod
    def l2_norm(f, f_den, dt=1e-3):
        return (dt * np.sum(np.square(f-f_den)))**0.5
    
    @staticmethod
    def l_inf_norm(f, f_den, dt=1e-3):
        return np.amax(np.abs(f-f_den))
    
    @staticmethod
    def min_euclid_distance(t, y, A, diff_span, loc):
        orig = np.array([t[loc], y])
        start = max(0, loc-diff_span)
        end = min(len(A), loc+diff_span)
        if start == end: end += 1
        min_dis = np.amin([np.linalg.norm(orig-np.array([t[i], A[i]])) for i in range(start, end)])
        return min_dis
    
    @staticmethod
    def visual_error(t, f, f_den, dt=1e-3):
        size = len(t)
        ve_prev = sum(Filters.min_euclid_distance(t
                                                  , f[i]
                                                  , f_den, math.ceil(abs(f[i]-f_den[i])/dt)
                                                  , i)**2 for i in range(size))
        return (dt * ve_prev)**0.5
    
    @staticmethod
    def sym_visual_error(t, f, f_den, dt=1e-3):
        ve_1 = Filters.visual_error(t, f, f_den)**2
        ve_2 = Filters.visual_error(t, f_den, f)**2
        return (ve_1 + ve_2)**0.5
    
    @staticmethod
    def highlight_min(s):
        '''
        highlight the minimum in a series bold.
        '''
        l = pd.Series([float(item.split(' ± ')[0]) for item in s])
        is_min = l == l.min()
        return ['font-weight: bold' if v else '' for v in is_min]

    def run_smoothing_methods(self, em_features=[0.2], lf_features=[5.0], ws_features=['db8', 5]
                              , ts_features=[8, 1, 0.1, 314]):
        self.__moving_average_filter()
        self.__exponential_smoothing_filter(a=em_features[0])
        self.__linear_fourier_smoothing(cutoff=lf_features[0])
        self.__wavelet_shrinkage(filt=ws_features[0], level=ws_features[1])
        self.__tisean_denoise(embedding_dimension=ts_features[0], iteration=ts_features[1]
                            , neighbourhood_size=ts_features[2], delay_time=ts_features[3])
        return

    def smoothing_plot(self, lbd=-0.5, ubd=0.5):

        idx = np.where((self.x>=lbd) & (self.x<=ubd))
        f_part = self.f[idx]
        x_part = self.x[idx]
        series_part = self.series[idx]
        ma_part = self.moving_average[idx]
        em_part = self.exponential_smoothing[idx]
        lf_part = self.linear_fourier[idx]
        ws_part = self.wavelet[idx]
        ts_part = self.tisean[idx]

        plt.figure(num=None, figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(2, 3, 1)
        plt.plot(x_part, f_part, label='original', linewidth=1)
        plt.title('Original ' + r'$\sigma$' + (' = %.2f' % (self.sigma)))
        plt.legend()
        plt.subplot(2, 3, 2)
        plt.scatter(x_part, series_part, color='r', label='with noise', s=0.5)
        plt.plot(x_part, ma_part, label='after smoothing', linewidth=1)
        plt.title('Moving Average')
        plt.legend()
        plt.subplot(2, 3, 3)
        plt.scatter(x_part, series_part, color='r', label='with noise', s=0.5)
        plt.plot(x_part, ws_part, label='after smoothing', linewidth=1)
        plt.title('Wavelet Shrinkage')
        plt.legend()
        plt.subplot(2, 3, 4)
        plt.scatter(x_part, series_part, color='r', label='with noise', s=0.5)
        plt.plot(x_part, ts_part, label='after smoothing', linewidth=1)
        plt.title('TISEAN')
        plt.legend()
        plt.subplot(2, 3, 5)
        plt.scatter(x_part, series_part, color='r', label='with noise', s=0.5)
        plt.plot(x_part, em_part, label='after smoothing', linewidth=1)
        plt.title('Exponential Smoothing')
        plt.legend()
        plt.subplot(2, 3, 6)
        plt.scatter(x_part, series_part, color='r', label='with noise', s=0.5)
        plt.plot(x_part, lf_part, label='after smoothing', linewidth=1)
        plt.title('Linear Fourier Smoothing')
        plt.legend()
    
    def quality_measures(self):
        df = pd.DataFrame(0, index=['Noisy data','MA filter','Wavelet filter'
                                    ,'TISEAN','Exp. smoothing','FFT filter']
                          , columns=(r'$L^1$',r'$L^2$',r'$L^{\infty}$',r'$SE_2$'))
        data = [self.series, self.moving_average, self.wavelet, self.tisean
                , self.exponential_smoothing, self.linear_fourier]
        for i in range(len(data)):
            temp_data = data[i]
            data_append = [self.l1_norm(self.f, temp_data), self.l2_norm(self.f, temp_data)
                           , self.l_inf_norm(self.f, temp_data), self.sym_visual_error(self.x, self.f, temp_data)]
            df.iloc[i,:] = data_append
        #df = df.round(3)
        #df = df.style.apply(Filters.highlight_min)
        return df
    
    