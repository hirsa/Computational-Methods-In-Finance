import numpy as np
import statsmodels.tsa.api as smt

class Denoise_Functions:

    def __init__(self, t):
        self.t = t
    
    '''
    This is the functions to generate time series for analysis.
    '''
    def f1(self):
        return np.sin(5*self.t)
    def f2(self):
        return np.piecewise(self.t, [(self.t>=-10)&(self.t<0), (self.t>=0)&(self.t<=10)], [lambda t: -t**2/2, lambda t: t**2/2])
    def f3(self):
        return np.piecewise(self.t, [(self.t>=-10)&(self.t<-2), (self.t>=-2)&(self.t<=7), (self.t>7)&(self.t<=10)], [5, -3, 8])
    def f4(self):
        return np.piecewise(self.t
                            , [(self.t>=-10)&(self.t<=-3.33), (self.t>-3.33)&(self.t<=-1.11), (self.t>-1.11)&(self.t<=1.11)
                               , (self.t>1.11)&(self.t<=3.33), (self.t>3.33)&(self.t<=6.66), (self.t>6.66)&(self.t<=10)]
                            , [lambda t: np.sin(5*t), lambda t: np.sin(-3.33*t), 2
                              , -3.33**2/2, lambda t: -(t-6.66)**2/2, lambda t: (t-6.66)**2/2])
    def f5(self):
        a = np.array([1, -.24, .13, .3, -.2, .3, -.3])
        b = np.array([1, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, .9])
        ar = np.r_[1, np.negative(a)]
        ma = np.r_[b]
        y = smt.arma_generate_sample(ar=ar, ma=ma, nsample=20001, burnin=300, sigma=0.02)
        return y
    '''
    This is the functions to plot the time series, not for analysis.
    '''
    def f1_plot(self):
        return self.f1()
    def f2_plot(self):
        return self.f2()
    def f3_plot(self):
        f_3 = np.select([(self.t>=-10)&(self.t<-2), (self.t>-2)&(self.t<7), (self.t>7)&(self.t<=10)], [5, -3, 8], np.nan)
        return f_3
    def f4_plot(self):
        f1 = lambda t: np.sin(5*t)
        f2 = lambda t: np.sin(-3.33*5)
        f5 = lambda t: -(t-6.66)**2/2
        f6 = lambda t: (t-6.66)**2/2
        f_4 = np.select([(self.t>=-10)&(self.t<-3.33)
                            , (self.t>-3.33)&(self.t<-1.11)&(np.logical_not(np.isclose(self.t,-1.11)))
                            , (np.logical_not(np.isclose(self.t,-1.11)))&(self.t>-1.11)&(self.t<1.11)&(np.logical_not(np.isclose(self.t,1.11)))
                            , (np.logical_not(np.isclose(self.t,1.11)))&(self.t>1.11)&(self.t<3.33), (self.t>3.33)&(self.t<6.66), (self.t>6.66)&(self.t<=10)]
                        , [f1(self.t), f2(self.t), 2, -3.33**2/2, f5(self.t), f6(self.t)]
                        , np.nan)
        return f_4
    def f5_plot(self):
        return self.f5()

