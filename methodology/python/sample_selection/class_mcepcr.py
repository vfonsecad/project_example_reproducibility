# ------------------------------------------------------------------------

#                    create MCEPLS as class_mcepls using scikit learn

# ------------------------------------------------------------------------

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_predict, KFold, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, make_scorer


# ------------------------- class_mcepls_sklearn ----------------------------------------------

# This class corresponds to the implementation of MCEPCR (maximum correlation entropy principal component regression) based on Robust Principal Component Analysis Based on Maximum Correntropy Criterion Ran He, Bao-Gang Hu, Senior Member, IEEE, Wei-Shi Zheng, Member, IEEE, and Xiang-Wei Kong IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 20, NO. 6, JUNE 2011. 
# First mcepcr_sklearn is created to define the mcepcr estimator of the type sklearn. Second class mcepcr is created as the object that the user will work with. Class mcepcr makes use of mcepcr_sklearn. Note that the latter is defined in the same structure as any regression object of sklearn.




class mcepcr_sklearn(BaseEstimator, RegressorMixin):

    def __init__(self, n_components=2, max_iter=30, V_initial=None, scale_sigma2=1):
        
        '''The mcepcr_sklearn object is created specifying the #comp, # of iterations to run, whether there is a            special V initial loadings vector/matrix and the factor scale for parameter sigma2 
        This class can be used to fit mcepcr or classical pcr. See below
        '''

        self.n_components = n_components
        self.max_iter = max_iter
        self.V_initial = V_initial
        self.scale_sigma2 = scale_sigma2
        
    def mcepca(self,xx):
        
        
        ncp = self.n_components
        sigma_factor=self.scale_sigma2
        V_initial=self.V_initial
        max_iter=self.max_iter
        
        X = xx.copy()
        mu = X.mean(axis=0)
        Xc = X - mu

        N = X.shape[0]
        K = X.shape[1]

        
        
        sigma_tol = 0.000001

        if V_initial is None:

            # ---  Initial PCA

            U, Sval, Vt = np.linalg.svd(Xc)

            U = U[:, 0:ncp]
            Sval = Sval[0:ncp]
            Vt = Vt[0:ncp, :]

            Smat = np.zeros((ncp, ncp))
            Smat[0:ncp, 0:ncp] = np.diag(Sval)
            U.dot(Smat).dot(Vt)
            V = Vt.T.copy()

        else:
            V = V_initial.copy()
            V.shape = (K, ncp)
            Vt = V.T.copy()

        # --- Iterative process

        kk = 0
        sigma2 = 100000
        P = np.identity(N)

        while kk < max_iter and sigma2 > sigma_tol:
            # print("-----------------", kk)

            # --- scores ----
            TT = Xc.dot(V)
            sigma_vector = np.power(Xc - TT.dot(Vt), 2).sum(axis=1)
            sigma2 = sigma_factor * (sigma_vector).mean()


            # --- weights ---
            pp = Xc.dot(Xc.T) - TT.dot(TT.T)
            P = np.multiply(np.exp(-pp / (2 * sigma2)), np.identity(N))

            # --- weighted mean ---

            mu = ((P.dot(X)).sum(axis=0)) / P.sum()
            Xc = X - mu

            # --- SVD weighted cov matrix ---

            P_sqrt = np.sqrt(P)

            U_hat, Sval_hat, Vt_hat = np.linalg.svd(P_sqrt.dot(Xc).dot(Xc.T).dot(P_sqrt))

            U_hat = U_hat[:, 0:ncp]
            Sval_hat = Sval_hat[0:ncp]
            Vt_hat = Vt_hat[0:ncp, :]

            Smat_hat_inv = np.zeros((ncp, ncp))
            Smat_hat_inv[0:ncp, 0:ncp] = np.diag(1 / Sval_hat)

            #V = (Xc.T.dot(Vt_hat.T)).dot(np.sqrt(Smat_hat_inv)) / P.sum()
            V = Xc.T.dot(P_sqrt).dot(U_hat).dot(np.sqrt(Smat_hat_inv))
            Vt = V.T.copy()

            kk += 1
            
        Output = dict()    
        Output["x_mean"] = mu
        Output["V"] = V
        Output["P"] = P
        
        return Output
        
        
        
        

    def fit(self, xx,yy):

        ''' Robust Principal Component Analysis Based on
            Maximum Correntropy Criterion
         Ran He, Bao-Gang Hu, Senior Member, IEEE, Wei-Shi Zheng, Member, IEEE, and Xiang-Wei Kong
         IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 20, NO. 6, JUNE 2011
         
         Explanation: This algorithm starts by centering X. If there is no V initial, the initial loadings are
         the classical PCA loadings. After this, the iterative process starts. Note that if max_iter is 0, the 
         function delivers classical PCA. 
         On each iteration, sample weights are calculated with kernel approach for max correntropy. With the 
         weights, a new X mean is calculated. Then original X is centered with the new mean. With the new X 
         centered and the weights, weighted PCA is performed on #comp selected. The process is repeated max_iter          times or with stopping criterion. 
         Finally, weighted classical regression is performed between final PCA components and Y, introducing              intercept in the model to calculate weighted mean of Y. Final regression vector for the model Y=XB is            calculated and all paramters are delivered: PCA scores and loadings, regression vector, sample weights,          final mean values.
         
         xx: X calibration matrix as numpy array N x K
         yy: Y calibration matrix N x YK. If Y is univariate, it still needs to be an N x 1 numpy matrix.
         
         '''
        Y = yy.copy()
        X = xx.copy()
        
        mcepca_output = self.mcepca(X)
        mu = mcepca_output["x_mean"]
        Xc = X - mu 
        V = mcepca_output["V"]
        P = mcepca_output["P"]

        TT = Xc.dot(V)

        # --- Weighted regression

        tcal_raw0 = np.concatenate((np.ones((X.shape[0], 1)), TT), axis=1)
        wtemp = np.linalg.solve(tcal_raw0.T.dot(P.dot(tcal_raw0)), tcal_raw0.T.dot(P.dot(Y)))

        BPCR = V.dot(wtemp[1:,:])

        self.x_scores_ = TT
        self.x_weights_ = V
        self.x_mu = mu
        self.y_mu = wtemp[0,0]
        self.BPCR = BPCR
        self.sample_weights = P

        return self

    def predict(self, X):
        
        ''' To use the mcepcr model, final means and final regression vectors are used as 
        Y = mean(Y) + (X-mean(X)) * BETA '''

        Ypred = self.y_mu + (X-self.x_mu).dot(self.BPCR)

        return Ypred


# ------------------------- class_mcepcr ----------------------------------------------

# This is the class the user has direct contact with to fit mcepcr models

class mcepcr(object):

    def __init__(self, xx, yy, n_components, model_name=""):
        ''' 
        
        Initialize a PCR Class object with calibration data. Baseline paper:
        Robust Principal Component Analysis Based on
            Maximum Correntropy Criterion
         Ran He, Bao-Gang Hu, Senior Member, IEEE, Wei-Shi Zheng, Member, IEEE, and Xiang-Wei Kong
         IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 20, NO. 6, JUNE 2011
         
         see help(mcepcr_sklearn.fit) for further explanation
        
       
        X and Y need to be specified, as well as the number of components to be fitted.
        Model name is just a string identifier of the object if the user wants to include one
        
        --- Input --- 
        
        xx: X calibration matrix as numpy array N x K
        yy: Y calibration matrix N x YK. If Y is univariate, it still needs to be an N x 1 numpy matrix.
        n_components: int of the number of components to consider
        model_name: string to identify the model name'''

        assert type(xx) is np.ndarray and type(yy) is np.ndarray and xx.shape[0] == yy.shape[0]

        self.xcal = xx.copy()
        self.ycal = yy.copy()
        self.Ncal = xx.shape[0]
        self.XK = xx.shape[1]
        self.YK = yy.shape[1]
        self.model_name = model_name
        self.n_components = n_components

    def __str__(self):
        return 'class_mcepcr'

        # --- Copy of data

    def get_xcal(self):
        ''' Get copy of xcal data '''
        return self.xcal

    def get_ycal(self):
        ''' Get copy of ycal data '''
        return self.ycal

        # --- Define y names and wv numbers (in nm)

    def set_wv_varlabel(self, x_wv0):
        
        '''
        Define labels for the X variables
        x_wv0: 1D numpy array of size K        
        '''        
        
        x_wv = np.array(x_wv0).flatten()
        self.wv_varlabel = x_wv

    def set_yy_varlabel(self, y_constituent0):
        '''
        Define label(s) for the Y variable(s)
        y_constituent0: 1D numpy array of size YK        
        ''' 
        
        y_names = np.array(y_constituent0).flatten()
        self.yy_varlabel = y_names

    def plot_spectra(self, xx):
        
        ''' 
        Get spectral plot for a matrix X
        xx: spectral matrix of size M x K (M is not necessarily the same as N as xx does not need to be the                 exact same calibration X matrix)
        '''
        fig, ax = plt.subplots()
        ax.plot(xx.T)
        ticks = np.round(np.arange(0, self.XK, self.XK / 6))
        plt.xticks(ticks, np.round(self.wv_varlabel[ticks.astype(int)].flatten(), 1))
        plt.xlabel("Wavelength (nm)")
        plt.title(self.model_name)
        plt.show()

        # --- Define performance measures

    def rmse(self, yy, y_pred, sample_weights=None):
        
        '''
        Root Mean Squared Error calculation as sqrt(1/n * sum(yy-y_pred)^2) where n is the number of rows in yy         (y_pred). If yy is multivariate, the output is a vector of its length
        
        --- Input ---
        
        yy: N x YK matrix of observed Y values
        y_pred : N x YK matrix of predited (fitted) Y values
        sample_weights: 1D or 2D array of sample weights for each row of yy. if 2D array, it must be of size Nx1
        
        
        --- Output ---
        
        rmse value
        
        '''

        if sample_weights is None:
            sample_weights = np.ones((yy.shape[0], 1)) / yy.shape[0]
        else:
            sample_weights = sample_weights / sample_weights.sum(axis=0)

        r = yy.copy() - y_pred.copy()
        r = r ** 2
        msep = np.average(r, axis=0, weights=sample_weights)
        rmse = np.sqrt(msep)

        return rmse

        # --- Define MCEPCR regression from my class in scikit learn

    def train(self, iters=30, current_v0=None, factor_sigma=1):
        
        '''
        MCEPCR train that calls fit from mcepcr_sklearn
        
        --- Input ---
        
        iters: maximum number of iterations. If 0, classical PCR is performed.
        current_v0: initial V loadings vector. If None, classical PCA loadings are the initial V
        factor_sigma: factor to rescale sigma for kernel during iterations (not used for classical PCR)
        
        --- Output ---
        
        mcepcr_Output dict with all the parameters of the fitting process:
        
                         'BPCR': final regression vector for Y=XB,
                         'x_mean': final X mean,
                         'y_mean': final y mean ,
                         'x_scores': final PCA scores for X,
                         'x_weights': final PCA V loadings for X,
                         'fitted': fitted Y values,
                         'trainObject': model training object to be used in other functions,
                         'sample_weights': final sample weights. Diag matrix N x N with weights in diag,
                         'factor_sigma': sigma scale factor used        
        '''

        mcepcr_trainObject = mcepcr_sklearn(n_components=self.n_components, max_iter=iters, V_initial=current_v0,
                                             scale_sigma2=factor_sigma)
        mcepcr_trainObject.fit(self.get_xcal(), self.get_ycal())

        mcepcr_fitted = mcepcr_trainObject.predict(self.get_xcal())
        mcepcr_coeff = mcepcr_trainObject.BPCR


        mcepcr_coeff.shape = (self.XK, self.YK)
        mcepcr_fitted.shape = (self.Ncal, self.YK)

        mcepcr_Output = {'BPCR': mcepcr_coeff,
                         'x_mean':mcepcr_trainObject.x_mu,
                         'y_mean':mcepcr_trainObject.y_mu,
                         'x_scores':mcepcr_trainObject.x_scores_,
                         'x_weights':mcepcr_trainObject.x_weights_,
                         'fitted': mcepcr_fitted,
                         'trainObject': mcepcr_trainObject,
                         'sample_weights': mcepcr_trainObject.sample_weights,
                         'factor_sigma':factor_sigma
                         }

        return mcepcr_Output

        # --- CrossValidation

    def crossval_KFold(self, trainObject,number_splits=10):
        
        '''
        Perform k fold cross validation
        
        --- Input --- 
        
        trainObject: object specified as mcepcr_Output["trainObject"] in function train
        number_splits: number of k fold for CV (number of data groups)
        
        --- Output ---
        
        cv_Output dict: 
        
            'cvPredicted': numpy array N x YK of crossval predicted Y values
        
        '''

        cvObject = KFold(n_splits=number_splits)
        cv_predicted = cross_val_predict(trainObject, self.get_xcal(), self.get_ycal(), cv=cvObject)

        cv_Output = {'cvPredicted': cv_predicted}

        return cv_Output

    def tune_sigma_factor(self, sigma_factor_range):
        
        '''
        Perform grid seach crossval by sklearn to tune sigma factor
        
        --- Input --- 
        
        sigma_factor_range: numpy array of sigma range
        
        --- Output ---
        
        tune_Output dict: 
        
            'rmsecv': numpy array of rmsecv values for the sigma values
            'grid': sigma grid used in the search
        
        '''

       

        mcepcr_trainObject = mcepcr_sklearn(max_iter=30)
        cv_sigma_factor = {'scale_sigma2': list(sigma_factor_range)}
        cvObject = KFold(n_splits=10)

        TuneCV = GridSearchCV(estimator=mcepcr_trainObject, param_grid=cv_sigma_factor, cv=cvObject,
                              scoring='neg_mean_squared_error', return_train_score=True)
        TuneCV.fit(self.get_xcal(), self.get_ycal())

        tune_Output = {'rmsecv': np.sqrt(-1 * TuneCV.cv_results_['mean_train_score']),
                       'grid': sigma_factor_range}

        return tune_Output
	
    def predict(self, X,mcepcr_Output):
        
        ''' Use trained calibration model
        
        --- Input ---
        
        X: numpy array M x K of new X samples in original scale
        mcepcr_Output: Output dict returned by train
        
        
        --- Output ---
        
        Ypred = numpy array of size M x YK of predicted values
        
        '''

        Ypred = mcepcr_Output["y_mean"] + (X-mcepcr_Output["x_mean"]).dot(mcepcr_Output["BPCR"])

        return Ypred

    def predict_x(self, X, mcepcr_Output):
        
        '''
        Calculate predicted X scores for new X samples
        
        --- Input ---
        X: numpy array M x K of new X samples in original scale
        
        --- Output ---
        
        Xpred: numpy array M x ncomp of scores for new X data
        
        
        '''

        Xpred =  (X - mcepcr_Output["x_mean"]).dot(mcepcr_Output["x_weights"])

        return Xpred










