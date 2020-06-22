# ------------------------------------------------------------------------

#                    create MCEPLS as class_mcepls using scikit learn

# ------------------------------------------------------------------------

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_predict, KFold, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, make_scorer


# ------------------------- class_mccpca_sklearn ----------------------------------------------

# This class corresponds to the implementation of mccpca (maximum correlation entropy principal component analysis) based on Robust Principal Component Analysis Based on Maximum Correntropy Criterion Ran He, Bao-Gang Hu, Senior Member, IEEE, Wei-Shi Zheng, Member, IEEE, and Xiang-Wei Kong IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 20, NO. 6, JUNE 2011. 



class mccpca_sklearn(BaseEstimator, RegressorMixin):

    def __init__(self, n_components=2, max_iter=30, V_initial=None, scale_sigma2=1):
        
        '''The mccpca_sklearn object is created specifying the #comp, # of iterations to run, whether there is a            special V initial loadings vector/matrix and the factor scale for parameter sigma2 
        This class can be used to fit mcepcr or classical pcr. See below
        '''

        self.n_components = n_components
        self.max_iter = max_iter
        self.V_initial = V_initial
        self.scale_sigma2 = scale_sigma2
        
    def fit(self,xx):
        
        
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
        
        TT = Xc.dot(V)

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
        
        self.mu_x = mu
        self.V = V
        self.sample_weights = P
        self.x_scores_ = TT
        
        return self       
 

    def predict(self, X):
        
        ''' To use the mccpca model, final means and final loadings vectors are used as 
        Y = mean(X) + (X-mean(X)) * V '''
        
        TT = (X - self.mu_x).dot(self.V)
        xx_predicted = TT.dot(self.V.T) + self.mu_x

        return xx_predicted


# ------------------------- class_mccpca ----------------------------------------------

# This is the class the user has direct contact with to fit mccpca models

class mccpca(object):

    def __init__(self, xx,n_components, model_name=""):
        ''' 
        
        Baseline paper:
        Robust Principal Component Analysis Based on
            Maximum Correntropy Criterion
         Ran He, Bao-Gang Hu, Senior Member, IEEE, Wei-Shi Zheng, Member, IEEE, and Xiang-Wei Kong
         IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 20, NO. 6, JUNE 2011
         
         see help(mcepcr_sklearn.fit) for further explanation
        
       
        X and Y need to be specified, as well as the number of components to be fitted.
        Model name is just a string identifier of the object if the user wants to include one
        
        --- Input --- 
        
        xx: X calibration matrix as numpy array N x K
        n_components: int of the number of components to consider
        model_name: string to identify the model name'''

        assert type(xx) is np.ndarray 

        self.xcal = xx.copy()        
        self.Ncal = xx.shape[0]
        self.XK = xx.shape[1]        
        self.model_name = model_name
        self.n_components = n_components

    def __str__(self):
        return 'class_mccpca'

        # --- Copy of data

    def get_xcal(self):
        ''' Get copy of xcal data '''
        return self.xcal


        # --- Define performance measures

    def rmse(self, xx, x_pred, sample_weights=None):
        
        '''
        Root Mean Squared Error calculation as sqrt(1/n * sum(xx-x_pred)^2) where n is the number of rows in yy         (y_pred). If yy is multivariate, the output is a vector of its length
        
        --- Input ---
        
        xx: N x XK matrix of observed X values
        x_pred : N x XK matrix of predited (fitted) X values
        sample_weights: 1D or 2D array of sample weights for each row of xx. if 2D array, it must be of size Nx1
        
        
        --- Output ---
        
        rmse value
        
        '''

        if sample_weights is None:
            sample_weights = np.ones((xx.shape[0], 1)) / xx.shape[0]
        else:
            sample_weights = sample_weights / sample_weights.sum(axis=0)

        r = xx.copy() - x_pred.copy()
        r = (r ** 2).sum(axis=1)
        r.shape = (r.shape[0],1)
        msep = np.average(r, axis=0, weights=sample_weights)
        rmse = np.sqrt(msep)

        return rmse

        # --- Define mccpca from my class in scikit learn

    def train(self, iters=30, current_v0=None, factor_sigma=1):
        
        '''
        mccpca train that calls fit from mccpca_sklearn
        
        --- Input ---
        
        iters: maximum number of iterations. If 0, classical PCR is performed.
        current_v0: initial V loadings vector. If None, classical PCA loadings are the initial V
        factor_sigma: factor to rescale sigma for kernel during iterations (not used for classical PCR)
        
        --- Output ---
        
        mccpca_Output dict with all the parameters of the fitting process:
        
                         
                         'x_mean': final X mean,                         
                         'x_scores': final PCA scores for X,
                         'x_weights': final PCA V loadings for X,
                         'fitted': fitted Y values,
                         'train_object': model training object to be used in other functions,
                         'sample_weights': final sample weights. Diag matrix N x N with weights in diag,
                         'factor_sigma': sigma scale factor used        
        '''

        mccpca_train_object = mccpca_sklearn(n_components=self.n_components, max_iter=iters, V_initial=current_v0,
                                             scale_sigma2=factor_sigma)
        mccpca_train_object.fit(self.get_xcal())

        mccpca_fitted = mccpca_train_object.predict(self.get_xcal())
        mccpca_coeff = mccpca_train_object.V


       
        mccpca_fitted.shape = (self.Ncal, self.XK)

        mccpca_Output = {'x_mean':mccpca_train_object.mu_x,
                         'x_scores':mccpca_train_object.x_scores_,
                         'x_weights':mccpca_train_object.V,
                         'fitted': mccpca_fitted,
                         'train_object': mccpca_train_object,
                         'sample_weights': mccpca_train_object.sample_weights,
                         'factor_sigma':factor_sigma
                         }

        return mccpca_Output

        # --- CrossValidation

    def crossval_KFold(self, train_object,number_splits=10):
        
        '''
        Perform k fold cross validation
        
        --- Input --- 
        
        train_object: object specified as mcepcr_Output["train_object"] in function train
        number_splits: number of k fold for CV (number of data groups)
        
        --- Output ---
        
        cv_Output dict: 
        
            'cvPredicted': numpy array N x XK of crossval predicted X values
        
        '''

        cvObject = KFold(n_splits=number_splits)
        cv_predicted = cross_val_predict(train_object, self.get_xcal(), cv=cvObject)

        cv_Output = {'cv_predicted': cv_predicted}

        return cv_Output


    def predict(self, X,mccpca_Output):
        
        ''' Use trained calibration model
        
        --- Input ---
        
        X: numpy array M x K of new X samples in original scale
        mcepcr_Output: Output dict returned by train
        
        
        --- Output ---
        
        Xpred = numpy array of size new_M x K of predicted values
        
        '''

        Xpred = mccpca_Output['train_object'].predict(X)

        return Xpred

    def predict_x_scores(self, X, mccpca_Output):
        
        '''
        Calculate predicted X scores for new X samples
        
        --- Input ---
        X: numpy array M x K of new X samples in original scale
        
        --- Output ---
        
        Xpred: numpy array M x ncomp of scores for new X data
        
        
        '''

        Xpred =  (X - mccpca_Output["x_mean"]).dot(mccpca_Output["x_weights"])

        return Xpred










