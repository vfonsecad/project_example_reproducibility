# ------------------------------------------------------------------------

#                               pcr
# by: valeria fonseca diaz
# supervisors: Wouter Saeys, Bart De Ketelaere
# ------------------------------------------------------------------------


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_predict, KFold, GridSearchCV
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



# ------------------------- class pcr ----------------------------------------------



class pcr_sklearn(BaseEstimator, RegressorMixin):

    def __init__(self, n_components=2):

        
        self.n_components = n_components    


    
    def fit(self, xx, yy):

        X = xx.copy()
        Y = yy.copy()

        N = X.shape[0]
        K = X.shape[1]
        q = Y.shape[1]
        
      
            
        P = np.identity(N)
        
        
        ncp = self.n_components       
  

        
        mu_x = ((P.dot(X)).sum(axis=0)) / P.sum()
        Xc = X - mu_x
        mu_y = ((P.dot(Y)).sum(axis=0)) / P.sum()
        Yc = Y - mu_y
        
        # --- pca loadings
        
        u, s, v = np.linalg.svd(Xc)
        R = v[0:ncp,:].T
        TT = Xc.dot(R)  # TT are not weighted
               
        # --- pcr final regression

        
        tcal_raw0 = np.concatenate((np.ones((X.shape[0], 1)), TT), axis=1)
        wtemp = sp.linalg.solve(tcal_raw0.T.dot(P.dot(tcal_raw0)), tcal_raw0.T.dot(P.dot(Y)))
            
            
        
        BPCR = R.dot(wtemp[1:, :])

        self.x_scores_ = TT
        self.x_weights_ = R
        self.x_mu = mu_x
        self.y_mu = wtemp[0:1, :]
        self.BPCR = BPCR
        self.sample_weights = P
        self.x_scores_coef_ = wtemp[1:,:]

    def predict(self, X):

        Ypred = self.y_mu + (X - self.x_mu).dot(self.BPCR)

        return Ypred


# ------------------------- class_pcr ----------------------------------------------


class pcr(object):

    def __init__(self, xx, yy, ncp, model_name=""):
        ''' Initialize a pcr Class object with calibration data '''

        assert type(xx) is np.ndarray and type(yy) is np.ndarray and xx.shape[0] == yy.shape[0]

        self.xcal = xx.copy()
        self.ycal = yy.copy()
        self.Ncal = xx.shape[0]
        self.XK = xx.shape[1]
        self.YK = yy.shape[1]
        self.model_name = model_name
        self.ncp = ncp

    def __str__(self):
        return 'class_pcr'

        # --- Copy of data

    def get_xcal(self):
        ''' Get copy of xcal data '''
        return self.xcal

    def get_ycal(self):
        ''' Get copy of ycal data '''
        return self.ycal

  

       # --- Define performance measures

    def rmse(self, yy, y_pred, sample_weights=None):
        
        
        N = yy.shape[0]

        if sample_weights is None:
            P = np.identity(n = N) / N
        else:
            sample_weights_vec = sample_weights.flatten()
            P = np.diag(v = sample_weights_vec) / sample_weights_vec.sum()

    
        r = yy.copy() - y_pred.copy()
        r = np.power(r,2)
        msep = ((P.dot(r)).sum(axis=0)) / P.sum()
        rmse = np.sqrt(msep)

        return rmse
    
    def r2(self, yy, y_pred, sample_weights=None, corrected = False):
        
        N = yy.shape[0]
        q = yy.shape[1]

        if sample_weights is None:
            P = np.identity(n = N) / N
        else:
            P = np.diag(sample_weights.flatten()) / sample_weights.flatten().sum()
            
        r2 = np.zeros((1, q))
        
        for ii in range(q):            

            yy_reg = np.concatenate((np.ones((y_pred.shape[0], 1)), y_pred[:,[ii]]), axis=1)
            coeffs = np.linalg.solve(yy_reg.T.dot(P.dot(yy_reg)), yy_reg.T.dot(P.dot(yy[:,[ii]])))
            y_pred_fitted = yy_reg.dot(coeffs)

            mu_y_pred = ((P.dot(yy[:,[ii]])).sum(axis=0)) / P.sum()
            y_pred_c = yy[:,[ii]] - mu_y_pred
            y_pred_res = y_pred_fitted - yy[:,[ii]]
            total_ss = ((P.dot(y_pred_c**2)).sum(axis=0)) 
            residual_ss = ((P.dot(y_pred_res**2)).sum(axis=0))
            
            intercept = coeffs[0,0]
            slope = coeffs[1,0]
            
            
            r2[0,ii] = 1 - (residual_ss / total_ss)
            
            if corrected:      
                
                factor = 1 - (np.abs(slope-1) / np.sqrt(total_ss))
                # (1-(np.abs(intercept) / (np.amax(yy) - np.amin(yy))) - (np.abs(slope-1) / np.sqrt(total_ss)))
                
                r2[0,ii] = factor*(1 - (residual_ss / total_ss))

            
        return r2
    
    def global_r2(self, yy, y_pred, sample_weights = None, corrected = False):
                
        
        single_r2 = self.r2(yy, y_pred, sample_weights, corrected)
        glob_corr_r2 = np.amin(single_r2,axis = 1)
        
        return glob_corr_r2     
    
    
    def r2_penalized_univariate(self,yy, y_pred, sample_weights=None, corrected = False):

        N = yy.shape[0]
        q = yy.shape[1]


        assert q == 1, "yy and y_pred must be a one column vector"

        if sample_weights is None:
            P = np.identity(n = N) / N
        else:
            P = np.diag(sample_weights.flatten()) / sample_weights.flatten().sum()


        yy_reg = np.concatenate((np.ones((y_pred.shape[0], 1)), y_pred), axis=1)
        coeffs = np.linalg.solve(yy_reg.T.dot(P.dot(yy_reg)), yy_reg.T.dot(P.dot(yy)))
        y_pred_fitted = yy_reg.dot(coeffs)

        mu_y_pred = ((P.dot(yy)).sum(axis=0)) / P.sum()
        y_pred_c = yy - mu_y_pred
        y_pred_res = y_pred_fitted - yy
        total_ss = ((P.dot(y_pred_c**2)).sum(axis=0)) 
        residual_ss = ((P.dot(y_pred_res**2)).sum(axis=0))

        r2 =  1 - (residual_ss / total_ss)

        if corrected:    

            dist1 = np.abs(np.amin(y_pred_fitted) - np.amin(y_pred))
            dist2 = np.abs(np.amax(y_pred_fitted) - np.amax(y_pred))
            r2 = (1 - (residual_ss / total_ss))*(1-np.amin([np.amax([dist1,dist2])/(np.amax(yy) - np.amin(yy)),1]))


        return r2


        # --- define mcw-pls regression from my class in scikit learn

    def train(self):

        

        pcr_train_object = pcr_sklearn(n_components=self.ncp)
        pcr_train_object.fit(self.get_xcal(), self.get_ycal())

        pcr_fitted = pcr_train_object.predict(self.get_xcal())
        pcr_coeff = pcr_train_object.BPCR


        pcr_coeff.shape = (self.XK, self.YK)
        pcr_fitted.shape = (self.Ncal, self.YK)

        pcr_output = {'BPCR': pcr_coeff,
                         'x_mean':pcr_train_object.x_mu,
                         'y_mean':pcr_train_object.y_mu,
                         'x_scores':pcr_train_object.x_scores_,
                         'x_weights':pcr_train_object.x_weights_,
                         'x_scores_coef' : pcr_train_object.x_scores_coef_,
                         'fitted': pcr_fitted,
                         'train_object': pcr_train_object,
                         'sample_weights': pcr_train_object.sample_weights
                         }

        return pcr_output


        # --- cross-validation

    def crossval_KFold(self, train_object, number_splits=10):

        cv_object = KFold(n_splits=number_splits)
        cv_predicted = cross_val_predict(train_object, self.get_xcal(), self.get_ycal(), cv=cv_object)

        cv_output = {'cv_predicted': cv_predicted}

        return cv_output


 

    def predict(self, X, pcr_output):

        y_pred = pcr_output["y_mean"] + (X - pcr_output["x_mean"]).dot(pcr_output["BPCR"])

        return y_pred

    def predict_x(self, X, pcr_output):

        x_pred = (X - pcr_output["x_mean"]).dot(pcr_output["x_weights"])

        return x_pred


