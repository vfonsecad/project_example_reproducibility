# ------------------------------------------------------------------------

#                    create MCEPLS as class_mcepls using scikit learn

# ------------------------------------------------------------------------

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_predict, KFold, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score, make_scorer


# ------------------------- class_mcepls_sklearn ----------------------------------------------



class mcepls1_sklearn(BaseEstimator, RegressorMixin):

    def __init__(self, rlambda = 0.1, max_iter=30, w0=None,slopeAdjust=True, scale_sigma2=1):

        self.rlambda = rlambda
        self.max_iter=max_iter
        self.w0 = w0
        self.slopeAdjust = slopeAdjust
        self.scale_sigma2=scale_sigma2


    def fit(self, X,Y):

        ''' Maximum correlation entropy PLS
         http://dx.doi.org/10.1016/j.chemolab.2016.12.002
         Algorithm for PLS class object. It is also possible to provide external data to use the function
         Only suitable for univate Y
         '''

        xxm = X.copy()
        yym = Y.copy()

        K = xxm.shape[1]
        N = xxm.shape[0]
        YK = yym.shape[1]
        A = 1

        assert YK == 1

        # - Initialize matrices

        W = np.zeros((K, A))  # Weights to get T components
        TT = np.zeros((N, A))  # T components X scores
        P = np.zeros((K, A))  # X loadings
        Q = np.zeros((YK, A))  # Y loadings
        TT_scale = np.zeros((1, A))  # scale of components before normalization
        #Xvar = np.zeros((1, A))  # store variance explained for X
        #Yvar = np.zeros((1, A))  # store variance explained for Y
        #BPLS = np.zeros((K, YK, A))  # reg coefficients for cumulative components

        aa = 0
        yym_copy = yym.copy()  # yym_copy for the loop and keep yym the same
        xxm_copy = xxm.copy()  # xxm_copy for the loop and keep xxm the same
        ep_conver = 0

        Am = np.identity(N)
        
        
        if self.w0 is None:
            w = np.linalg.solve(xxm_copy.T.dot(Am.dot(xxm_copy)) + self.rlambda * np.identity(K),
                                       xxm_copy.T.dot(Am.dot(yym_copy)))
        else:
            w=self.w0.copy()   

        w.shape = (K, 1)

        epsilon = self.max_iter


        while epsilon > ep_conver:
            e2 = np.power((yym_copy - xxm_copy.dot(w)), 2)
            sigma2 = e2.mean()*self.scale_sigma2
            Am = np.diag((np.exp(-e2 / (2 * sigma2)) / (2 * sigma2)).flatten())
            wtemp = np.linalg.solve(xxm_copy.T.dot(Am.dot(xxm_copy)) + self.rlambda * np.identity(K),
                                        xxm_copy.T.dot(Am.dot(yym_copy)))
            wtemp.shape = (K, 1)
            epsilon -= 1
            w = wtemp.copy()


        tt = xxm_copy.dot(w)
        tt.shape = (N, 1)  # lv. current scores
        TT[:, aa] = tt[:, 0]  # Store
        TT_scale[0, aa] = np.sqrt(tt.T.dot(tt))

        Am2 = np.identity(N)
        p = xxm_copy.T.dot(tt)
        p.shape = (K, 1)
        q = yym_copy.T.dot(Am2).dot(tt)/(tt.T.dot(Am2).dot(tt))
        q.shape = (YK, 1)


        # Store
        W[:, aa] = w[:, 0]
        P[:, aa] = p[:, 0]
        Q[:, aa] = q[:, 0]

        if self.slopeAdjust:
            BetaPLS = W.dot(Q.T)
        else:
            BetaPLS = W



        self.x_scores_ = TT
        self.x_weights_ = W
        self.x_loadings_ = P
        self.y_loadings_ = Q
        self.coef_ = BetaPLS
        self.sample_weights = Am

        return self

    def predict(self, X):

        Ypred = np.dot(X, self.coef_)

        return Ypred





# ------------------------- class_mcepls ----------------------------------------------




class mcepls(object):


    def __init__(self, xx, yy, rlambda, model_name=""):
        ''' Initialize a PCR Class object with calibration data '''

        assert type(xx) is np.ndarray and type(yy) is np.ndarray and xx.shape[0] == yy.shape[0]

        self.xcal = xx.copy()
        self.ycal = yy.copy()
        self.Ncal = xx.shape[0]
        self.XK = xx.shape[1]
        self.YK = yy.shape[1]
        self.model_name = model_name
        self.rlambda = rlambda

    def __str__(self):
        return 'class_mcepls1'

        # --- Copy of data

    def get_xcal(self):
        ''' Get copy of xcal data '''
        return self.xcal

    def get_ycal(self):
        ''' Get copy of ycal data '''
        return self.ycal

        # --- Define y names and wv numbers (in nm)

    def set_wv_varlabel(self, x_wv0):
        x_wv = np.array(x_wv0).flatten()
        self.wv_varlabel = x_wv

    def set_yy_varlabel(self, y_constituent0):
        y_names = np.array(y_constituent0).flatten()
        self.yy_varlabel = y_names

    def plot_spectra(self, xx):
        fig, ax = plt.subplots()
        ax.plot(xx.T)
        ticks = np.round(np.arange(0, self.XK, self.XK / 6))
        plt.xticks(ticks, np.round(self.wv_varlabel[ticks.astype(int)].flatten(), 1))
        plt.xlabel("Wavelength (nm)")
        plt.title(self.model_name)
        plt.show()

        # --- Define performance measures

    def rmse(self,yy, y_pred, sample_weights=None):

        if sample_weights is None:
            sample_weights = np.ones((yy.shape[0], 1)) / yy.shape[0]
        else:
            sample_weights = sample_weights / sample_weights.sum(axis=0)

        r = yy.copy() - y_pred.copy()
        r = r ** 2
        msep = np.average(r,axis=0, weights=sample_weights)
        rmse = np.sqrt(msep)

        return rmse


        # --- Define MCEPLS1 regression from my class in scikit learn

    def train(self, iters = 30, current_w0=None,current_slopeAdjust=True, factor_sigma=1):

        mcepls_trainObject = mcepls1_sklearn(rlambda=self.rlambda, max_iter=iters, w0=current_w0,slopeAdjust=current_slopeAdjust, scale_sigma2=factor_sigma)
        mcepls_trainObject.fit(self.get_xcal(), self.get_ycal())


        mcepls_fitted = mcepls_trainObject.predict(self.get_xcal())
        mcepls_coeff = mcepls_trainObject.coef_

        mcepls_coeff.shape = (self.XK, self.YK)
        mcepls_fitted.shape = (self.Ncal, self.YK)

        mcepls_Output = {'BPLS': mcepls_coeff,
                         'fitted': mcepls_fitted,
                         'trainObject': mcepls_trainObject,
                         'sample_weights':mcepls_trainObject.sample_weights
                        }

        return mcepls_Output

        # --- CrossValidation

    def crossval_KFold(self, trainObject,number_splits=10):

        cvObject = KFold(n_splits=number_splits)
        cv_predicted = cross_val_predict(trainObject, self.get_xcal(), self.get_ycal(), cv=cvObject)

        cv_Output = {'cvPredicted': cv_predicted}

        return cv_Output

    def tune_rlambda(self, rlambda_range):



        #rmsecv_score = make_scorer(self.rmsecv)

        mcepls_trainObject = mcepls1_sklearn(max_iter=30)
        cv_rlambda = {'rlambda': list(rlambda_range)}
        cvObject = KFold(n_splits=10)

        TuneCV = GridSearchCV(estimator=mcepls_trainObject, param_grid=cv_rlambda, cv=cvObject,
                              scoring='neg_mean_squared_error', return_train_score=True)
        TuneCV.fit(self.get_xcal(), self.get_ycal())

        tune_Output = {'rmsecv': np.sqrt(-1 * TuneCV.cv_results_['mean_train_score']),
                       'grid': rlambda_range}

        return tune_Output



    def predict(self, X,mcepls_Output):

        Ypred = np.dot(X, mcepls_Output["BPLS"])

        return Ypred







