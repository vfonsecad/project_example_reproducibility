# ------------------------------------------------------------------------

#                               mcw-pls
# by: valeria fonseca diaz
# supervisors: Wouter Saeys, Bart De Ketelaere
# ------------------------------------------------------------------------


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_val_predict, KFold, GridSearchCV
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



# ------------------------- class_weighted_mcesimpls_sklearn ----------------------------------------------



class mcw_pls_sklearn(BaseEstimator, RegressorMixin):

    def __init__(self, n_components=2, max_iter=30, R_initial=None, scale_sigma2=1):

        
        self.n_components = n_components
        self.max_iter = max_iter
        self.R_initial = R_initial
        self.scale_sigma2 = scale_sigma2
    
    def simpls_loadings(self,xx, yy, P, ncp):
        
        
        
        mu_x = ((P.dot(xx)).sum(axis=0))/ P.sum()
        mu_y = ((P.dot(yy)).sum(axis=0))/ P.sum()
        

        Xc = xx.copy() - mu_x
        Yc = yy.copy() - mu_y

        N = Xc.shape[0]
        K = Xc.shape[1]
        q = Yc.shape[1]

        R = np.zeros((K, ncp))  # Weights to get T components
        V = np.zeros((K, ncp))  # orthonormal base for X loadings
        S = Xc.T.dot(P).dot(Yc)  # cov matrix

        aa = 0

        while aa < ncp:
            
            r = S[:,:]            
            
            if q > 1:

                U0, sval, Qsvd = sp.linalg.svd(S, full_matrices=True, compute_uv=True)
                Sval = np.zeros((U0.shape[0], Qsvd.shape[0]))
                Sval[0:sval.shape[0], 0:sval.shape[0]] = np.diag(sval)

                Rsvd = U0.dot(Sval)
                r = Rsvd[:, 0]
                
                
            tt = Xc.dot(r)
            tt.shape = (N, 1)
            tt = tt - ((P.dot(tt)).sum(axis=0)/ P.sum())
            TT_scale = np.sqrt(tt.T.dot(P).dot(tt))
            # Normalize
            tt = tt / TT_scale            
            r = r / TT_scale
            r.shape = (K, 1)
            p = Xc.T.dot(P).dot(tt)
            v = p
            v.shape = (K, 1)

            if aa > 0:
                v = v - V.dot(V.T.dot(p))

            v = v / np.sqrt(v.T.dot(v))
            S = S - v.dot(v.T.dot(S))

            R[:, aa] = r[:, 0]
            V[:, aa] = v[:, 0]

            aa += 1

        return R
    


    
    def fit(self, xx, yy):

        X = xx.copy()
        Y = yy.copy()

        N = X.shape[0]
        K = X.shape[1]
        q = Y.shape[1]
        
      
            
        P = np.identity(N)
        
        
        ncp = self.n_components
        sigma_factor = self.scale_sigma2
        max_iter = self.max_iter
        R_initial = self.R_initial
        sigma_tol = 0.000001

  

        
        mu_x = ((P.dot(X)).sum(axis=0)) / P.sum()
        Xc = X - mu_x
        mu_y = ((P.dot(Y)).sum(axis=0)) / P.sum()
        Yc = Y - mu_y


        if R_initial is None:

            # ---  Initial SIMPLS (RSIMPLS)

            R = self.simpls_loadings(X, Y, P, ncp)


        else:

            R = R_initial.copy()
            R.shape = (K, ncp)




        # --- Iterative process

        kk = 0
        sigma2 = 100000
        
        

        while kk < max_iter and sigma2 > sigma_tol:

            TT = Xc.dot(R)  # TT are not weighted

            tcal_raw0 = np.concatenate((np.ones((Xc.shape[0], 1)), TT), axis=1)
            wtemp = sp.linalg.solve(tcal_raw0.T.dot(P.dot(tcal_raw0)), tcal_raw0.T.dot(P.dot(Y)))

            sigma_vector = np.power(Y - tcal_raw0.dot(wtemp), 2).sum(axis=1)
            sigma2 = sigma_factor * (sigma_vector).mean()
            P = np.multiply(np.exp(-sigma_vector / (2 * sigma2)), np.identity(N))

            mu_x = ((P.dot(X)).sum(axis=0)) / P.sum()
            Xc = X - mu_x
            mu_y = wtemp[0:1, :]
            Yc = Y - mu_y

            R = self.simpls_loadings(X, Y, P, ncp)

            kk += 1
            
        TT = Xc.dot(R)
               
        # --- mcw-pls final regression

        
        tcal_raw0 = np.concatenate((np.ones((X.shape[0], 1)), TT), axis=1)
        wtemp = sp.linalg.solve(tcal_raw0.T.dot(P.dot(tcal_raw0)), tcal_raw0.T.dot(P.dot(Y)))
            
            
        
        BPLS = R.dot(wtemp[1:, :])

        self.x_scores_ = TT
        self.x_weights_ = R
        self.x_mu = mu_x
        self.y_mu = wtemp[0:1, :]
        self.BPLS = BPLS
        self.sample_weights = P
        self.x_scores_coef_ = wtemp[1:,:]

    def predict(self, X):

        Ypred = self.y_mu + (X - self.x_mu).dot(self.BPLS)

        return Ypred


# ------------------------- class_weighted_mcesimpls ----------------------------------------------


class mcw_pls(object):

    def __init__(self, xx, yy, lv, model_name=""):
        ''' Initialize a Weighted mcesimpls proposal Class object with calibration data '''

        assert type(xx) is np.ndarray and type(yy) is np.ndarray and xx.shape[0] == yy.shape[0]

        self.xcal = xx.copy()
        self.ycal = yy.copy()
        self.Ncal = xx.shape[0]
        self.XK = xx.shape[1]
        self.YK = yy.shape[1]
        self.model_name = model_name
        self.lv = lv

    def __str__(self):
        return 'class_weighted_mcesimpls'

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

    def train(self, iters=30, current_R0=None, factor_sigma=1):

        

        mcw_pls_train_object = mcw_pls_sklearn(n_components=self.lv, max_iter=iters, R_initial=current_R0,scale_sigma2=factor_sigma)
        mcw_pls_train_object.fit(self.get_xcal(), self.get_ycal())

        mcw_pls_fitted = mcw_pls_train_object.predict(self.get_xcal())
        mcw_pls_coeff = mcw_pls_train_object.BPLS


        mcw_pls_coeff.shape = (self.XK, self.YK)
        mcw_pls_fitted.shape = (self.Ncal, self.YK)

        mcw_pls_output = {'BPLS': mcw_pls_coeff,
                         'x_mean':mcw_pls_train_object.x_mu,
                         'y_mean':mcw_pls_train_object.y_mu,
                         'x_scores':mcw_pls_train_object.x_scores_,
                         'x_weights':mcw_pls_train_object.x_weights_,
                         'x_scores_coef' : mcw_pls_train_object.x_scores_coef_,
                         'fitted': mcw_pls_fitted,
                         'train_object': mcw_pls_train_object,
                         'sample_weights': mcw_pls_train_object.sample_weights,
                         'factor_sigma': factor_sigma
                         }

        return mcw_pls_output


        # --- cross-validation

    def crossval_KFold(self, train_object, number_splits=10):

        cv_object = KFold(n_splits=number_splits)
        cv_predicted = cross_val_predict(train_object, self.get_xcal(), self.get_ycal(), cv=cv_object)

        cv_output = {'cv_predicted': cv_predicted}

        return cv_output


    def tune_sigma_factor(self, sigma_factor_range):

        mcw_pls_train_object = mcw_pls_sklearn(n_components=self.lv, max_iter=30, R_initial=None)
        cv_sigma_factor = {'scale_sigma2': list(sigma_factor_range)}
        cv_object = KFold(n_splits=10)

        TuneCV = GridSearchCV(estimator=mcw_pls_train_object, param_grid=cv_sigma_factor, cv=cv_object,
                              scoring='neg_mean_squared_error', return_train_score=True)
        TuneCV.fit(self.get_xcal(), self.get_ycal())

        tune_output = {'rmsecv': np.sqrt(-1 * TuneCV.cv_results_['mean_train_score']),
                       'grid': sigma_factor_range}

        return tune_output

    def predict(self, X, mcw_pls_output):

        y_pred = mcw_pls_output["y_mean"] + (X - mcw_pls_output["x_mean"]).dot(mcw_pls_output["BPLS"])

        return y_pred

    def predict_x(self, X, mcw_pls_output):

        x_pred = (X - mcw_pls_output["x_mean"]).dot(mcw_pls_output["x_weights"])

        return x_pred


# ¡¡¡ --- !!! ---> optimal model complexity function


def optimal_lv_simpls(xx_cal, yy_cal, total_ncp = 20, total_shuffles = 20, total_repetitions = 10):

    current_n = xx_cal.shape[0]
    kfold_splits = 10
    current_R = None
    maximum_iters  = 0

    optimal_lvs = np.zeros((total_repetitions,1))
    
    for lv_ii in range(total_repetitions):
        
        
        rmsecv_shuffles = np.zeros((total_ncp, total_shuffles))


        jj = 0
        for shuffle_id in range(total_shuffles):    

            ii = 0    

            current_shuffle = np.random.permutation(np.arange(current_n))
            current_xx_cal = xx_cal[current_shuffle,:]
            current_yy_cal = yy_cal[current_shuffle,:]

            for ncp in range(1,total_ncp+1):


                my_model = mcw_pls(xx = current_xx_cal,yy = current_yy_cal, lv = ncp)

                # --- training model

                my_model_trained = my_model.train(iters = maximum_iters, current_R0 = current_R, factor_sigma = 0)

                # --- cv    

                my_model_cv_pred = my_model.crossval_KFold(train_object = my_model_trained["train_object"],number_splits=kfold_splits)
                my_model_cv_error = my_model.rmse(y_pred = my_model_cv_pred["cv_predicted"], yy = current_yy_cal)

                rmsecv_shuffles[ii,jj] = my_model_cv_error[0]

                ii+=1
            jj+=1


        rmsecv_mean = rmsecv_shuffles.mean(axis=1)
        rmsecv_mean.shape = [total_ncp,1]

        min_rmsecv_lv_optimal_ii = np.argmin(rmsecv_mean[:,0])
        limit = np.percentile(rmsecv_shuffles[min_rmsecv_lv_optimal_ii,:].flatten(),90)


        rmsecv_ii = min_rmsecv_lv_optimal_ii
        found = False

        while found == False and rmsecv_ii < total_ncp-1:
            rmsecv_ii += 1
            if rmsecv_mean[rmsecv_ii] >= limit:
                found = True

        if found:
            optimal_lv = min_rmsecv_lv_optimal_ii + 1
        else:
            rmsecv_ii = min_rmsecv_lv_optimal_ii
            while found == False and rmsecv_ii >= 0:
                rmsecv_ii -= 1
                if rmsecv_mean[rmsecv_ii] >= limit:
                    found = True
            if found:
                optimal_lv = rmsecv_ii + 1
            else:
                optimal_lv = 0
                print("not optimal lv found")
    
        optimal_lvs[lv_ii,0] = optimal_lv
        
        
    (values,counts) = np.unique(optimal_lvs,return_counts=True)
    ind = np.argmax(counts)
    final_optimal_lv = values[ind]

    return (rmsecv_shuffles, optimal_lvs)


def min_rmsecv_lv_simpls(xx_cal, yy_cal, total_ncp = 20):

    current_n = xx_cal.shape[0]
    kfold_splits = 10
    current_R = None
    maximum_iters  = 0
    
    rmsecv_values = np.zeros(total_ncp)
    
    ii = 0


    
    for ncp in range(1,total_ncp+1):


        my_model = mcw_pls(xx = xx_cal,yy = yy_cal, lv = ncp)

                    # --- training model

        my_model_trained = my_model.train(iters = maximum_iters, current_R0 = current_R, factor_sigma = 0)

                    # --- cv    

        my_model_cv_pred = my_model.crossval_KFold(train_object = my_model_trained["train_object"],number_splits=kfold_splits)
        my_model_cv_error = my_model.rmse(y_pred = my_model_cv_pred["cv_predicted"], yy = yy_cal)

        rmsecv_values[ii] = my_model_cv_error[0]

        ii+=1
   

    return np.argmin(rmsecv_values) + 1


def cv_lv_simpls(xx_cal0, yy_cal0, total_ncp = 20, total_repetitions = 10):

    current_n = xx_cal0.shape[0]
    kfold_splits = 10
    current_R = None
    maximum_iters  = 0
    
    
    rmsecv_values = np.zeros((total_ncp, total_repetitions))
    
    
    
    all_samples = np.arange(current_n)
    
    xx_cal = xx_cal0.copy()
    yy_cal = yy_cal0.copy()

    for rep in range(total_repetitions):
        
        if rep > 0:
            
            np.random.shuffle(all_samples)

            xx_cal = xx_cal0[all_samples,:]
            yy_cal = yy_cal0[all_samples,:]  
            
        ii = 0        
    
        for ncp in range(1,total_ncp+1):


            my_model = mcw_pls(xx = xx_cal,yy = yy_cal, lv = ncp)

                        # --- training model

            my_model_trained = my_model.train(iters = maximum_iters, current_R0 = current_R, factor_sigma = 0)

                        # --- cv    

            my_model_cv_pred = my_model.crossval_KFold(train_object = my_model_trained["train_object"],number_splits=kfold_splits)
            my_model_cv_error = my_model.rmse(y_pred = my_model_cv_pred["cv_predicted"], yy = yy_cal)

            rmsecv_values[ii, rep] = my_model_cv_error[0]

            ii+=1
        
        rep += 1


    return rmsecv_values