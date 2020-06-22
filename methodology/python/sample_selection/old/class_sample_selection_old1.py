# ############## ------------------------------------------------------------------------
#                            CLASS sample_selection
# ############## ------------------------------------------------------------------------


import numpy as np
from scipy.spatial import distance
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN
from rpy2.robjects.packages import importr
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri



class sample_selection(object):

    def __init__(self, xx, yy=np.empty([0,0]), ncp = 10):

        ''' Initialize a Sample selection Class object with provided spectral data and possible reference values (optional)
        For some of the sample selection strategies a dimension reduction is needed, especially for cases where K (#vars) >> N (# samples)
        
        --- Input ---
        xx: Spectral matrix of N rows and K columns
        yy: optional. Reference values, matrix of N rows and YK columns
        ncp: number of components for PCA dimension reduction, Default ncp=10
        
        '''

        assert type(xx) is np.ndarray and type(yy) is np.ndarray and xx.shape[0] >= yy.shape[0]
        
        self.xcal = xx.copy()
        self.ycal = yy.copy()        
        self.Ncal = xx.shape[0]
        self.XK = xx.shape[1]
        self.YK = yy.shape[1]
        self.ncp = ncp
        
    
        
              
    def __str__(self):
        return 'SampleSelection'

    # --- Copy of data

    def get_xcal(self):
        ''' Get copy of xcal data '''
        return self.xcal

    def get_ycal(self):
        ''' Get copy of ycal data '''
        return self.ycal
    
    # ----------------------------- PCA REDUCTION ------------------------------------
    
    
    def get_xcal_pca_scores(self):
        
        '''
        Dimension reduction based on X = USV'
        
        --- Output ---
        
        xcal_u: N x ncp matrix of U scores (no scaling with singular values in S)
        xcal_t: T = US. N x ncp matrix of T scores (scaled with singular values in S)
        
        U with euclidean is equivalent to T=US with mahalanobis
        
        '''
        
        xx = self.get_xcal()
        Nin = self.Ncal
        xx_c = (xx - xx.mean(axis=0))
        ncp = self.ncp
        U,sval,Vt = np.linalg.svd(xx_c, full_matrices = False, compute_uv=True)
        Sval = np.zeros((U.shape[0], Vt.shape[0]))
        Sval[0:sval.shape[0], 0:sval.shape[0]] = np.diag(sval)
        xx_u = U[:,0:ncp]
        xx_t = U[:,0:ncp].dot(Sval[0:ncp, 0:ncp])
        
        self.xcal_u = xx_u.copy()
        self.xcal_t = xx_t.copy()
        
    def get_xcal_u(self):
        ''' Get copy of xcal_u data '''
        return self.xcal_u 
    
    def get_xcal_t(self):
        ''' Get copy of xcal_t data '''
        return self.xcal_t
    
    
        


    # ------------------------------------------ Kennard Stone -----------------------------

    def kennard_stone(self, Nout=10, fixed_samples = None, dim_reduction = True, distance_measure = "mahalanobis"):

        ''' This algorithm corresponds to Kennard Stone classical alg.
        It enables the update of a current selected subset by entering fixed_samples as a 1-D array of 0's and 1's where 1 = part of current subset
        Nout is yet the total number of samples, i.e, current + to be selected in the update
        
        --- Input ---
        
        dim_reduction: Use PCA scores
        distance: default : "mahalanobis" or "euclidean". If "mahalanobis", dim_reduction is taken as True
        
        --- Output ---
        
Output dict with:

'sample_id': id's of the final selected samples
'xout': xx matrix of size Nout x K (original data)

        
        '''
        
        
        if distance_measure == "mahalanobis":
            xx = self.get_xcal_u()
        elif dim_reduction:
            xx = self.get_xcal_t()
        else:
            xx = self.get_xcal()
                
        
        
        Nin = xx.shape[0]
        K = xx.shape[1]
        sample_selected = np.zeros((Nin, 1))

        xcal_in = xx.copy()
        max_DD = 1000000
        
        # Initialize

        if fixed_samples is None or fixed_samples.flatten().sum()==0:
            iin = 0
            DD = distance.cdist(xcal_in, xcal_in.mean(axis=0).reshape((1,K)), metric = "euclidean")
            ID = DD.argmin()
            sample_selected[ID, 0] = 1
        else:
            iin = fixed_samples.sum()-1
            sample_selected = fixed_samples.copy().reshape((Nin,1))

        assert Nout >= sample_selected.flatten().sum()

        while  iin < (Nout-1) and max_DD > 0.00001:

            iin += 1
            DD = distance.cdist(xcal_in, xcal_in[sample_selected.flatten()==1,:], metric = "euclidean")
            DD_row = DD.min(axis=1)
            max_DD = DD_row.max()
            ID = DD_row.argmax()
            sample_selected[ID, 0] = 1



        Output = dict()
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]

        return Output


    # ------------------------------------------ K MEDOIDS -----------------------------


    def kmedoids(self, Nout=10, fixed_samples = None, dim_reduction = True, distance_measure = "mahalanobis"):

        ''' This algorithm corresponds to Kmedoids, which is like K means but selecting an actual point of the data as a center classical alg
        It enables the update of a current selected subset by entering fixed_samples as a 1-D array of 0's and 1's where 1 = part of current subset
        Nout is yet the total number of samples, i.e, current + to be selected in the update

        --- Input ---
        
        dim_reduction: Use PCA scores
        distance: default : "mahalanobis" or "euclidean". If "mahalanobis", dim_reduction is taken as True
        
        --- Output ---
        
Output dict with:

'sample_id': id's of the final selected samples
'xout': xx matrix of size Nout x K (original data)

        '''


        if distance_measure == "mahalanobis":
            xx = self.get_xcal_u()
        elif dim_reduction:
            xx = self.get_xcal_t()
        else:
            xx = self.get_xcal()
            
            
        Nin = xx.shape[0]
        #K = xx.shape[0]
        xcal_in = xx.copy()
        all_samples = np.arange(0,Nin)


        # -- Initialize

        if fixed_samples is None or fixed_samples.flatten.sum()==0:
            fixed_samples = np.empty((0,0)).flatten()
            current_samples = np.empty((0,0)).flatten()

        else:
            fixed_samples = fixed_samples.flatten()
            current_samples = all_samples[np.where(fixed_samples == 1)]

        center_id = np.concatenate((current_samples,np.random.choice(all_samples,int(Nout-fixed_samples.sum()))))

        assert Nout >= fixed_samples.sum()
        stop = False
        NoutCurrent = int(fixed_samples.sum())


        while not stop:

            current_centers = center_id.astype(int).copy()
            centers = xcal_in[current_centers,:]
            DD = distance.cdist(xcal_in, centers, metric = "euclidean")
            min_id = DD.argmin(axis=1)
            center_id = np.concatenate((current_samples,np.zeros((Nout-NoutCurrent, 1)).flatten()))

            for im in range(NoutCurrent,Nout):

                group = all_samples[min_id == im]


                if group.shape[0]>1:
                    DD_im = distance.cdist(xcal_in[group,:],xcal_in[group,:], metric = "euclidean")
                    min_id_im = DD_im.sum(axis=1).argmin()
                    center_id[im] = group[min_id_im]

                else:
                    center_id[im] = current_centers[im]


            center_id = center_id.astype(int).flatten()


            current_centers_sorted = current_centers.copy().flatten()
            center_id_sorted = center_id.copy().flatten()
            current_centers_sorted.sort()
            center_id_sorted.sort()

            if np.array_equal(current_centers_sorted,center_id_sorted):
                stop = True

        sample_selected = (np.isin(all_samples,center_id))*1
        sample_selected.shape = (sample_selected.shape[0],1)

        Output = dict()
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[center_id,:]

        return(Output)
    
    # ------------------------------------------------------- SUCCESSIVE PROJECTIONS ------------------------
    
    
    
    def successive_projections(self, Nout=10, fixed_samples = None, center = True):
        
        
        '''
        Successive projections alg as proposed in 
        Heronides Adonias Dantas Filho, Roberto Kawakami Harrop Galvão, Mário Cesar Ugulino Araújo, Edvan Cirino da Silva, Teresa Cristina Bezerra Saldanha, Gledson Emidio José, Celio Pasquini, Ivo Milton Raimundo, Jarbas José Rodrigues Rohwedder,
A strategy for selecting calibration samples for multivariate modelling,
Chemometrics and Intelligent Laboratory Systems,
Volume 72, Issue 1,
2004,
Pages 83-91,
ISSN 0169-7439,
https://doi.org/10.1016/j.chemolab.2004.02.008.
(http://www.sciencedirect.com/science/article/pii/S0169743904000681)

This procedure is performed on high dimensional X matrix (preferably centered. By rows because the procedure will work with X' of size K x N)

--- Input ---



Nout: Total number of final selected samples (including fixed_samples)
fixed_samples: 1-D array of 0's and 1's where 1 = part of current subset
center: logical. Centering xx matrix by rows. default True


--- Output ---

Output dict with:

'sample_id': id's of the final selected samples
'xout': xx matrix of size Nout x K (not centered)

        '''
        
        xx = self.get_xcal().T #Transpose of X
        K = xx.shape[0]
        
        if center:
            xx_c = xx - xx.mean(axis=0) # center by row
        else:
            xx_c = xx.copy()
                
        Nin = xx_c.shape[1]

        all_samples = np.arange(0,Nin)
        sample_selected = np.zeros((Nin, 1))

        xcal_in = xx_c.copy()
        xcal_in_projected = xx_c.copy()


        if fixed_samples is None or fixed_samples.sum()==0:

            ii = 0
            current_id = np.random.choice(Nin,1)[0] # initial sample
            sample_selected[current_id,0] = 1
            
        else:

            Nfixed = fixed_samples.sum()
            assert (Nfixed < Nout), "Nout must be estrictly less than number of fixed samples"
            ii = Nfixed-1   
            sample_selected[:,0] = (fixed_samples==1)*1 # initial samples
            selected_ids = all_samples[sample_selected.flatten()==1]
            
            # Orthogonal set for fixed samples
            
            X_in_projected = xcal_in_projected[:,selected_ids]
            U,sval,Vt = np.linalg.svd(X_in_projected, full_matrices = False, compute_uv=True)
            Sval = np.zeros((U.shape[0], Vt.shape[0]))
            Sval[0:sval.shape[0], 0:sval.shape[0]] = np.diag(sval)
            xcal_in_projected[:,selected_ids] = U[:,0:Nfixed].dot(Sval[0:Nfixed, 0:Nfixed])
            
            
        while ii < (Nout-1): 

            ii += 1

            candidate_ids = all_samples[sample_selected.flatten()==0]
            selected_ids = all_samples[sample_selected.flatten()==1]

            S = xcal_in[:,candidate_ids]
            X_in_projected = xcal_in_projected[:,selected_ids]
            xcal_in_orth = np.identity(K) - (X_in_projected.dot(np.linalg.inv(X_in_projected.T.dot(X_in_projected))).dot(X_in_projected.T))

            S_projected = xcal_in_orth.dot(S)
            S_max_proj = np.argmax(np.sqrt(np.diag(S_projected.T.dot(S_projected))))
            
            current_id = candidate_ids[S_max_proj]
            sample_selected[current_id,0] = 1
            xcal_in_projected[:,current_id] = S_projected[:,S_max_proj]
            
        
        Output = dict()
            
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]
       
        

        return Output
        
    # ---------------------------------------------- SIMPLISMA -----------------------------------------------
        
    def simplisma(self,Nout=10, fixed_samples = None, alpha_factor = 0.01, center=True):
        
        '''
        SIMPLISMA ALGORITHM as proposed in 
http://www.spectroscopyonline.com/training-set-sample-selection-method-based-simplisma-robust-calibration-near-infrared-spectral-analy
21. L. N. Li, T. L. Lin and R. C. Zhang, Spectroscopy 29 (2014) 62
SIMPLSMA: SIMPLe-to-use Interactive Self-modeling Mixture Analysis 

This procedure is performed on high dimensional Z matrix (First X is centered and then normalized  where each row is a vector of length 1)

--- Input ---



Nout: Total number of final selected samples (including fixed_samples)
fixed_samples: 1-D array of 0's and 1's where 1 = part of current subset
center: logical. Centering xx matrix by rows. default True (recommended)
alpha_factor: factor of mean by which to increase samples means for pure values (recommeded between 0.01 and 0.05) See baseline paper for more information

--- Output ---

Output dict with:

'sample_id': id's of the final selected samples
'xout': xx matrix of size Nout x K

        '''
    
    
        xx = self.get_xcal().T # This has to become an array K x N!
        xx_means0 = xx.mean(axis=0)
        xx_stds = xx.std(axis=0)
        
        
        if center:
            xx_c = xx - xx_means0
        else:
            xx_c = xx.copy()

        
        Nin = xx_c.shape[1]
        K = xx_c.shape[0]

        all_samples = np.arange(0,Nin)
        sample_selected = np.zeros((Nin, 1))
        
        xcal_in = xx_c.copy() 
        

        zz = (xcal_in)/np.sqrt(K*(np.power(xx_means0,2)+xcal_in.var(axis=0)))
        
        
        if fixed_samples is None or fixed_samples.sum()==0:
            
            ii = 0
            
        else:

            Nfixed = fixed_samples.sum()
            assert (Nfixed < Nout), "Nout must be estrictly less than number of fixed samples"
            ii = Nfixed-1   
            sample_selected[:,0] = (fixed_samples==1)*1


        

        p = np.zeros((Nin, Nout)) # store pure values       
        xx_means = np.abs(xx_means0)
        alpha_total = alpha_factor * np.amax(xx_means) + 0.000001


        while ii < (Nout-1): 
            
            ii += 1
            
            
    
            candidate_ids = all_samples[sample_selected.flatten()==0]
            


            for candidate in candidate_ids:

                Y = np.concatenate((zz[:,[candidate]], zz[:,sample_selected.flatten()==1]), axis=1)
                p[candidate,ii] = np.linalg.det(Y.T.dot(Y))*xx_stds[candidate]/(xx_means[candidate]+alpha_total)
                


            current_id = np.argmax(p[:,ii])
            
            sample_selected[current_id,0] = 1

            
        
        
        Output = dict()
            
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]
        

        return Output
    
    
    # --------------------------------------------------- OPTIMAL DESIGNS ------------------------------------------
    
    
    def optfederov_r(self, Nout=10, fixed_samples = None, optimality_criterion = "D"):
        
        '''
    
    See https://cran.r-project.org/web/packages/AlgDesign/AlgDesign.pdf for full documentation
    Here, T=US is used so that T'T is invertible
    It is also assummed to have linear and quadratic terms
        

--- Input ---


Nout: Total number of final selected samples (including fixed_samples)
fixed_samples: 1-D array of 0's and 1's where 1 = part of current subset
optimality_criterion: string of optimality criterion for Federov alg. "D", "A", or "I"

--- Output ---

Output dict with:

'sample_id': id's of the final selected samples
'xout': xx matrix of size Nout x K

        '''
        
        
        xx = np.concatenate((self.get_xcal_t(),np.power(self.get_xcal_t(),2)), axis=1)

        Nin = xx.shape[0]
        
        assert (Nout) > xx.shape[1] , "Number of final selected samples must be bigger than ncp"



        AlgDesign = importr('AlgDesign')
        xcal_in = xx.copy()


        if fixed_samples is None or fixed_samples.sum()==0:

            numpy2ri.activate()
            optfederov_output = AlgDesign.optFederov(data = xcal_in, nTrials = Nout,criterion=optimality_criterion,center=True,
                                                augment=False)

        else:

            design_augment = True    
            design_fixed_rows = np.where(fixed_samples.flatten()==1)[0]+1
            numpy2ri.activate()
            optfederov_output = AlgDesign.optFederov(data = xcal_in, nTrials = Nout,criterion=optimality_criterion,center=True, 
                                                augment=design_augment, rows=design_fixed_rows)
            numpy2ri.deactivate()
        
        
        sample_selected_id = (np.array(optfederov_output.rx("rows"))-1)[0]
        sample_selected_id = sample_selected_id.astype(int)

        sample_selected = np.zeros((Nin, 1))
        sample_selected[sample_selected_id,0] = 1
        
        Output = dict()
            
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]

        return Output
    
    
    
    # --------------------------------------------------------- MEAN SHIFT --------------------------------------------------
    
    
    def mean_shift_sample_sel(self,Nout = 10, sample_epsilon = 1, x_type = "u_scores"):
        '''
    
    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html for full documentation
            

--- Input ---


Nout: Total number of final selected samples (including fixed_samples)
sample_epsilon: tolerance in deviation of number of final Nout
x_type : "original_centered", "u_scores" (default), "t_scores"

--- Output ---


Output dict with:

'sample_id': id's of the final selected samples
'xout': xx original matrix of size Nout x K

        '''
        
        def quick_MeanShift(xx,bw = 0.1):

            ms = MeanShift(bandwidth=bw)     
            ms.fit(xx)
            labels = ms.labels_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)

            return n_clusters_ 
        
        if x_type == "original_centered":
            xx = self.get_xcal() - self.get_xcal().mean(axis=0)
        elif x_type == "u_scores":
            xx = self.get_xcal_u()
        elif x_type == "t_scores":
            xx = self.get_xcal_t()

        Nin = xx.shape[0]

        
        a = estimate_bandwidth(xx, quantile=0.03)
        b = estimate_bandwidth(xx, quantile=0.99)

        if a == 0:
            fa = quick_MeanShift(xx, 0.00001)
        else:
            fa = quick_MeanShift(xx, a)
        fb = quick_MeanShift(xx, b)

        stop = False
        iters = 0


        while not stop:

            c = (a+b)/2    
            fc = quick_MeanShift(xx, c)


            if fc < Nout:
                b = c
                diff = np.abs(fc - Nout)
            else:
                a = c
                diff = np.abs(fc - Nout)


            if diff <= sample_epsilon or iters > 100:
                stop = True
            
            iters += 1


        all_samples = np.arange(0, Nin)
        sample_selected = np.zeros((Nin,1))

        ms = MeanShift(bandwidth=c) 
        ms.fit(xx)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)

        for kk in labels_unique:    

            cluster_samples = all_samples[labels==kk]
            kernel_similarities = np.exp(-np.power(xx[labels==kk,:] - cluster_centers[kk,:],2).mean(axis=1))
            sample_selected[cluster_samples[np.argmax(kernel_similarities)],0] = 1 
        


        Output = dict()

        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]

        return Output
    
    
    
    # --------------------------------------------------- DBSCAN --------------------------------------------------------
    
    
    def dbscan_sample_sel(self,Nout = 10, sample_epsilon = 1):
        '''
    
    See 
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html and
    A Density-Based Algorithm for Discovering Clusters
in Large Spatial Databases with Noise
Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu 
Published in Proceedings of 2nd International Conference on Knowledge Discovery and Data Mining (KDD-96)

for full documentation
            
This procedure relies on euclidean distance. Therefore here U scores are used 

--- Input ---


Nout: Total number of final selected samples (including fixed_samples)
sample_epsilon: tolerance in deviation of number of final Nout

--- Output ---


Output dict with:

'sample_id': id's of the final selected samples
'xout': xx original matrix of size Nout x K
"outliers": outliers detected by the clustering 

        '''
        
        
        def dbscan_search_n(xx,eps_tol=0.09, min_samples_tol=2):       
    
            Nin = xx.shape[0]

            db = DBSCAN(eps=eps_tol, min_samples = min_samples_tol).fit(xx) # lower min_samples and higher eps means less density is needed to form a cluster
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            return [n_clusters_,n_noise_]
        


        xx = self.get_xcal_u().copy()        
        Nin = xx.shape[0]
        
        eps_range = np.linspace(0.0000000001, 0.2, 100)
        min_samples_range = np.arange(1, int(Nin*0.5))

        
        current_parms = [0.0000000001,1]
        stop = False
           
       

        for eps_kk in eps_range:            
            
            for min_samples_kk in min_samples_range:                
                n_search = dbscan_search_n(xx,eps_tol=eps_kk, min_samples_tol=min_samples_kk)                
                if np.abs(n_search[0] - Nout) <= sample_epsilon and n_search[1] <= int(Nin*0.2):
                    current_parms = [eps_kk,min_samples_kk]
                    stop = True
                if stop:
                    break
            if stop:
                break
                
                
        if not stop:
            
            print("DBSCAN: It was not possible to find a sample of Nout specified with sample_tolerance. Try changing these parameters")
            return None
        
        else:

            all_samples = np.arange(0, Nin)
            sample_selected = np.zeros((Nin,1))

            db = DBSCAN(eps=current_parms[0], min_samples = current_parms[1]).fit(xx) # lower min_samples and higher eps means less density is needed to form a cluster
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)


            for kk in range(n_clusters_):        

                cluster_samples = all_samples[labels==kk]
                similarities = -np.sqrt((np.power(xx[labels==kk,:] - xx[labels==kk,:].mean(axis=0),2).mean(axis=1)))
                sample_selected[cluster_samples[np.argmax(similarities)],0] = 1 



            Output = dict()

            Output['sample_id'] = sample_selected.astype(int)
            Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]
            Output["outliers"] = (labels==-1)*1

            return Output
    





    
    

    # --------------------------------------------------------- PLOTS ---------------------------------------------------------

    def pca_subsample(self,subsample_id=None):

        # - PCA

        plot_sample = subsample_id.flatten()==1

        pca = PCA(n_components=2)
        x_pca = self.get_xcal_t()

        fig, ax2 = plt.subplots(figsize=(15,8))
        ax2.set_title("PCA")
        ax2.plot(x_pca[:, 0], x_pca[:, 1], 'o', markerfacecolor="red",
             markeredgecolor='k', markersize=14)
        ax2.plot(x_pca[plot_sample, 0], x_pca[plot_sample, 1], 'o', markerfacecolor="blue",
             markeredgecolor='k', markersize=14)
        plt.show()

        return x_pca

    def tsne_subsample(self,perp = 10, subsample_id=None):

        # - tsne

        plot_sample = subsample_id.flatten() == 1


        tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=perp)
        x_tsne = tsne.fit_transform(self.get_xcal())

        fig, ax = plt.subplots(figsize = (15,8))
        ax.set_title("tSNE")
        ax.plot(x_tsne[:, 0], x_tsne[:, 1], 'o', markerfacecolor="red", markeredgecolor = 'k', markersize = 14)
        ax.plot(x_tsne[plot_sample, 0], x_tsne[plot_sample, 1], 'o', markerfacecolor = "blue", markeredgecolor = 'k', markersize = 14)
        plt.show()

        return x_tsne
