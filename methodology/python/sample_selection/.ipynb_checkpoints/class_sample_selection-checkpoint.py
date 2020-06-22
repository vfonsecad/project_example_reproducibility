# ############## ------------------------------------------------------------------------
#                            class sample_selection
# ############## ------------------------------------------------------------------------


import numpy as np
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
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
        return 'sample selection class'

    # --- Copy of data

    def get_xcal(self):
        ''' Get copy of xcal data '''
        return self.xcal

    def get_ycal(self):
        ''' Get copy of ycal data '''
        return self.ycal
    
    # ----------------------------- PCA REDUCTION ------------------------------------
    
    
    def get_xcal_pca_scores(self, first_ncp = 0):
        
        '''
        Dimension reduction based on X = USV' with X centered
        
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
        xx_u = U[:,first_ncp:ncp]
        xx_t = U[:,first_ncp:ncp].dot(Sval[first_ncp:ncp, first_ncp:ncp])
        
        self.xcal_u = xx_u.copy()
        self.xcal_t = xx_t.copy()      
        
        
    def get_xcal_u(self):
        ''' Get copy of xcal_u data '''
        return self.xcal_u 
    
    def get_xcal_t(self):
        ''' Get copy of xcal_t data '''
        return self.xcal_t
    
    
    # ---------------------------- random sample --------------------------------
        
    def random_sample(self, Nout = 10):
        
        
        xx = self.get_xcal() 
        Nin = xx.shape[0]
        all_samples = np.arange(0, Nin)
        
        included = np.random.choice(all_samples, size=Nout, replace=False)
        
        sample_selected = np.zeros((Nin, 1))
        sample_selected[included,0] = 1
        
        Output = dict()
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]

        return Output

        
        
        
        
    # ------------------------------------------ Kennard Stone -----------------------------

    def kennard_stone(self, Nout=10, fixed_samples = None, dim_reduction = True, distance_measure = "mahalanobis"):

        ''' 
        This algorithm corresponds to Kennard Stone CADEX alg. KENNARD. R. W. and STONE. L. (1969). Computer Aided Design of Experiments,Technometrics, 11, 137-148
        It enables the update of a current selected subset by entering fixed_samples as a 1-D array of 0's and 1's where 1 = part of current subset
        Nout is yet the total number of samples, i.e, current + to be selected in the update
        
        
        --- Input ---
        
        Nout: total number of sample selected, including fixed_samples
        fixed_samples: 1-D numpy array with 1's and 0's of shape (Nin,), where Nin is total number of samples available. 1 is specified for the initial fixed samples
        dim_reduction: use PCA scores
        distance: "mahalanobis" (default) or "euclidean". If "mahalanobis", dim_reduction is taken as True
        
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
        n_vector = list(range(Nin))

        xcal_in = xx.copy()
        
        
        # Initialize

        if fixed_samples is None or fixed_samples.flatten().sum()==0:
            iin = 0
            DD = distance.cdist(xcal_in, xcal_in.mean(axis=0).reshape((1,K)), metric = "euclidean")
            ID = DD.argmin()
            sample_selected[ID, 0] = 1
        else:
            iin = fixed_samples.sum()-1
            sample_selected = fixed_samples.copy().reshape((Nin,1))

        assert Nout >= sample_selected.flatten().sum(), "Nout must be bigger or equal the number of fixed_samples"
        
        DD_all = distance.cdist(xcal_in, xcal_in, metric = "euclidean")
        
        for ii in range(Nin):
            if sample_selected[ii,0] == 1:
                n_vector.remove(ii)
        

        while  iin < (Nout-1) and len(n_vector)>0:

            iin += 1
            DD = DD_all[sample_selected.flatten()==0,:][:,sample_selected.flatten()==1]            
            DD_row = DD.min(axis=1)
            max_DD = DD_row.max(axis=0)
            ID = DD_row.argmax(axis=0)
            sample_selected[n_vector[ID], 0] = 1
            n_vector.remove(n_vector[ID])
             
            



        Output = dict()
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]

        return Output


    # ------------------------------------------ K MEDOIDS -----------------------------


    def kmedoids_deprecated(self, Nout=10, fixed_samples = None, dim_reduction = True, distance_measure = "mahalanobis"):

        ''' This algorithm corresponds to Kmedoids, which is similar to Kmeans but selecting an actual point of the data as a center classical alg
        It enables the update of a current selected subset by entering fixed_samples as a 1-D array of 0's and 1's where 1 = part of current subset
        Nout is yet the total number of samples, i.e, current + to be selected in the update

       --- Input ---
        
        Nout: total number of sample selected, including fixed_samples
        fixed_samples: 1-D numpy array with 1's and 0's of shape (Nin,), where Nin is total number of samples available. 1 is specified for the initial fixed samples
        dim_reduction: use PCA scores
        distance: "mahalanobis" (default) or "euclidean". If "mahalanobis", dim_reduction is taken as True
        
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
        xcal_in = xx.copy()
        all_samples = np.arange(0,Nin)


        # -- Initialize

        if fixed_samples is None or fixed_samples.flatten.sum()==0:
            fixed_samples = np.zeros((Nin,1)).flatten()
            current_samples = np.empty((0,0)).flatten()

        else:
            fixed_samples = fixed_samples.flatten()
            current_samples = all_samples[np.where(fixed_samples == 1)[0]]
        

        center_id = np.concatenate((current_samples,np.random.choice(all_samples[np.where(fixed_samples == 0)[0]],int(Nout-fixed_samples.sum()),replace=False)))
       

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
    
    
    def kmedoids(self, Nout=10, dim_reduction = True, distance_measure = "mahalanobis"):

        ''' This algorithm corresponds to Kmedoids, which is similar to Kmeans but selecting an actual point of the data as a center classical alg
        
       --- Input ---
        
        Nout: total number of sample selected, including fixed_samples
        dim_reduction: use PCA scores
        distance: "mahalanobis" (default) or "euclidean". If "mahalanobis", dim_reduction is taken as True
        
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
        xcal_in = xx.copy()
        all_samples = np.arange(0,Nin)
        
        km = KMeans(n_clusters=Nout,  init='k-means++')
        km_model = km.fit(xcal_in)
        assign_clusters = km_model.fit_predict(xcal_in)
        chosen_samples = np.zeros((Nout,1), dtype = int)

        for cluster in np.unique(assign_clusters):
            cluster_samples_id = all_samples[assign_clusters == cluster]
            current_cluster_samples = xcal_in[cluster_samples_id, :]
            distances = pairwise_distances(current_cluster_samples, metric = "euclidean")
            chosen_samples[cluster] = cluster_samples_id[np.argmin(np.amax(distances, axis = 1))] 
            
        
        sample_selected = (np.isin(all_samples,chosen_samples))*1
        sample_selected.shape = (sample_selected.shape[0],1)

        Output = dict()
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[chosen_samples,:]

        return Output 
    
    
    # ---------------------------------- hierarchical clustering ----------------------------------
    
    def clustering(self, Nout=10, dim_reduction = True, distance_measure = "mahalanobis"):

        ''' This algorithm corresponds to agglomerative or hierarchical clustering. It will performe average linkage and 
        provide most central sample for each cluster
        
       --- Input ---
        
        Nout: total number of sample selected, including fixed_samples
        dim_reduction: use PCA scores
        distance: "mahalanobis" (default) or "euclidean". If "mahalanobis", dim_reduction is taken as True
        
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
            
        
        
        my_clustering = AgglomerativeClustering(n_clusters = Nout, affinity = "euclidean", linkage = "complete").fit(xx)
        
        
        Nin = xx.shape[0]
        xcal_in = xx.copy()
        all_samples = np.arange(0,Nin)
        assign_clusters = my_clustering.labels_
        chosen_samples = np.zeros((Nout,1), dtype = int)

        for cluster in np.unique(assign_clusters):
            cluster_samples_id = all_samples[assign_clusters == cluster]
            current_cluster_samples = xcal_in[cluster_samples_id, :]
            distances = pairwise_distances(current_cluster_samples, metric = "euclidean")
            chosen_samples[cluster] = cluster_samples_id[np.argmin(np.amax(distances, axis = 1))] 
        
        sample_selected = (np.isin(all_samples,chosen_samples))*1
        sample_selected.shape = (sample_selected.shape[0],1)

        Output = dict()
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[chosen_samples,:]
        
        return Output
    
    
    
    
    
    # ------------------------------------------------------- SUCCESSIVE PROJECTIONS ------------------------
    
    
    
    
    
    
    def successive_projections(self, Nout=10, fixed_samples = None, center = True):
        
        
        '''
        Successive projections alg as proposed in 
        
        Heronides Adonias Dantas Filho, Roberto Kawakami Harrop Galvão, Mário Cesar Ugulino Araújo, Edvan Cirino da Silva, 
        Teresa Cristina Bezerra Saldanha, Gledson   Emidio José, Celio Pasquini, Ivo Milton Raimundo, Jarbas José Rodrigues Rohwedder,
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
            assert (Nfixed < Nout), "Nout must be estrictly bigger than number of fixed samples"
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
            assert (Nfixed < Nout), "Nout must be estrictly bigger than number of fixed samples"
            ii = Nfixed   
            sample_selected[:,0] = (fixed_samples==1)*1


        

        p = np.zeros((Nin, Nout)) # store pure values       
        xx_means = np.abs(xx_means0)
        alpha_total = alpha_factor * np.amax(xx_means) + 0.000001


        while ii < (Nout): 
            
           
    
            candidate_ids = all_samples[sample_selected.flatten()==0]
            


            for candidate in candidate_ids:

                Y = np.concatenate((zz[:,[candidate]], zz[:,sample_selected.flatten()==1]), axis=1)
                p[candidate,ii] = np.linalg.det(Y.T.dot(Y))*xx_stds[candidate]/(xx_means[candidate]+alpha_total)
                               


            current_id = np.argmax(p[:,ii])
            
            sample_selected[current_id,0] = 1            
            
           
            ii += 1
            
            

            
        
        
        Output = dict()
            
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]
        

        return Output
    
    
    # --------------------------------------------------- OPTIMAL DESIGNS ------------------------------------------
    
    
    def optfederov_r(self, Nout=10, fixed_samples = None, optimality_criterion = "D"):
        
        '''
    
        See https://cran.r-project.org/web/packages/AlgDesign/AlgDesign.pdf for full documentation
        Here, T=US is used so that T'T is invertible
        It is also assummed to have linear, (no longer this) quadratic and cubic terms. No interactions and no intercept because data is centered


        --- Input ---


        Nout: Total number of final selected samples (including fixed_samples)
        fixed_samples: 1-D array of 0's and 1's where 1 = part of current subset
        optimality_criterion: string of optimality criterion for Federov alg. "D", "A", or "I"

        --- Output ---

        Output dict with:

        'sample_id': id's of the final selected samples
        'xout': xx matrix of size Nout x K

        '''
        
        
        #xx = np.concatenate((self.get_xcal_t(),np.power(self.get_xcal_t(),2),np.power(self.get_xcal_t(),3)), axis=1)
        xx = self.get_xcal_t()

        Nin = xx.shape[0]
        
        assert (Nout) > xx.shape[1] , "Number of final selected samples must be bigger than ncp"



        AlgDesign = importr('AlgDesign')
        xcal_in = xx.copy() - xx.mean(axis=0)


        if fixed_samples is None or fixed_samples.sum()==0:

            numpy2ri.activate()
            optfederov_output = AlgDesign.optFederov(data = xcal_in, nTrials = Nout,criterion=optimality_criterion,center=False,
                                                augment=False)

        else:

            design_augment = True    
            design_fixed_rows = np.where(fixed_samples.flatten()==1)[0]+1
            numpy2ri.activate()
            optfederov_output = AlgDesign.optFederov(data = xcal_in, nTrials = Nout,criterion=optimality_criterion,center=False, 
                                                augment=design_augment, rows=design_fixed_rows)
            numpy2ri.deactivate()
        
        
        sample_selected_id = (np.array([ii for ii in optfederov_output.rx("rows")])-1)[0]
        sample_selected_id = sample_selected_id.astype(int)

        sample_selected = np.zeros((Nin, 1))
        sample_selected[sample_selected_id,0] = 1
        
        Output = dict()
            
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]

        return Output
    
    
    
    # --------------------------------------------------------- PUCHWEIN --------------------------------------------------
    
    
    def puchwein(self, Nout = 10, factor_k = 0.05, dim_reduction = True, distance_measure = "mahalanobis"):
        
        '''
    
        See https://cran.r-project.org/web/packages/prospectr/vignettes/prospectr-intro.pdf for more documentation


        --- Input ---      
      
       
        Nout: total number of sample selected, including fixed_samples
        factor_k: factor by which multiply the distance threshold
        fixed_samples: 1-D numpy array with 1's and 0's of shape (Nin,), where Nin is total number of samples available. 1 is specified for the initial fixed samples
        dim_reduction: use PCA scores
        distance: "mahalanobis" (default) or "euclidean". If "mahalanobis", dim_reduction is taken as True
        

        --- Output ---


        Output dict with:

        'sample_id': id's of the final selected samples
        'xout': xx original matrix of size Nout x K

        '''

        
        if distance_measure == "mahalanobis":
            xx = self.get_xcal_u()
        elif dim_reduction:
            xx = self.get_xcal_t()
        else:
            xx = self.get_xcal()

        Nin = xx.shape[0]
        current_ncp = xx.shape[1]
        
        xx_mean = xx.mean(axis=0)
        xx_mean.shape = (1,xx_mean.shape[0])

        hh = distance.cdist(xx, xx_mean, metric = "euclidean")
        hh_id = np.argsort(hh, axis=0)
        d = distance.cdist(xx, xx, metric = "euclidean")[hh_id[:,0],:][:, hh_id[:,0]]
        d_ini = factor_k * np.amax([current_ncp-2,1])

        m = 1
        lsel = []
        n_sel = Nin
        
        while n_sel > Nout:
                
            dm = m * d_ini
            min_d = d[-1,:] <= dm
            sel = [hh_id[-1][0]]

            for ii in range(Nin-1, -1, -1):    
                if ii not in np.where(min_d)[0]:
                    sel.append(hh_id[ii][0])
                    min_d = np.logical_or(min_d,d[ii,:] <= dm)
            lsel.append(sel)
            n_sel = len(sel)
            m += 1
        
        
        sample_selected = np.array([ii in sel for ii in range(Nin)])
        sample_selected.shape = (sample_selected.shape[0], 1)

        Output = dict()

        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]

        return Output
    
    
    
    # --------------------------------------------------- SHENKWEST --------------------------------------------------------
    
    
    def shenkwest(self,Nout = 10, rm_outlier = False, dim_reduction = True, distance_measure = "mahalanobis"):
        
        '''
    
            See 
            https://cran.r-project.org/web/packages/prospectr/vignettes/prospectr-intro.pdf for further documentation

        This procedure relies on euclidean distance. Therefore here U scores are used. The factor d_min for minimum distance is calculated based on Nout
        with bisection 

        --- Input ---


        Nout: Total number of final selected samples (including fixed_samples)
        rm_outlier: logical default False. Remove outliers prior to sample selection
        dim_reduction: use PCA scores
        distance: "mahalanobis" (default) or "euclidean". If "mahalanobis", dim_reduction is taken as True
        

        --- Output ---


        Output dict with:

        'sample_id': id's of the final selected samples
        'xout': xx original matrix of size Nout x K
       

        '''
        
        
        if distance_measure == "mahalanobis":
            xx = self.get_xcal_u()
        elif dim_reduction:
            xx = self.get_xcal_t()
        else:
            xx = self.get_xcal()
        
        
       
        Nin = xx.shape[0]
        current_ncp = xx.shape[1]
        
        xx = (xx - xx.mean(axis=0))/(xx.std(axis=0))
        n_vector = np.arange(0,Nin)
        
        if rm_outlier:
            
            hh = distance.cdist(xx, xx.mean(axis=0), metric = "euclidean")
            hh = hh / current_ncp
            hh_id = hh <= 3
            xx = xx[hh_id,:]
            n_vector = np.where(hh_id)[0]
        
        
        # --- determine optimal factor_d_min according to Nout

        a = 0.00000000001
        b = current_ncp
        stop = False

        while not stop:

            factor_d_min = (a+b)/2
            d = (distance.cdist(xx, xx, metric = "euclidean") / current_ncp) < factor_d_min
            idx = np.argmax(d.sum(axis=0))
            current_trues = d[:,idx]
            current_rate = current_trues.sum()/Nin

            if current_rate < 1/Nout:    
                a = factor_d_min
            else:
                b = factor_d_min

            if np.abs(a - b) <= 0.001:
                stop = True




        factor_d_min = (a+b)/2
        
        
        
        d = (distance.cdist(xx, xx, metric = "euclidean") / current_ncp) < factor_d_min
        model = []

        stop = False

        while d.shape[1] > 1 and not stop:
            
    
            d_sum = d.sum(axis=0)
            idx = np.argmax(d_sum)
            current_trues = d[:,idx]
            knn = np.where(current_trues)[0]
            stop = d_sum.max(axis=0) == 1
            model.append(n_vector[idx])
            n_vector = np.delete(n_vector, axis = 0, obj = knn)
            d = d[np.where(1-current_trues)[0],:][:,np.where(1-current_trues)[0]]



        sample_selected = np.array([ii in model for ii in range(Nin)])
        sample_selected.shape = (sample_selected.shape[0], 1)

        Output = dict()

        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]

        return Output
    
        
    # --------------------------------------------------- HONIGS --------------------------------------------------------
    
    
    def honigs(self,Nout = 10):
        
        '''
    
        See 
        https://cran.r-project.org/web/packages/prospectr/vignettes/prospectr-intro.pdf for further documentation

        This procedure takes original x values

        --- Input ---


        Nout: Total number of final selected samples 

        --- Output ---


        Output dict with:

        'sample_id': id's of the final selected samples
        'xout': xx original matrix of size Nout x K

        '''
        
        
        xx = self.get_xcal()
        Nin = xx.shape[0]
        K = xx.shape[1]
        n = np.arange(0,  Nin)
        p = np.arange(0,  K)
        
        model = np.zeros((Nout,1))
        psel = np.zeros((Nout,1))

        ii = 0

        for ii in range(Nout):

            axx = np.abs(xx)
            idx = np.unravel_index(np.argmax(axx),axx.shape)
            model[ii,0] = n[idx[0]]
            psel[ii,0] = p[idx[1]]
            n = np.delete(n, obj = idx[0], axis = 0)
            weight = xx[:, idx[1]] / xx[idx[0], idx[1]]  # weighting factor
            weight.shape = (weight.shape[0],1)
            current_xx = xx[idx[0],:]
            current_xx.shape = (current_xx.shape[0], 1)
            xx2 = np.dot(current_xx, weight.T).T
            xx = xx - xx2
            p = np.delete(p, obj = idx[1], axis = 0)
            xx = np.delete(np.delete(xx, obj = idx[0], axis = 0), obj = idx[1], axis = 1)
            

        sample_selected = np.array([ii in model[:,0] for ii in range(Nin)])
        sample_selected.shape = (sample_selected.shape[0], 1)

        Output = dict()

        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]

        return Output



    
    # --------------------------------------------------- DUPLEX --------------------------------------------------------
    
    
    def duplex(self,Nout = 10, dim_reduction = True, distance_measure = "mahalanobis"):
        
        '''
    
        See 
        https://cran.r-project.org/web/packages/prospectr/vignettes/prospectr-intro.pdf for further documentation

       
        --- Input ---
        
        Nout: total number of sample selected, including fixed_samples
        dim_reduction: use PCA scores
        distance: "mahalanobis" (default) or "euclidean". If "mahalanobis", dim_reduction is taken as True
        
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
       


        def furthest_point_to_set(xp, xs):

            D = distance.cdist(xp,xs, metric = "euclidean")
            D_min = np.argmax(np.amin(D, axis = 1).flatten())
            return D_min



        n = xx.shape[0]
        n_vector = list(range(n))
        model = []
        test = []
        half = np.floor(n/2).astype(int)
        
        if Nout <= half:
            temp_Nout = Nout
        else:
            temp_Nout = n - Nout
        
        current_n_vector = n_vector.copy()    
        D = distance.cdist(xx[n_vector,:], xx[n_vector,:], metric = "euclidean") 
        id_d = np.unravel_index(np.argmax(D),D.shape)

        for ii in id_d:
            model.append(current_n_vector[ii])
            n_vector.remove(current_n_vector[ii])


        current_n_vector = n_vector.copy()    
        D = distance.cdist(xx[n_vector,:], xx[n_vector,:], metric = "euclidean") 
        id_d = np.unravel_index(np.argmax(D),D.shape)

        for ii in id_d:
            test.append(current_n_vector[ii])
            n_vector.remove(current_n_vector[ii])
    
        icount = len(model)

        while icount < temp_Nout:

            # model
            current_n_vector = n_vector.copy()         
            id_d = furthest_point_to_set(xx[n_vector,:], xx[model,:])

            model.append(current_n_vector[id_d])
            n_vector.remove(current_n_vector[id_d])

            # test
            current_n_vector = n_vector.copy()    
            id_d = furthest_point_to_set(xx[n_vector,:], xx[test,:])

            test.append(current_n_vector[id_d])
            n_vector.remove(current_n_vector[id_d])

            icount = len(model)
        
        for sample in current_n_vector:
            test.append(sample)
        

        if (Nout > half):
            print("Nout > half, therefore selection performed on half and return test instead of model")
            sample_selected = np.array([ii in test for ii in range(n)])
        else:
            sample_selected = np.array([ii in model for ii in range(n)])
        
        sample_selected.shape = (sample_selected.shape[0], 1)

        Output = dict()

        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]

        return Output
    
    # --------------------------------------------------- COVMAP --------------------------------------------------------

    
    # --------------------------------------------------- MCC SUBSAMPLE --------------------------------------------------------

    
    
    def mcc_subsample(self, Nout = 10, sigma_factor = 0.1, first_ncp = 0):
    
    
        ''' 
    
        This idea is based on the fixed size lssvm Nystrom method. the algorithm starts with a random
 selection of the subset, then one sample of included is swaped with one sample from excluded and correntropy is measured, if it increases, 
 the included sample is retained, otherwise the samples are brought back to their pool and the procedure is repeated until convergence or number of iterations

        
        
        --- Input ---
        
        Nout: total number of samples selected
        sigma_factor: Default 0.1
        
        --- Output ---
        
        Output dict with:
        
        'sample_id': id's of the final selected samples
        'xout': xx matrix of size Nout x K (original data)

        
                
        '''
        
        
        # ¡¡¡ --- !!! # ---> pca functions

        def pca(xx, ncp):

            mu = np.zeros((1, xx.shape[1]))

            X = xx.copy()
            mu[0,:] = X.mean(axis=0)
            Xc = X - mu
            U, Sval, Vt = np.linalg.svd(Xc)
            U = U[:, 0:ncp]
            Sval = Sval[0:ncp]
            Vt = Vt[0:ncp, :]
            Smat = np.zeros((ncp, ncp))
            Smat[0:ncp, 0:ncp] = np.diag(Sval)
            X_pred = U.dot(Smat).dot(Vt) + mu
            V = Vt.T.copy()
            TT = U.dot(Smat) #Xc.dot(V)

            return (V, mu, TT, X_pred)

        def pca_predict(xx_new, pca_V, mu_x, ncp_initial = 0):
            
            pca_V_sub = pca_V[ncp_initial:,:]
            TT = (xx_new - mu_x).dot(pca_V)
            xx_predicted = TT.dot(pca_V.T) + mu_x

            return (TT, xx_predicted)

        
        X = self.get_xcal()
        n = Nout
        ncp = self.ncp


        N = X.shape[0]
        all_samples = np.arange(0, N)
        total_initializations = 500

        initial_correntropy = 0

        for initialization in range(total_initializations):


            included = np.random.choice(all_samples, size=n, replace=False)
            excluded = np.array([ii for ii in all_samples if ii not in included])

            # --- pca on all samples

            V, mu, tt_all, X_predicted_all = pca(X, ncp)
            X_predicted_all_order = X_predicted_all[np.concatenate((included,excluded)),:]

            # --- pca on division

            xx_train = X[included,:]
            xx_test = X[excluded,:]

            pca_train = pca(xx_train, ncp)
            pca_test = pca_predict(xx_test, pca_train[0],pca_train[1], ncp_initial = first_ncp)
            X_pred_train  = pca_train[3]
            X_pred_test  = pca_test[1]

            X_predicted = np.concatenate((X_pred_train, X_pred_test), axis = 0)
            correntropy = np.exp((-1/(2*sigma_factor))*np.power(X_predicted_all_order - X_predicted,2).sum(axis=1)).sum(axis=0)
            
            if correntropy > initial_correntropy:
                initial_included = included.copy()
                initial_correntropy = correntropy.copy()
                
        included = initial_included.copy()
        excluded = np.array([ii for ii in all_samples if ii not in included])
        correntropy = 0
        epsilon = 100000
        iters = 0
        iters_total = 0
        
        # --- loop with selected initial
        print("--------------- mcc loop started ---------------")

        while epsilon > 0.000001 and iters < 500:
            

            iters += 1
            iters_total += 1


            candidate_included = np.random.choice(included, size=1, replace=False)
            candidate_excluded = np.random.choice(excluded, size=1, replace=False)
            candidate_included_id = np.where(included==candidate_included)[0][0]
            candidate_excluded_id = np.where(excluded==candidate_excluded)[0][0]


            temp_included = included.copy()
            temp_included[candidate_included_id] = candidate_excluded
            temp_excluded = excluded.copy()
            temp_excluded[candidate_excluded_id] = candidate_included


            X_predicted_all_order = X_predicted_all[np.concatenate((temp_included,temp_excluded)),:]

            # --- pca on division

            xx_train = X[temp_included,:]
            xx_test = X[temp_excluded,:]

            pca_train = pca(xx_train, ncp)
            pca_test = pca_predict(xx_test, pca_train[0],pca_train[1], ncp_initial = first_ncp)
            X_pred_train  = pca_train[3]
            X_pred_test  = pca_test[1]

            X_predicted = np.concatenate((X_pred_train, X_pred_test), axis = 0)
            temp_correntropy = np.exp((-1/(2*sigma_factor))*np.power(X_predicted_all_order -X_predicted,2).sum(axis=1)).sum(axis=0)


            if correntropy < temp_correntropy:
                epsilon = np.abs(temp_correntropy - correntropy)
                iters = 0
                included = temp_included.copy()
                excluded = temp_excluded.copy()
                correntropy = temp_correntropy.copy()  
                
        final_included = included.copy()
        max_correntropy = correntropy


        sample_selected = np.zeros((N, 1))
        sample_selected[final_included,0] = 1

        Output = dict()
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = self.get_xcal()[sample_selected.flatten()==1,:]
        Output['correntropy'] = max_correntropy 

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

    
    
# --------------------------------------------------------- PC compensation function ---------------------------------------------------------


def sample_selection_pc_compensation_ks(X, current_ncp, sample_proportion = 0.2):
    
    '''
    this is the pc compensation procedure using kennard stone algorithm
    
    '''

    
    xcal = X.copy()
    N = xcal.shape[0]
    initial_ncp = 1
    final_ncp = current_ncp - 1

    selected_samples = np.zeros((N,final_ncp-initial_ncp), dtype=int)


    n = int(sample_proportion*N)

    # --- get common samples

    jj = 0
    
    for ii in range(initial_ncp,final_ncp):
        
        my_sample_selection = sample_selection(xcal, ncp = current_ncp)
        my_sample_selection.get_xcal_pca_scores(first_ncp = ii - 1)
        selected_sample = my_sample_selection.kennard_stone(Nout = n, fixed_samples=None, dim_reduction = True, distance_measure = "mahalanobis")

        selected_samples[:,jj] = selected_sample["sample_id"][:,0]
        jj += 1



    common_selected_sample = 1*(selected_samples.sum(axis=1) == selected_samples.shape[1])
    current_n = common_selected_sample.sum(axis=0)

    current_selected_sample = common_selected_sample.copy()
    
    # --- start pc compensation + design augmentation
    
    step_binary = 0
    iteration_counter = 0

    while current_n < n:

        if step_binary == 0:
            lower_lim = initial_ncp
            upper_lim = final_ncp + 1
            increase_step = 1
        if step_binary == 1:
            lower_lim = final_ncp 
            upper_lim = initial_ncp -1
            increase_step = -1

        for ii in range(lower_lim,upper_lim, increase_step):


            my_sample_selection = sample_selection(xcal, ncp = current_ncp)
            my_sample_selection.get_xcal_pca_scores(first_ncp = current_ncp - ii - 1)
            selected_sample = my_sample_selection.kennard_stone(Nout = current_n + 1, fixed_samples=current_selected_sample, dim_reduction = True, distance_measure = "mahalanobis")
            current_selected_sample = selected_sample["sample_id"][:,0]
            current_n = current_selected_sample.sum(axis=0)


        iteration_counter += 1
        step_binary = np.mod(iteration_counter, 2)



    final_selected_sample = current_selected_sample.copy()   
    final_selected_sample.shape = (N, 1)
    
    return final_selected_sample

    
def sample_selection_pc_compensation_optfed(X, current_ncp, sample_proportion = 0.2):
    
    '''
    this is the pc compensation procedure using optfederov D optimal
    
    '''

    
    xcal = X.copy()
    N = xcal.shape[0]
    initial_ncp = 1
    final_ncp = current_ncp - 1

    selected_samples = np.zeros((N,final_ncp-initial_ncp), dtype=int)


    n = int(sample_proportion*N)

    # --- get common samples

    jj = 0
    
    for ii in range(initial_ncp,final_ncp):
        
        my_sample_selection = sample_selection(xcal, ncp = current_ncp)
        my_sample_selection.get_xcal_pca_scores(first_ncp = ii - 1)
        selected_sample = my_sample_selection.optfederov_r(Nout = n, fixed_samples=None, optimality_criterion='D')

        selected_samples[:,jj] = selected_sample["sample_id"][:,0]
        jj += 1



    common_selected_sample = 1*(selected_samples.sum(axis=1) == selected_samples.shape[1])
    current_n = common_selected_sample.sum(axis=0)

    current_selected_sample = common_selected_sample.copy()
    
    # --- start pc compensation + design augmentation
    
    step_binary = 0
    iteration_counter = 0

    while current_n < n:

        if step_binary == 0:
            lower_lim = initial_ncp
            upper_lim = final_ncp + 1
            increase_step = 1
        if step_binary == 1:
            lower_lim = final_ncp 
            upper_lim = initial_ncp -1
            increase_step = -1

        for ii in range(lower_lim,upper_lim, increase_step):


            my_sample_selection = sample_selection(xcal, ncp = current_ncp)
            my_sample_selection.get_xcal_pca_scores(first_ncp = current_ncp - ii - 1)
            selected_sample = my_sample_selection.optfederov_r(Nout = current_n + 1, fixed_samples=current_selected_sample, optimality_criterion='D')
            current_selected_sample = selected_sample["sample_id"][:,0]
            current_n = current_selected_sample.sum(axis=0)


        iteration_counter += 1
        step_binary = np.mod(iteration_counter, 2)



    final_selected_sample = current_selected_sample.copy()   
    final_selected_sample.shape = (N, 1)
    
    
    return final_selected_sample


# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------


# -------------------------- covmap algorithm





# -------------------------------------------------------------------


def covmap(xx,n_sel,rank_covmatrix, total_iters=4000, total_starts = 5):

    ''' 
    covmap covariance maximum approximation algorithm performs a sample selection procedure. The selected samples are such that they produce an svd
    of the covariance matrix as close as possible to the one that all samples produce
    '''
    
    # --- f1: determinant criterion
    
    def get_criterion(X,selected_rows,r):

        ''' 
        evaluate criterion based on selected rows of X with rank r

        '''

        N = X.shape[0]

        # --- split matrix

        Xs = X[selected_rows,:]
        Xus = np.delete(X, obj=selected_rows, axis=0)

        # --- cov(X_s)

        mean_Xs = Xs.mean(axis=0)
        mean_Xs.shape = (1,mean_Xs.shape[0])
        Xs_c = Xs - mean_Xs
        cov_Xs = Xs_c.T.dot(Xs_c)/Xs_c.shape[0]

        # --- svd of cov(X_s)

        Us,Ds,Vs = np.linalg.svd(cov_Xs)

        # --- apply previous svd to total matrix centering with mean_Xs

        X_cs = X - mean_Xs
        T = X_cs.dot(Vs[0:r,:].T)
        T_norm = T.dot(np.diag(np.sqrt(1/np.diag(T.T.dot(T)))))

        # --- get optimality criterion

        A = T_norm.T.dot(T_norm)
        criterion = np.linalg.det(A)

        return criterion

    # --- f2: swap rows

    def swap_candidate(all_obs, selected_obs, except_ob):

        ''' 
        randomly swap one sample currently in for one sample currently out

        '''

        return_selected = selected_obs.copy()

        except_ob_id = np.where(selected_obs==except_ob)[0][0]

        candidate_selected = np.delete(selected_obs, obj = [except_ob_id], axis=0)
        candidate_unselected = np.delete(all_obs, obj=selected_obs, axis=0)

        selected_for_out = np.random.choice(a=candidate_selected, size=1, replace=False)
        selected_for_in = np.random.choice(a=candidate_unselected, size=1, replace=False)

        return_selected[np.where(selected_obs==selected_for_out)[0][0]] = selected_for_in[0]

        return (return_selected,selected_for_in[0])

    # --- start algorithm

    X_all = xx.copy()
    n_all = X_all.shape[0]
    all_rows = np.arange(n_all)
    convergence_algorithm = np.zeros(total_iters)
       
    
    final_criterion = 0
    
    for jj in range(total_starts):

        # --- initialize random selection, row exchange and criterion

        current_selected_rows = np.random.choice(a=n_all, size=n_sel, replace=False)
        current_criterion = 0
        current_keep_row = current_selected_rows[1]

        # --- start search

        for ii in range(total_iters):

                # --- swap

            current_exchanged_rows, current_exchanged_row = swap_candidate(all_obs=all_rows, selected_obs=current_selected_rows, except_ob=current_keep_row)


                # --- function to calculate optimality criterion

            criterion_result = get_criterion(X=X_all,selected_rows=current_exchanged_rows,r=rank_covmatrix)
            convergence_algorithm[ii] = criterion_result

                # --- evaluation statement

            if criterion_result>=current_criterion:

                current_keep_row = current_exchanged_row
                current_selected_rows = current_exchanged_rows.copy()
                current_criterion = criterion_result.copy()

        if current_criterion>=final_criterion:            

            
            final_selected_rows = current_selected_rows.copy()
            final_criterion = current_criterion.copy()
            final_convergence_algorithm = convergence_algorithm.copy()
        

    
    sample_selected = (np.isin(all_rows,final_selected_rows))*1
    sample_selected.shape = (sample_selected.shape[0],1)

    output = dict()
    output['sample_id'] = sample_selected.astype(int)
    output['criterion_value'] = final_criterion
    output['convergence_algorithm'] = final_convergence_algorithm
    
    return output