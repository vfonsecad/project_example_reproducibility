# ############## ------------------------------------------------------------------------
#                            CLASS SampleSelection
# ############## ------------------------------------------------------------------------


import numpy as np
from scipy.spatial import distance
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



class SampleSelection(object):

    def __init__(self, xx, yy, sample_sel_name=""):

        ''' Initialize a Sample selection Class object with provided spectral data '''

        assert type(xx) is np.ndarray and type(yy) is np.ndarray and xx.shape[0] == yy.shape[0]

        self.xcal = xx.copy()
        self.ycal = yy.copy()
        self.Ncal = xx.shape[0]
        self.XK = xx.shape[1]
        self.YK = yy.shape[1]
        self.sample_sel_name = sample_sel_name

    def __str__(self):
        return 'SampleSelection'

    # --- Copy of data

    def get_xcal(self):
        ''' Get copy of xcal data '''
        return self.xcal

    def get_ycal(self):
        ''' Get copy of ycal data '''
        return self.ycal

    # ------------------------------------------ Kennard Stone -----------------------------

    def kennard_stone(self, xx=None, Nout=10, fixed_samples = None):

        ''' This algorithm corresponds to Kennard Stone classical alg.
        It enables the update of a current selected subset by entering fixed_samples as a 1-D array of 0's and 1's where 1 = part of current subset
        Nout is yet the total number of samples, i.e, current + to be selected in the update'''


        if xx is None:
            xx = self.get_xcal()

        Nin = xx.shape[0]
        K = xx.shape[1]
        sample_selected = np.zeros((Nin, 1))

        xcal_in = xx.copy()
        xcal_out = np.zeros((Nout, K))
        max_DD = 1000000
        # Initialize

        if fixed_samples is None or fixed_samples.flatten().sum()==0:
            iin = 0
            DD = distance.cdist(xcal_in, xcal_in.mean(axis=0).reshape((1,K)))
            id = DD.argmin()
            xcal_out[iin,:] = xcal_in[id,:]
            sample_selected[id, 0] = 1
        else:
            iin = fixed_samples.sum()-1
            xcal_out[0:(iin+1), :] = xcal_in[fixed_samples.flatten()==1, :]
            sample_selected = fixed_samples.copy().reshape((Nin,1))

        assert Nout >= sample_selected.flatten().sum()

        while  iin < (Nout-1) and max_DD > 0.00001:

            iin += 1
            DD = distance.cdist(xcal_in, xcal_in[sample_selected.flatten()==1,:])
            DD_row = DD.min(axis=1)
            max_DD = DD_row.max()
            id = DD_row.argmax()
            xcal_out[iin,:] = xcal_in[id,:]
            sample_selected[id, 0] = 1



        Output = dict()
        Output['sample_id'] = sample_selected.astype(int)
        Output['xout'] = xcal_out

        return Output


    # ------------------------------------------ K MEDOIDS -----------------------------


    def kmedoids(self,xx=None, Nout=10, fixed_samples = None):

        ''' This algorithm corresponds to Kmedoids, which is like K means but selecting an actual point of the data as a center classical alg
        It enables the update of a current selected subset by entering fixed_samples as a 1-D array of 0's and 1's where 1 = part of current subset
        Nout is yet the total number of samples, i.e, current + to be selected in the update'''




        if xx is None:
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
            DD = distance.cdist(xcal_in, centers)
            min_id = DD.argmin(axis=1)
            center_id = np.concatenate((current_samples,np.zeros((Nout-NoutCurrent, 1)).flatten()))

            for im in range(NoutCurrent,Nout):

                group = all_samples[min_id == im]


                if group.shape[0]>1:
                    DD_im = distance.cdist(xcal_in[group,:],xcal_in[group,:])
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



        Output = dict()
        Output['sample_id'] = np.isin(all_samples,center_id)
        Output['xout'] = xcal_in[center_id,:]

        return(Output)




    def pca_subsample(self, xx, subsample_id=None):

        # - PCA

        plot_sample = subsample_id.flatten()==1

        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(xx)

        fig, ax2 = plt.subplots()
        ax2.set_title("PCA")
        ax2.scatter(x_pca[:, 0], x_pca[:, 1], c="r")
        ax2.scatter(x_pca[plot_sample, 0], x_pca[plot_sample, 1], c="b")
        plt.show()

        return x_pca

    def tsne_subsample(self, xx, perp = 10, subsample_id=None):

        # - tsne

        plot_sample = subsample_id.flatten() == 1


        tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=perp)
        x_tsne = tsne.fit_transform(xx)

        fig, ax = plt.subplots()
        ax.set_title("tSNE")
        ax.scatter(x_tsne[:, 0], x_tsne[:, 1], c="r")
        ax.scatter(x_tsne[plot_sample, 0], x_tsne[plot_sample, 1], c="b")
        plt.show()

        return x_tsne
