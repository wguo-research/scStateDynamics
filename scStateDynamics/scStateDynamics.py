import os
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import pickle
import random
import ot
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from scipy import sparse
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.manifold import MDS
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from matplotlib.gridspec import GridSpec

# import logging
import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.poutine as poutine



def splitScObjects(scobj, sam_key, sam_values):
    scobj.obsm['X_pca_comb'] = scobj.obsm['X_pca']
    data_split = [scobj[scobj.obs[sam_key] == v,:] for v in sam_values]
    return(data_split)



def runScClustering(scobj, cls_res = 1, clq_res = 50):
    sc.tl.leiden(scobj, key_added='cluster', resolution=cls_res)
    sc.tl.leiden(scobj, key_added='t_clique', resolution=clq_res)
    
    add_qi = len(scobj.obs['t_clique'].unique())
    tmp_clqs = np.array(scobj.obs['t_clique'])
    for t_qi in scobj.obs['t_clique'].unique():
        cur_q_cls = scobj.obs[scobj.obs['t_clique'] == t_qi]['cluster'].unique()
        if len(cur_q_cls) > 1:
            for c_tqi in cur_q_cls[1:]:
                tmp_clqs[(scobj.obs['t_clique'] == t_qi) & (scobj.obs['cluster'] == c_tqi)] = str(add_qi)
                add_qi = add_qi + 1
    scobj.obs['clique'] = tmp_clqs
    scobj.obs = scobj.obs.drop(columns = ['t_clique'])
    
    return(scobj)


def plotScatter(sc_obj, value = 'cluster', axis_value = 'X_umap', labs = ['UMAP_1', 'UMAP_2'], 
                title = None, palette = None, legend_title = 'Cluster', saveFig = False, saveName = 'CellScatter.png'):
    p_data = pd.DataFrame({labs[0]:sc_obj.obsm[axis_value][:,0],
                              labs[1]:sc_obj.obsm[axis_value][:,1], 
                              value:list(sc_obj.obs[value])})
    p_data = p_data.sort_values(by=value, axis=0, ascending=True)
    p_ratio = p_data[labs].max() - p_data[labs].min()
    p_ratio = p_ratio[labs[0]] / p_ratio[labs[1]]

    if saveFig:
        fig = plt.figure(figsize = (3.5, 3), tight_layout = True)
    ax = sns.scatterplot(data=p_data, x=labs[0], y=labs[1], hue=value, palette=palette, 
                         s = 5, alpha = 0.6, linewidth = 0)
    if legend_title is None:
        legend_title = value
    ax.legend(title = legend_title, loc = 6, bbox_to_anchor = (1.01,0.5), ncol=1, handletextpad = 0)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, loc = 'left', fontsize = 13)
    ax.set_aspect(p_ratio)

    if saveFig:
        fig.savefig(saveName, dpi=300)
        print('| - Saving figure:', saveName)

    return(ax)



def getCliqueCls(scobj):
    cur_n_clq = len(np.unique(list(scobj.obs["clique"])))
    clq_cls = [np.argmax(np.bincount(scobj.obs['cluster_id'][scobj.obs['clique'] == str(x)])) for x in range(cur_n_clq)]
    clq_cls = np.array(clq_cls)
    return(clq_cls)



def getCliquePC(scobj, column = 'X_pca_comb'):
    pc_data = scobj.obsm[column]
    n_clq = len(np.unique(list(scobj.obs["clique"])))
    clq_pc_data = [np.average(pc_data[np.where(scobj.obs['clique'] == str(x))[0], :], axis=0) for x in range(n_clq)]
    clq_pc_data = np.array(clq_pc_data)
    return(clq_pc_data)




def getBoxOutlierThres(values, type = "up"):
    q3, q1 = np.percentile(values, [75 ,25])
    iqr = q3 - q1
    if type == "up":
        return q3 + 1.5 * iqr
    elif type == "bottom":
        return q1 - 1.5 * iqr
    else:
        print("Error! The parameter `type` can only be 'up' or 'bottom'.")




def _assignFlowType(values, method = 'Outlier', criterion = "AIC", threshold = None):
    flow_info = pd.DataFrame({'dist':values})
    if method == 'Outlier':
        threshold = getBoxOutlierThres(flow_info['dist'])
        flow_info['type'] = "keep"
        flow_info.loc[flow_info['dist'] > threshold, 'type'] = "error"
    elif method == 'GMM':
        values = np.array(flow_info['dist']).reshape(-1, 1)

        ks = np.arange(1, 4)
        models = [GaussianMixture(k, random_state=0).fit(values) for k in ks]
        
        gmm_model_comp = pd.DataFrame({"k" : ks,
                                       "BIC" : [m.bic(values) for m in models],
                                       "AIC" : [m.aic(values) for m in models]})
        best_k = gmm_model_comp['k'][np.argmin(gmm_model_comp[criterion])]
        gmm_model = models[best_k-1]

        gmm_types = gmm_model.predict(values)
        gmm_means = gmm_model.means_.reshape(1,-1)
        flow_info['type'] = np.array(['keep','change','error'])[np.argsort(np.argsort(gmm_means))[0]][gmm_types]
        threshold =  min(flow_info.loc[flow_info['type'] == 'error', 'dist'])

    elif method == 'Manual':
        if threshold is None:
            print("Error: please set the parameter 'threshold' when 'method' is 'Outlier'.")
        flow_info['type'] = "keep"
        flow_info.loc[flow_info['dist'] > threshold, 'type'] = "error"
    
    return(flow_info['type'].tolist(), threshold)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def getCliqueNormData(scobj):
    # print('sdfsd')
    # print(len(scobj.var.index[scobj.var.highly_variable]))
    # print(scobj.var.index[scobj.var.highly_variable])
    norm_data = scobj.raw.to_adata()[:, scobj.var.index[scobj.var.highly_variable]].X.toarray()
    n_clq = len(np.unique(list(scobj.obs["clique"])))
    clq_norm_data = [np.average(norm_data[np.where(scobj.obs['clique'] == str(x))[0], :], axis=0) for x in range(n_clq)]
    clq_norm_data = np.array(clq_norm_data)
    return(clq_norm_data)



def calSigScore(sc, name, geneList, geneWeight = None):
    com_genes = list(set(geneList) & set(sc.var.index))
    scores = sc.X[:, [x in com_genes for x in sc.var.index]]
    if geneWeight is not None:
        geneWeight = np.array(pd.DataFrame({'weight':geneWeight}, index = geneList).loc[com_genes, :]['weight'])
        scores = scores * geneWeight
    sc.obs[name] = np.sum(scores, axis = 1)
    return(sc)





class scStateDynamics:

    def __init__(self, data_pre, data_pos, pre_name, pos_name, run_label, pre_colors=None, pos_colors=None, 
                 cls_prefixes = ['S', 'T'], savePath = "", saveFigFormat = "png"):
        self.data_pre = data_pre
        self.data_pos = data_pos
        self.pre_name = pre_name
        self.pos_name = pos_name
        self.run_label = run_label
        self.savePath = savePath
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        self.saveFigFormat = saveFigFormat

        self.pre_colors = pre_colors
        self.pos_colors = pos_colors
        if self.pre_colors is None:
            self.pre_colors = ["#FEC643", "#437BFE", "#43FE69", "#FE6943", "#E78AC3", 
                               "#43D9FE", "#FFEC1A", "#E5C494", "#A6D854", "#33AEB1",
                               "#EA6F5D", "#FEE8C3", "#3DBA79", "#D6EBF9", "#7F6699",
                               "#cb7c77", "#68d359", "#6a7dc9", "#c9d73d", "#c555cb", 
                               "#333399", "#679966", "#c12e34", "#66d5a5", "#5599ff"]
        if self.pos_colors is None:
            self.pos_colors = ['#EE854A', '#4878D0', '#6ACC64', '#D65F5F', '#956CB4', 
                               '#82C6E2', '#D5BB67', '#8C613C', '#DC7EC0', '#797979']
        self.cls_prefixes = cls_prefixes
        self.cls_resolutions = [0.5, 0.5]
        self.clq_resolutions = [50, 50]
        self.n_clqs = np.array([])
        self.n_clus = np.array([])
        self.clq_cls_pre = pd.DataFrame()
        self.clq_cls_pos = pd.DataFrame()
        self.k = 3
        self.alpha = 10
        self.step = 5

        self.clq_PC_data = np.array([])
        self.pair_dist = np.array([])
        self.global_dist = np.array([])
        self.flowCostMax = None

        self.Align_results = []

        self.delta_x = torch.tensor([])
        self.n_factor = None
        self.trace = None
        self.FA_params = None

        self.p_clusterUMAP = None
        self.p_growRateBar = None
        self.p_flowSankey = None
        self.p_flowHeatmap = None
        self.p_flowNormalizedHeatmap = None
        self.p_flowCostHist = None
        # self.p_FlowMDS = None
        self.p_ELBO = None

        self.bool_Clustering = False
        self.bool_Distance = False
        self.bool_Align = False
        self.bool_Correct = False
        self.bool_FA = False




    def __repr__(self):
        all_str = "scStateDynamics object: %s" % self.run_label

        if self.savePath != "":
            all_str = all_str + "\n+ 0. Data information: \n  - savePath: " + self.savePath

        all_str = all_str + "\n  - Pre-data dimensions: " + str(self.data_pre.shape)
        all_str = all_str + "\n  - Post-data dimensions: " + str(self.data_pos.shape)

        if self.bool_Clustering:
            all_str = all_str + "\n+ 1. Identifying cell clusters and cell states is finished."
            all_str = all_str + "\n  - Number of clusters: " + str(self.n_clus)
            all_str = all_str + "\n  - Number of cliques: " + str(self.n_clqs)

        if self.bool_Distance:
            all_str = all_str + "\n+ 2. Measuring the distances between cell states is finished."

        if self.bool_Align:
            all_str = all_str + "\n+ 3. Aligning the cell states between two time points is finished."

        if self.bool_Correct:
            all_str = all_str + "\n+ 4. Correcting cluster proliferation or inhibition rates is finished."

        if self.bool_FA:
            all_str = all_str + "\n+ 5. Performing factor analysis on the cell-level expression changes is finished."

        return(all_str)




    ### 1. Identify cell clusters and cell states ---------------------------
    def plotPairScatter(self, value, axis_value, x_lab, y_lab, pre_colors = None, pos_colors = None, 
                        titles = None, saveFig = True, saveName = 'Cell-PairScatter.png'):
        if isinstance(value, str):
            value = [value, value]

        if pre_colors is None:
            pre_colors = self.pre_colors[0:len(self.data_pre.obs[value[0]].unique())]
        if pos_colors is None:
            pos_colors = self.pos_colors[0:len(self.data_pos.obs[value[1]].unique())]
            
        fig = plt.figure(figsize=(8, 3))
        ax1 = fig.add_subplot(1,2,1)

        if titles is None:
            pre_title = self.pre_name
            pos_title = self.pos_name
        elif type(titles) == list:
            if len(titles) == 2:
                pre_title, pos_title = titles
        else:
            print('The parameter `titles` should be a list with two items. Use the default titles instead.')
            pre_title = self.pre_name
            pos_title = self.pos_name

        plotScatter(self.data_pre, value = value[0], axis_value = axis_value, labs = [x_lab, y_lab], 
                    title = pre_title, palette = pre_colors)
        ax2 = fig.add_subplot(1,2,2)
        plotScatter(self.data_pos, value = value[1], axis_value = axis_value, labs = [x_lab, y_lab], 
                    title = pos_title, palette = pos_colors)
        fig.tight_layout()

        if saveFig:
            fig.savefig(saveName, dpi=300)
            print('| - Saving figure:', saveName)

        return(fig)



    def runClustering(self, cls_resolutions, clq_resolutions, savePath = None):
        '''
        Identify clusters and meta-cells (cliques) in the pre- and post- data of scStateDynamics object.
        The scStateDynamics object will contain the cluster and meta-cell labels. This function will also save the paired clustering plot and the text files containing the corresponding relationship between the cluster and meta-cell (clique) labels. 

        Parameters
        ----------
        cls_resolutions: 'list'
            A list with length 2, containing the resolution parameters used in the `sc.tl.leiden` function of `scanpy` to identify clusters at pre- and post-timepoint.
        clq_resolutions: 'list'
            A list with length 2, containing the resolution parameters used in the `sc.tl.leiden` function of `scanpy` to identify meta-cells (cliques) at pre- and post-timepoint. Generally, these resolutions can be set as a large value, such as 30 or 50.
        savePath: 'path-like str'
            A path to save the analysis results. The default is to save at current directory. If 'None', the default savePath in scStateDynamics object will be used.
        
        '''
        print("\n## 1. Identify cell clusters and cell states ---------------------------")
        if savePath is None:
            savePath = self.savePath

        self.cls_resolutions = cls_resolutions
        self.clq_resolutions = clq_resolutions

        self.data_pre = runScClustering(self.data_pre, self.cls_resolutions[0], self.clq_resolutions[0])
        self.data_pos = runScClustering(self.data_pos, self.cls_resolutions[1], self.clq_resolutions[1])
        self.data_pre.obs['cluster_id'] = self.data_pre.obs['cluster']
        self.data_pre.obs['cluster'] = [self.cls_prefixes[0] + i for i in self.data_pre.obs['cluster_id']]
        self.data_pos.obs['cluster_id'] = self.data_pos.obs['cluster']
        self.data_pos.obs['cluster'] = [self.cls_prefixes[1] + i for i in self.data_pos.obs['cluster_id']]

        self.n_clqs = np.array([len(self.data_pre.obs["clique"].unique()), len(self.data_pos.obs["clique"].unique())])
        self.n_clus = np.array([len(self.data_pre.obs["cluster"].unique()), len(self.data_pos.obs["cluster"].unique())])
    
        self.clq_cls_pre = pd.DataFrame({'Clique':range(self.n_clqs[0]), 'Cluster_id':getCliqueCls(self.data_pre)})
        self.clq_cls_pos = pd.DataFrame({'Clique':range(self.n_clqs[1]), 'Cluster_id':getCliqueCls(self.data_pos)})
        self.clq_cls_pre['Cluster'] = [self.cls_prefixes[0]+str(x) for x in self.clq_cls_pre['Cluster_id']]
        self.clq_cls_pos['Cluster'] = [self.cls_prefixes[1]+str(x) for x in self.clq_cls_pos['Cluster_id']]
        
        print('| - N_cells:', [self.data_pre.shape[0], self.data_pos.shape[0]])
        print('| - N_clusters:', self.n_clus)
        print('| - N_cliques:', self.n_clqs)

        saveName = savePath + self.run_label + '_Cell-PairUMAP.' + self.saveFigFormat

        self.p_clusterUMAP = self.plotPairScatter(value = 'cluster', axis_value = 'X_umap', x_lab = 'UMAP_1', y_lab = 'UMAP_2', saveName = saveName)

        self.clq_cls_pre.to_csv(savePath + self.run_label + '_clique-clsLabel_pre.txt', sep='\t', header=True, index=True)
        self.clq_cls_pos.to_csv(savePath + self.run_label + '_clique-clsLabel_pos.txt', sep='\t', header=True, index=True)

        self.bool_Clustering = True



    # def genMergeClqPC(self, column = 'X_pca_comb'):
    #     pre_clq_PC_data = getCliquePC(scobj = self.data_pre, column = column)
    #     pos_clq_PC_data = getCliquePC(scobj = self.data_pos, column = column)
    #     self.clq_PC_data = np.vstack((pre_clq_PC_data, pos_clq_PC_data))
        


    ### 2. Measure the distances between cell states ---------------------------
    def calcGlobalDist(self, k = 5, alpha = 10, step = 5, savePath = None):
        '''
        Measure the distances between cell states along the low-dimensional manifold. 
        The scStateDynamics object will contain the distance between cell states (meta-cells or cliques) along the low-dimensional manifold. This function will also save the distance heatmap and its distribution histogram.

        Parameters
        ----------
        k: 'int'
            The k-nearest-neighbor distance ϵ_k(x) will be used as an adaptive bandwidth in Gaussian kernel function.
        alpha: 'int'
            The exponent `alpha` is introduced to mitigate the heavy tail of Gaussian kernel when ϵ_k (x) is large.
        step: 'int'
            The number of random-walk steps to obtain the t-step diffusion probability.
        savePath: 'path-like str'
            A path to save the analysis results. The default is to save at current directory. If 'None', the default savePath in scStateDynamics object will be used.
        '''

        if savePath is None:
            savePath = self.savePath

        print("\n## 2. Measure the distances between cell states ---------------------------")
        pre_clq_PC_data = getCliquePC(scobj = self.data_pre, column = 'X_pca_comb')
        pos_clq_PC_data = getCliquePC(scobj = self.data_pos, column = 'X_pca_comb')
        self.clq_PC_data = np.vstack((pre_clq_PC_data, pos_clq_PC_data))

        self.k = k
        self.alpha = alpha
        self.step = step

        dist = distance.cdist(self.clq_PC_data, self.clq_PC_data, 'sqeuclidean')
        dist_ord = dist.argsort()
        k_dist = [dist[i,j] for a,(i,j) in enumerate(zip(np.array(range(0,dist.shape[0])), dist_ord[:,self.k]))]
        
        affinity = np.exp(- (dist / k_dist) ** self.alpha)
        affinity = affinity / 2 + affinity.T / 2
        diff_prob = affinity / affinity.sum(axis = 1)[:,None]
    
        diff_prob_t = diff_prob
        for ti in range(self.step-1):
            diff_prob_t = np.dot(diff_prob_t, diff_prob)
        diff_prob_t_log = np.log(diff_prob_t)
        global_dist = distance.cdist(diff_prob_t_log, diff_prob_t_log, 'euclidean')

        self.global_dist = global_dist
        self.pair_dist = global_dist[0:self.n_clqs[0], self.n_clqs[0]:global_dist.shape[0]]

        saveName = savePath + self.run_label + '_Distance-histogram.' + self.saveFigFormat
        saveName = savePath + self.run_label + '_Distance-heatmap.' + self.saveFigFormat

        self.bool_Distance = True




    ### 3. Align the cell states between two time points --------------------------------------
    def runOT(self, cls_grow_rates = None):
        a = np.array([sum(self.data_pre.obs['clique'] == str(x)) for x in range(self.pair_dist.shape[0])])
        b = np.array([sum(self.data_pos.obs['clique'] == str(x)) for x in range(self.pair_dist.shape[1])])
        if cls_grow_rates is not None:
            a = a * cls_grow_rates[self.clq_cls_pre['Cluster_id']]
        a = a / sum(a)
        b = b / sum(b)
        ot_clq = ot.emd(a, b, self.pair_dist)

        ot_cls = [[np.sum(ot_clq[np.where(self.clq_cls_pre['Cluster_id'] == i)[0]][:, np.where(self.clq_cls_pos['Cluster_id'] == j)[0]]) for j in range(self.n_clus[1])] for i in range(self.n_clus[0])]
        ot_cls = np.array(ot_cls)

        return(ot_clq, ot_cls)


    def calcFlowInfo(self, ot_clq, ot_cls, savePath = None):
        if savePath is None:
            savePath = self.savePath

        flows = [{'s':i, 't':j, 'v':ot_cls[i,j]} for i in range(ot_cls.shape[0]) for j in range(ot_cls.shape[1])]
        flows = list(np.array(flows)[np.array([x['v'] > 0 for x in flows])])
        
        flow_dists = []
        for f in flows:
            ot_flow = ot_clq[np.where(self.clq_cls_pre['Cluster_id'] == f['s'])[0]][:, np.where(self.clq_cls_pos['Cluster_id'] == f['t'])[0]]
            dist_flow = self.pair_dist[np.where(self.clq_cls_pre['Cluster_id'] == f['s'])[0]][:, np.where(self.clq_cls_pos['Cluster_id'] == f['t'])[0]]
            flow_dists.append(np.sum(dist_flow  * ot_flow) / f['v'])
        
        flow_info = {'s':[f['s'] for f in flows], 't':[f['t'] for f in flows], 'pm':[f['v'] for f in flows],
                    'dist':flow_dists}
        flow_info = pd.DataFrame(flow_info)
        
        flow_info['s_pm'] = ot_cls.sum(axis=1)[list(flow_info['s'])]
        flow_info['t_pm'] = ot_cls.sum(axis=0)[list(flow_info['t'])]
        flow_info['s_pm_perc'] = flow_info['pm'] / flow_info['s_pm']
        flow_info['t_pm_perc'] = flow_info['pm'] / flow_info['t_pm']
        all_dist = self.pair_dist.reshape(1,-1)[0]
        flow_info['dist_percent'] = [sum(all_dist < dist_cur) / len(all_dist) for dist_cur in flow_info['dist']]

        flow_info.to_csv(savePath + self.run_label + '_FlowInfo-init.txt', sep='\t', header=True, index=True)
        
        return(flow_info)



    def plotFlowCostDistr(self, round_i = -1, threshold = False, bins = None, color = "#598cbe", 
                          title = True, saveFig = True, saveName = "FlowCost-hist.png"):
        flow_info = self.getAlign_results('flow_info', round_i = round_i)
        if isinstance(threshold, bool):
            if threshold:
                threshold = self.getAlign_results('error_flow_thres', round_i = round_i)

        if bins is None:
            bins = flow_info.shape[0] // 3
            if bins < 6:
                bins = 'auto'
    
        fig = plt.figure(figsize = (2.8, 2.5))
        ax = sns.histplot(data=flow_info, x="dist", bins=bins, color=color, kde = True)
        if threshold:
            ax.axvline(threshold, color = "red", linestyle ='--')
        ax.set_title('')
        ax.set_xlabel("Average transport cost of flows")
        ax.set_ylabel("Number of flows")
        ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

        if self.flowCostMax is None:
            self.flowCostMax = max(flow_info['dist']) * 1.1
        if self.flowCostMax >= max(flow_info['dist']):
            ax.set_xlim(0, self.flowCostMax)
        else:
            ax.set_xlim(0, max(flow_info['dist']) * 1.1)

        if isinstance(title, bool):
            if title:
                if round_i == -1:
                    round_i = len(self.Align_results)
                ax.set_title('Round ' + str(round_i - 1))
        elif isinstance(title, str):
            ax.set_title(title)

        fig.tight_layout()
        if saveFig:
            fig.savefig(saveName, dpi=300)
            print('| - Saving figure:', saveName)
        return(fig)




    def plotFlowSankey(self, round_i = -1, title = True, saveFig = True, saveName = 'Flow-Sankey.png'):
        init_flow_info = self.getAlign_results('flow_info', round_i = 0)
        final_flow_info = self.getAlign_results('flow_info', round_i = round_i)
    
        p_df = final_flow_info[['s', 't', 'pm', 't_pm', 'dist']]
        p_df.insert(3, 's_pm', [init_flow_info['s_pm'][init_flow_info['s'] == x].iloc[0] for x in final_flow_info['s']])
        p_df.insert(6, 's_pm_of_cls', [p_df['pm'][p_df['s'] == x].sum() for x in p_df['s']])
        p_df.insert(7, 's_pm_in_cls', p_df['pm'] / p_df['s_pm_of_cls'])
        p_df.insert(8, 's_pm_in_source', p_df['s_pm'] * p_df['s_pm_in_cls'])
        
        p_df.insert(9, 't_pm_of_cls', [p_df['pm'][p_df['t'] == x].sum() for x in p_df['t']])
        p_df.insert(10, 't_pm_in_cls', p_df['pm'] / p_df['t_pm_of_cls'])
        p_df.insert(11, 't_pm_in_target', p_df['t_pm'] * p_df['t_pm_in_cls'])
        
        pre_fractions = np.array([init_flow_info['s_pm'][init_flow_info['s'] == x].iloc[0] for x in range(self.n_clus[0])])
        pos_fractions = np.array([init_flow_info['t_pm'][init_flow_info['t'] == x].iloc[0] for x in range(self.n_clus[1])])
        
        fig = plt.figure(figsize=(4, 4))
        gs = GridSpec(p_df.shape[0], 2, width_ratios=[2.1, 1], hspace=0.1, wspace = 0.7)
        ax = fig.add_subplot(gs[0:p_df.shape[0], 0])
        axes = [fig.add_subplot(gs[0, 1])]
        axes.extend([fig.add_subplot(gs[axi, 1], sharex = axes[0], sharey = axes[0]) for axi in range(1, p_df.shape[0])])
        
        dist_cmap = cm.get_cmap('viridis')
        dist_range = max(p_df['dist']) - min(p_df['dist'])
        smap = cm.ScalarMappable(cmap=dist_cmap)
        smap.set_array(p_df['dist'])
        
        for i in range(self.n_clus[0]):
            bottom = pre_fractions[(i+1):].sum()
            rectangle = ax.bar(x=[0], height=pre_fractions[i], bottom=bottom, color=self.pre_colors[i], 
                               edgecolor='black', fill=True, linewidth=0.7, width=0.16)
            text_y = rectangle[0].get_height() / 2 + bottom
            ax.text(x=0, y=text_y, s=str(i), horizontalalignment='center', verticalalignment='center', fontsize = 13)
        for i in range(self.n_clus[1]):
            bottom = pos_fractions[(i+1):].sum()
            rectangle = ax.bar(x=[1], height=pos_fractions[i], bottom=bottom, color=self.pos_colors[i], 
                               edgecolor='black', fill=True, linewidth=0.7, width=0.16)
            text_y = rectangle[0].get_height() / 2 + bottom
            ax.text(x=1, y=text_y, s=str(i), horizontalalignment='center', verticalalignment='center', fontsize = 13)
        
        for pos in ['right', 'top', 'bottom', 'left']:
            ax.spines[pos].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.text(x=0, y=-0.05, s=self.pre_name, horizontalalignment='center', verticalalignment='center', fontsize = 13)
        ax.text(x=1, y=-0.05, s=self.pos_name, horizontalalignment='center', verticalalignment='center', fontsize = 13)
        
        xs = np.linspace(-5, 5, num=100)
        ys = np.array([sigmoid(x) for x in xs])
        xs = xs / 10 + 0.5
        xs *= 0.83
        xs += 0.085
        
        y_start_record = [1 - pre_fractions[0:ii].sum() for ii in range(len(pre_fractions))]
        y_end_record = [1 - pos_fractions[0:ii].sum() for ii in range(len(pos_fractions))]
    
        y_up_start, y_dw_start = 1, 1
        y_up_end, y_dw_end = 1, 1
        axi = 0
        for si in range(self.n_clus[0]):
            cur_p_df = p_df.loc[p_df['s'] == si, :]
            if cur_p_df.shape[0] > 0:
                for fi in range(cur_p_df.shape[0]):
                    y_up_start = y_start_record[si]
                    y_dw_start = y_up_start - cur_p_df['s_pm_in_source'].iloc[fi]
                    y_start_record[si] = y_dw_start
                    
                    ti =  cur_p_df['t'].iloc[fi]
                    y_up_end = y_end_record[ti]
                    y_dw_end = y_up_end - cur_p_df['t_pm_in_target'].iloc[fi]
                    y_end_record[ti] = y_dw_end
    
                    y_up_start -= 0.003
                    y_dw_start += 0.004
                    y_up_end -= 0.003
                    y_dw_end += 0.004
                    
                    ys_up = y_up_start + (y_up_end - y_up_start) * ys
                    ys_dw = y_dw_start + (y_dw_end - y_dw_start) * ys
                    
                    color_s_t = [self.pre_colors[si], self.pos_colors[ti]]
                    cmap = LinearSegmentedColormap.from_list('mycmap', [color_s_t[0], color_s_t[1]])
                    grad_colors = cmap(np.linspace(0, 1, len(xs)-1))
                    grad_colors = [rgb2hex(color) for color in grad_colors]
                    for pi in range(len(xs) - 1):
                        ax.fill_between(xs[pi:(pi+2)], ys_dw[pi:(pi+2)], ys_up[pi:(pi+2)], alpha=0.7, 
                                        color=grad_colors[pi], edgecolor = None)
    #                 ax.fill_between(xs, ys_dw, ys_up, alpha=0.7, color=self.pre_colors[si])
                    
                    ot_clq = self.getAlign_results('ot_clq', round_i = round_i)
                    ot_flow = ot_clq[np.where(self.clq_cls_pre['Cluster_id'] == si)[0]][:, np.where(self.clq_cls_pos['Cluster_id'] == ti)[0]].reshape(1,-1)[0]
                    dist_flow = self.pair_dist[np.where(self.clq_cls_pre['Cluster_id'] == si)[0]][:, np.where(self.clq_cls_pos['Cluster_id'] == ti)[0]].reshape(1,-1)[0]
                    distances = dist_flow[ot_flow > 0]
                    dist_color = dist_cmap((cur_p_df['dist'].iloc[fi] - min(p_df['dist'])) / dist_range)
                    sns.histplot(distances, color = dist_color, kde = True, edgecolor="None", ax=axes[axi])
                    axes[axi].set_ylabel('  '+self.cls_prefixes[0]+str(si)+'->'+self.cls_prefixes[1]+str(ti), rotation = 0, labelpad = 20, 
                                         fontsize = 10, verticalalignment='center')
                    axes[axi].set_yticklabels([])
                    axi = axi + 1
                    
            elif cur_p_df.shape[0] == 0:
                y_up_start = y_start_record[si]
                y_dw_start = y_up_start - pre_fractions[si]
                y_start_record[si] = y_dw_start
                
                y_up_end = y_dw_end
                y_dw_end = y_up_end - 0
    
                y_up_start -= 0.003
                y_dw_start += 0.004
                y_up_end -= 0.003
                y_dw_end = y_up_end
                
                ys_up = y_up_start + (y_up_end - y_up_start) * ys
                ys_dw = y_dw_start + (y_dw_end - y_dw_start) * ys
                
                color_s_t = [self.pre_colors[si], 'white']
                cmap = LinearSegmentedColormap.from_list('mycmap', [color_s_t[0], color_s_t[1]])
                grad_colors = cmap(np.linspace(0, 1, len(xs)-1))
                grad_colors = [rgb2hex(color) for color in grad_colors]
                for pi in range(len(xs) - 1):
                    ax.fill_between(xs[pi:(pi+2)], ys_dw[pi:(pi+2)], ys_up[pi:(pi+2)], alpha=0.7, 
                                    color=grad_colors[pi], edgecolor = None)
    #             ax.fill_between(xs, ys_dw, ys_up, alpha=0.7, color=self.pre_colors[si])
    
        ax.set_ylim(-0.003,1.003)
    
        if isinstance(title, bool):
            if title:
                if round_i == -1:
                    round_i = len(self.Align_results)
                ax.set_title(' Round ' + str(round_i - 1), loc = 'left')
        elif isinstance(title, str):
            ax.set_title(' ' + title, loc = 'left')
        
        for axi in range(0, (p_df.shape[0]-1)):
            axes[axi].xaxis.set_visible(False)
        
        axes[-1].set_xlabel('Transport distance', fontsize = 11, labelpad = 1.5)
        axes[-1].set_xticklabels([])
        axes[-1].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title('Flow cost distribution')
        
        
        cbar_ax = fig.add_axes([0.82, 0.15, 0.015, 0.7])
        cbar_ax.tick_params(labelsize=9)
        fig.colorbar(smap, cax=cbar_ax, label = 'Average transport cost for each flow')
    
        fig.subplots_adjust(left=0.1, right = 0.8)
        
    #     fig.tight_layout()
        if saveFig:
            fig.savefig(saveName, dpi=300)
            print('| - Saving figure:', saveName)
    
        return(fig)



    def alignCellStates(self, savePath = None):
        '''
        Align the cell states between two time points.
        The scStateDynamics object will record the alignment results between cell states at the current last iteration round.
        
        Parameters
        ----------
        savePath: 'path-like str'
            A path to save the analysis results. The default is to save at current directory. If 'None', the default savePath in scStateDynamics object will be used.
        '''

        print("\n## 3. Align the cell states between two time points ---------------------------")
        if savePath is None:
            savePath = self.savePath

        init_ot_clq, init_ot_cls = self.runOT()
        init_flow_info = self.calcFlowInfo(ot_clq = init_ot_clq, ot_cls = init_ot_cls)
        self.Align_results = [{'round_i': 0,
                            'flow_info': init_flow_info,
                            'ot_clq': init_ot_clq,
                            'ot_cls': init_ot_cls,
                            'grow_rates': np.ones(self.n_clus[0])}]
        saveName = savePath + self.run_label + '_FlowCost-hist-0.' + self.saveFigFormat
        p_flowCostHist = self.plotFlowCostDistr(threshold = False, bins = None, color = "#7ccd7c", saveFig = False, saveName = saveName)
        p_flowSankey = self.plotFlowSankey(saveFig = False, saveName = savePath + self.run_label + '_Flow-Sankey-0.' + self.saveFigFormat)
        self.Align_results[0]['p_flowCostHist'] = p_flowCostHist
        self.Align_results[0]['p_flowSankey'] = p_flowSankey

        self.bool_Align = True



    def getAlign_results(self, variable = None, round_i = -1):
        '''
        Get the cell state alignment results.
        
        Parameters
        ----------
        variable: 'str'
            The variable name you want to extract. `flow_info`, `ot_clq`, `ot_cls`, `grow_rates`, `p_flowCostHist`, or `p_flowSankey` are allowed.
        round_i: 'int'
            The iteration round to be assign flow types. Setting as -1 indicates the last round.

        Returns
        -------
        The value of `variable` at round `round_i`.
        - `flow_info`: A pandas DataFrame recording the basic information of cell subcluster flows. The meaning of the columns:
            - 's' and 't': the source and target cluster identity of flows.
            - 'pm': the probability mass of the flow in the source distribution (pre-timepoint). 
            - 's_pm': the probability mass of the source cluster 's' in the source distribution.
            - 't_pm': the probability mass of the target cluster 't' in the target distribution.
            - 's_pm_perc': the probability mass proportion of the flow in its source cluster 's', equal to the 'pm' value divided by the 's_pm' value.
            - 't_pm_perc': the probability mass proportion of the flow in its target cluster 't', equal to the 'pm' value divided by the 't_pm' value.
            - 'dist_percent': the percentile of the flow's distance (transport cost) in the all distances between cell states.
        - 'ot_clq': the OT transport matrix at cell state (clique or meta-cell) level.
        - 'ot_cls': the OT transport matrix at cell cluster level.
        - 'grow_rates': the proliferation or inhibition rates of the pre-clusters. Greater than 1 indicates proliferation, while less than 1 indicates inhibition.
        - 'p_flowCostHist': a histogram showing the distribution of cell state changes (flow costs) of cell subcluster flows.
        - 'p_flowSankey': a Sankey plot showing the inferred cell subcluster alignment relationships (subcluster flows), and some histogram plots showing the distribution of the transport distances (costs) in each flow.
        '''
        if variable is None:
            print('You should set the `variable` parameter. `flow_info`, `ot_clq`, `ot_cls`, `grow_rates`, `p_flowCostHist`, or `p_flowSankey` are allowed.')
        else:
            return(self.Align_results[round_i][variable])




    def assignFlowType(self, round_i = -1, method = 'Outlier', criterion = "AIC", threshold = None, savePath = None):
        '''
        Identify the types of cell subcluster flows.
        The scStateDynamics object will record the identified cell subcluster flow types at the current iteration round (`round_i`).
        
        Parameters
        ----------
        round_i: 'int'
            The iteration round to be assign flow types. Setting as -1 indicates the last round.
        method: 'str'
            The method name to identify the flow types. Allowed values:
            - 'Outlier' means to detect the unreasonable flow based on the outlier detection algorithms. 
            - 'GMM' means to use the Gaussian Mixture Model (GMM) to assign the flow types. Parameter `criterion` ('AIC' or 'BIC') is used to identify the best number of components in GMM.
            - 'Manual' means to identify the unreasonable flow by setting threshold manually. Parameter `threshold` is used to input the manual setting.
        criterion: 'str'
            'AIC' and 'BIC' are available for selection to identify the best number of components in GMM. Only works if 'method' parameter is 'GMM'.
        threshold: 'float'
            Threshold to identify the unreasonable flow manually. Only works if 'method' parameter is 'Manual'.  
        savePath: 'path-like str'
            A path to save the analysis results. The default is to save at current directory. If 'None', the default savePath in scStateDynamics object will be used.
        '''

        if savePath is None:
            savePath = self.savePath

        if round_i == -1:
            round_i = len(self.Align_results)

        if isinstance(round_i, int) & (round_i > 0):
            if round_i > len(self.Align_results):
                round_i = len(self.Align_results)

            self.Align_results = self.Align_results[0:round_i]

            flow_type, error_thres = _assignFlowType(self.getAlign_results('flow_info')['dist'], method = method, 
                                                     criterion = criterion, threshold = threshold)
            self.Align_results[-1]['flow_info']['type'] = flow_type
            self.Align_results[-1]['error_flow_thres'] = error_thres
    
            saveName = savePath + self.run_label + '_FlowCost-hist-' + str(self.Align_results[-1]['round_i']) + '.' + self.saveFigFormat
            p_flowCostHist = self.plotFlowCostDistr(threshold = True, bins = None, color = "#7ccd7c", 
                                                    saveFig = False, saveName = saveName)
    
            self.Align_results[-1]['p_flowCostHist'] = p_flowCostHist
            print("| - Current error number:", sum(np.array(flow_type) == 'error'))




    def calcGrowthRates(self, flow_info):
        cls_grow_rates = np.ones(self.n_clus[0])
        error_info = flow_info[flow_info['type'] == "error"]
        for s_i in error_info['s'].unique():
            cls_grow_rates[s_i] = cls_grow_rates[s_i] - sum(error_info[error_info['s'] == s_i]['s_pm_perc'])
        
        for s_i in error_info['s'].unique():
            cur_e_info = error_info[error_info['s'] == s_i]
            for i in range(cur_e_info.shape[0]):
                t_i = cur_e_info['t'].iloc[i]
                t_n_err_info = flow_info[(flow_info['t'] == t_i) & (flow_info['type'] != "error")]
                cur_err_pm = cur_e_info.iloc[i]['pm']
                if t_n_err_info.shape[0] == 0:
                    errf_dist = [np.mean(self.pair_dist[np.where(self.clq_cls_pre['Cluster_id'] == i)[0]][:, np.where(self.clq_cls_pos['Cluster_id'] == t_i)[0]]) for i in range(self.n_clus[0])]
                    nearest_cls = np.argmin(np.array(errf_dist))
                    nearest_cls_pm = flow_info[flow_info['s'] == nearest_cls]['s_pm'].iloc[0]
                    cls_grow_rates[nearest_cls] = cls_grow_rates[np.argmin(errf_dist)] + cur_err_pm / nearest_cls_pm
                else:
                    cur_grow_pm = cur_err_pm * t_n_err_info['pm'] / sum(t_n_err_info['pm'])
                    cls_grow_rates[t_n_err_info['s']] = cls_grow_rates[t_n_err_info['s']] + cur_grow_pm / t_n_err_info['s_pm']
        return(cls_grow_rates)




    def correctGrowthDeath(self, round_i = -1, savePath = None):
        '''
        Correct the unreasonable cell flows and estimate proliferation or inhibition rates of clusters.
        The scStateDynamics object will record the estimated cluster proliferation or inhibition rates and the corrected cell alignments at the current iteration round (`round_i`).

        Parameters
        ----------
        round_i: 'int'
            The iteration round to be assign flow types. Setting as -1 indicates the last round.
        savePath: 'path-like str'
            A path to save the analysis results. The default is to save at current directory. If 'None', the default savePath in scStateDynamics object will be used.
        '''

        if savePath is None:
            savePath = self.savePath

        if round_i == -1:
            round_i = len(self.Align_results)

        if isinstance(round_i, int) & (round_i > 0):
            if round_i > len(self.Align_results):
                round_i = len(self.Align_results)

            if round_i == 1:
                print("\n## 4. Correct cluster proliferation or inhibition rates ---------------------------")

            self.Align_results = self.Align_results[0:round_i]

            flow_info = self.Align_results[-1]['flow_info']
            grow_rates = self.Align_results[-1]['grow_rates']

            print('| Repreforming OT: round ', round_i, '---------')
            grow_rates = grow_rates * self.calcGrowthRates(flow_info = flow_info)
            ot_clq, ot_cls = self.runOT(cls_grow_rates = grow_rates)
            flow_info = self.calcFlowInfo(ot_clq = ot_clq, ot_cls = ot_cls)
            flow_info = flow_info.loc[flow_info['pm'] > 5 / self.data_pre.shape[0], ]

            self.Align_results.append({'round_i': round_i,  'flow_info': flow_info,
                                    'ot_clq': ot_clq,  'ot_cls': ot_cls,
                                    'grow_rates': grow_rates})

            saveName = savePath + self.run_label + '_FlowCost-hist-', str(round_i), '.' + self.saveFigFormat
            p_flowCostHist = self.plotFlowCostDistr(threshold = False, bins = None, color = "#7ccd7c", saveFig = False, saveName = saveName)
            saveName = savePath + self.run_label + '_Flow-Sankey-' + str(round_i), '.' + self.saveFigFormat
            p_flowSankey = self.plotFlowSankey(saveFig = False, saveName = saveName)
            self.Align_results[-1]['p_flowCostHist'] = p_flowCostHist
            self.Align_results[-1]['p_flowSankey'] = p_flowSankey

        else:
            print("Error: the parameter 'round_i' should be positive integer or -1.")





    def addCellFlow(self):
        flow_info = self.getAlign_results('flow_info', round_i = -1)
        ot_clq = self.getAlign_results('ot_clq', round_i = -1)

        ot_clq_cls = np.array([ot_clq[:, np.where(self.clq_cls_pos['Cluster_id'] == j)[0]].sum(axis=1) for j in range(self.n_clus[1])]).T
        pre_ot_clq = ot_clq_cls / ot_clq_cls.sum(axis=1).repeat(self.n_clus[1]).reshape(-1,self.n_clus[1])
        pre_clq_nCell = np.array([sum(self.data_pre.obs['clique'] == str(x)) for x in range(ot_clq.shape[0])])
        pre_ot_clq = pre_ot_clq * pre_clq_nCell.repeat(self.n_clus[1]).reshape(-1,self.n_clus[1])
        pre_ot_clq = pre_ot_clq.round().astype(np.int64)
        
        cell_clq2cls = dict()
        for i in range(pre_ot_clq.shape[0]):
            cur_cells = self.data_pre.obs.index[self.data_pre.obs['clique'] == str(i)].tolist()
            random.seed(123)
            random.shuffle(cur_cells)
            csum = 0
            for j in np.where(pre_ot_clq[i] > 0)[0]:
                cell_clq2cls[str(i)+'_'+str(j)] = cur_cells[csum:(csum+pre_ot_clq[i][j])]
                csum += pre_ot_clq[i][j]
            for j in np.where(pre_ot_clq[i] == 0)[0]:
                cell_clq2cls[str(i)+'_'+str(j)] = []
        
        cell_flows = dict()
        for i in range(flow_info.shape[0]):
            s = flow_info['s'].iloc[i]
            t = flow_info['t'].iloc[i]
            cell_flows[self.cls_prefixes[0]+str(s)+'->'+self.cls_prefixes[1]+str(t)] = sum([cell_clq2cls[str(clq_i)+'_'+str(t)] for clq_i in np.where(self.clq_cls_pre['Cluster_id'] == s)[0]], [])
        # self.cell_flows = cell_flows

        ## Add cell fate to data_pre
        if 'fate' in self.data_pre.obs.columns:
            self.data_pre.obs = self.data_pre.obs.drop(columns=['fate'])
        
        fate_pd = pd.DataFrame({'barcode':[bar for key in cell_flows.keys() for bar in cell_flows[key]], 
                                'fate':[key for key in cell_flows.keys() for bar in cell_flows[key]]})
        lose_pd = pd.DataFrame({'barcode':list(set(self.data_pre.obs.index.tolist()) - set(fate_pd['barcode'])),
                                'fate':'NA'})
        fate_pd = pd.concat([fate_pd.copy(), lose_pd])
        fate_pd.index = fate_pd['barcode']
        fate_pd = fate_pd.drop(columns=['barcode'])
        
        self.data_pre.obs = pd.concat([self.data_pre.obs, fate_pd.loc[self.data_pre.obs.index]], axis=1)
        
        # return(cell_flows, ot_clq_cls)

        

    def addCellSource(self):
        flow_info = self.getAlign_results('flow_info', round_i = -1)
        ot_clq = self.getAlign_results('ot_clq', round_i = -1)

        ot_cls_clq = np.array([ot_clq[np.where(self.clq_cls_pre['Cluster_id'] == j)[0], :].sum(axis=0) for j in range(self.n_clus[0])])
        
        pos_ot_clq = ot_cls_clq / ot_cls_clq.sum(axis=0).repeat(self.n_clus[0]).reshape(-1,self.n_clus[0]).T
        pos_clq_nCell = np.array([sum(self.data_pos.obs['clique'] == str(x)) for x in range(ot_clq.shape[1])])
        pos_ot_clq = pos_ot_clq * pos_clq_nCell.repeat(self.n_clus[0]).reshape(-1,self.n_clus[0]).T
        pos_ot_clq = pos_ot_clq.round().astype(np.int64)
        
        cell_clu2clq = dict()
        for i in range(pos_ot_clq.shape[1]):
            cur_cells = self.data_pos.obs.index[self.data_pos.obs['clique'] == str(i)].tolist()
            random.seed(123)
            random.shuffle(cur_cells)
            csum = 0
            for j in np.where(pos_ot_clq[:,i] > 0)[0]:
                cell_clu2clq[str(j)+'_'+str(i)] = cur_cells[csum:(csum+pos_ot_clq[j][i])]
                csum += pos_ot_clq[j][i]
            for j in np.where(pos_ot_clq[:,i] == 0)[0]:
                cell_clu2clq[str(j)+'_'+str(i)] = []
        
        cell_sources = dict()
        for i in range(flow_info.shape[0]):
            s = flow_info['s'].iloc[i]
            t = flow_info['t'].iloc[i]
            cell_sources[self.cls_prefixes[0]+str(s)+'->'+self.cls_prefixes[1]+str(t)] = sum([cell_clu2clq[str(s)+'_'+str(clq_i)] for clq_i in np.where(self.clq_cls_pos['Cluster_id'] == t)[0]], [])
        # self.cell_sources = cell_sources

        ## Add cell source to data_pos
        if 'source' in self.data_pos.obs.columns:
            self.data_pos.obs = self.data_pos.obs.drop(columns=['source'])
        
        source_pd = pd.DataFrame({'barcode':[bar for key in cell_sources.keys() for bar in cell_sources[key]], 
                                 'source':[key for key in cell_sources.keys() for bar in cell_sources[key]]})
        lose_pd = pd.DataFrame({'barcode':list(set(self.data_pos.obs.index.tolist()) - set(source_pd['barcode'])),
                                'source':'NA'})
        source_pd = pd.concat([source_pd.copy(), lose_pd])
        source_pd.index = source_pd['barcode']
        source_pd = source_pd.drop(columns=['barcode'])
        
        self.data_pos.obs = pd.concat([self.data_pos.obs, source_pd.loc[self.data_pos.obs.index]], axis=1)

        # return(cell_sources, ot_cls_clq)



    def calcOT_clq_sp(self):
        ot_clq = self.getAlign_results('ot_clq')

        ot_clq_sp = sparse.dok_matrix(ot_clq)
        ot_clq_sp = pd.DataFrame({'i':[list(ot_clq_sp.keys())[i][0] for i in range(len(ot_clq_sp))],
                                  'j':[list(ot_clq_sp.keys())[i][1] for i in range(len(ot_clq_sp))],
                                  'v':[list(ot_clq_sp.values())[i] for i in range(len(ot_clq_sp))]})
        ot_clq_sp['i_cls'] = np.array(self.clq_cls_pre.iloc[ot_clq_sp['i'], :]['Cluster_id'])
        ot_clq_sp['j_cls'] = np.array(self.clq_cls_pos.iloc[ot_clq_sp['j'], :]['Cluster_id'])
        self.Align_results[-1]['ot_clq_sparse'] = ot_clq_sp



    def plotGrowRate(self, round_i = -1, saveFig = True, saveName = 'GrowthRate-bar.png'):
        grow_rates = self.getAlign_results('grow_rates', round_i = round_i)

        p_data = pd.DataFrame({'x':[self.cls_prefixes[0]+str(x) for x in np.arange(0, len(grow_rates))], 'y':grow_rates-1})
    
        fig = plt.figure(figsize = (0.6+0.5*len(grow_rates), 3))
        ax = sns.barplot(x='x', y='y', palette=self.pre_colors, data=p_data)
        y_format = mtick.PercentFormatter(xmax=1.0)
        ax.yaxis.set_major_formatter(y_format)
        ax.set_title(None)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.text(1.05, -0.12, self.pre_name + '-clusters', transform=ax.transAxes, va='center', ha='right', fontsize = 12)
        ax.text(0, 1.05, 'Proliferation rate', transform=ax.transAxes, va='center', ha='center', fontsize = 11)
        ax.text(0, -0.05, 'Inhibition rate', transform=ax.transAxes, va='center', ha='center', fontsize =  11)
        
        ax.spines.bottom.set_position('zero')
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_axisbelow(False)
        fig.tight_layout()
        if saveFig:
            fig.savefig(saveName, dpi=300)
            print('| - Saving figure:', saveName)

        return(fig)



    def plotFlowHeat(self, round_i = -1, title = None, saveFig = True, saveName = 'Flow-heatmap.png'):
        ot_cls = self.getAlign_results('ot_cls', round_i = round_i)

        nr,nc = ot_cls.shape

        fig, ax = plt.subplots(figsize=((nc+1.5)*0.8, 0.5*(nr+1)), tight_layout = True)
        sns.heatmap(ot_cls, annot = True, fmt=".2g", annot_kws={"size":"small"}, linewidths=.5, cmap="GnBu", ax = ax)
        if title is None:
            title = 'Optiaml transport probability'
        ax.set_title(title)
        ax.set_xlabel(self.pos_name + '-cluster', size = 11)
        ax.set_ylabel(self.pre_name + '-cluster', size = 11)
        ax.set_xticklabels([self.cls_prefixes[1]+str(i) for i in range(ot_cls.shape[1])], size = 10)
        ax.set_yticklabels([self.cls_prefixes[0]+str(i) for i in range(ot_cls.shape[0])], size = 10)
        ax.tick_params(length = 2)

        fig.tight_layout()
        if saveFig:
            fig.savefig(saveName, dpi=300)
            print('| - Saving figure:', saveName)

        return(fig)



    def plotFlowNormalizedHeat(self, round_i = -1, saveFig = True, saveName = 'Flow-NormalizedHeatmap.png'):
        ot_cls = self.getAlign_results('ot_cls', round_i = round_i)

        nr,nc = ot_cls.shape

        fig = plt.figure(figsize = ((2*nc+3)*0.8, 0.5*(nr+1)))
        
        ax1 = fig.add_subplot(1,2,1)
        sns.heatmap(ot_cls / ot_cls.sum(axis = 1).reshape(-1, 1), annot = True, fmt=".2g", 
                    annot_kws={"size":"small"}, linewidths=.5, cmap="Greens", ax = ax1)
        ax1.set_title(self.pre_name + '-cluster\'s destination fraction')
        ax1.set_xlabel(self.pos_name + '-cluster', size = 11)
        ax1.set_ylabel(self.pre_name + '-cluster', size = 11)
        ax1.set_xticklabels([self.cls_prefixes[1]+str(i) for i in range(ot_cls.shape[1])], size = 10)
        ax1.set_yticklabels([self.cls_prefixes[0]+str(i) for i in range(ot_cls.shape[0])], size = 10)
        ax1.tick_params(length = 2)
        
        ax2 = fig.add_subplot(1,2,2)
        sns.heatmap((ot_cls.T / ot_cls.sum(axis = 0).reshape(-1, 1)).T, annot = True, fmt=".2g", 
                    annot_kws={"size":"small"}, linewidths=.5, cmap="Blues", ax = ax2)
        ax2.set_title(self.pos_name + '-cluster\'s source fraction')
        ax2.set_xlabel(self.pos_name + '-cluster', size = 11)
        ax2.set_ylabel(self.pre_name + '-cluster', size = 11)
        ax2.set_xticklabels([self.cls_prefixes[1]+str(i) for i in range(ot_cls.shape[1])], size = 10)
        ax2.set_yticklabels([self.cls_prefixes[0]+str(i) for i in range(ot_cls.shape[0])], size = 10)
        ax2.tick_params(length = 2)
        
        fig.tight_layout()
        if saveFig:
            fig.savefig(saveName, dpi=300)
            print('| - Saving figure:', saveName)

        return(fig)



    def saveResults(self, savePath = None, save_label = None):
        '''
        Save the cell state alignment results.
        Parameters
        ----------
        savePath: 'path-like str'
            A path to save the analysis results. The default is to save at current directory. If 'None', the default savePath in scStateDynamics object will be used.
        save_label: 'str'
            A label to mark this running experiment.
        '''

        if savePath is None:
            savePath = self.savePath

        flow_info = self.getAlign_results('flow_info')
        grow_rates = self.getAlign_results('grow_rates')
        ot_cls = self.getAlign_results('ot_cls')
        ot_clq = self.getAlign_results('ot_clq')

        savePrefix = savePath + self.run_label + '_'
        if save_label is not None:
            savePrefix = savePrefix + save_label + '_'

        self.p_flowHeatmap = self.plotFlowHeat(saveName = savePrefix + 'Flow-heatmap.' + self.saveFigFormat)
        self.p_flowNormalizedHeatmap = self.plotFlowNormalizedHeat(saveName = savePrefix + 'Flow-NormalizedHeatmap.' + self.saveFigFormat)
        self.p_flowCostHist = self.plotFlowCostDistr(color = "#7ccd7c", saveName = savePrefix + 'FlowCost-hist.' + self.saveFigFormat)
        self.p_growRateBar = self.plotGrowRate(saveName = savePrefix + 'GrowthRate-bar.' + self.saveFigFormat)
        self.p_flowSankey = self.plotFlowSankey(saveName = savePrefix + "Flow-Sankey." + self.saveFigFormat, title = 'Cluster flows')
        # self.p_FlowMDS = self.plotFlowLandscape(saveName = savePrefix + "Flow-Landscape." + self.saveFigFormat)

        self.addCellFlow()
        self.addCellSource()
        self.calcOT_clq_sp()

        ot_clq_sp = self.getAlign_results('ot_clq_sparse')
        ot_clq_sp.to_csv(savePath + self.run_label + '_OT-clq_sparse.txt', sep='\t', header=True, index=True)
        flow_info.to_csv(savePath + self.run_label + '_FlowInfo.txt', sep='\t', header=True, index=True)
        grow_rates = pd.DataFrame(grow_rates - 1, index=[self.cls_prefixes[0] + str(ci) for ci in range(self.n_clus[0])], columns=['Grow_rate'])
        grow_rates.to_csv(savePath + self.run_label + '_GrowthRates.txt', sep='\t', header=True, index=True)

        self.bool_Correct = True

        f = open(savePrefix + 'object.pkl', 'wb')
        pickle.dump(self, f)
        f.close()




    def model(self):
        n,g = self.delta_x.shape
        ot_clq_sparse = self.getAlign_results('ot_clq_sparse')
        
        with pyro.plate('S', self.n_clus[0]):
            u = pyro.sample('u', dist.Normal(0, torch.ones(g)*1).to_event(1))
            
        with pyro.plate('T', self.n_clus[1]):
            v = pyro.sample('v', dist.Normal(0, torch.ones(g)*1).to_event(1))
        
        with pyro.plate('G', g):
            # tau = pyro.sample('tau', dist.InverseGamma(alpha_tau, beta_tau))
            tau = pyro.sample('tau', dist.InverseGamma(0.001, 0.001))

        with pyro.plate('K', self.n_factor):
            # w = pyro.sample('w', dist.Normal(0, torch.ones(g)*1).to_event(1))
            w = pyro.sample('w', dist.LogNormal(0, torch.ones(g)*1).to_event(1))
        
        with pyro.plate('N', n) as ni:
            # z = pyro.sample('z', dist.Normal(0, torch.ones(self.n_factor)*sigma_z).to_event(1))
            # z = pyro.sample('z', dist.HalfNormal(torch.ones(self.n_factor)*sigma_z).to_event(1))
            # z = pyro.sample('z', dist.HalfNormal(0，torch.ones(self.n_factor)*1))
            z = pyro.sample('z', dist.Normal(0, torch.ones(self.n_factor)*1).to_event(1))
            # z = pyro.sample('z', dist.Dirichlet(torch.ones(self.n_factor)))
            mean = torch.mm(z, w) + u[ot_clq_sparse['i_cls']] + v[ot_clq_sparse['j_cls']]
            pyro.sample('delta_x', dist.Normal(mean, torch.ones(g)*tau).to_event(1), obs=self.delta_x)



    def guide(self):
        n,g = self.delta_x.shape 

        u_loc = pyro.param('u_loc', lambda: torch.randn((self.n_clus[0],g)))
        u_scale = pyro.param('u_scale', lambda: torch.ones(g), constraint=constraints.positive)

        v_loc = pyro.param('v_loc', lambda: torch.randn((self.n_clus[1],g)))
        v_scale = pyro.param('v_scale', lambda: torch.ones(g), constraint=constraints.positive)

        # tau_loc = pyro.param('tau_loc', lambda: torch.ones(g), constraint=constraints.positive)
        # tau_scale = pyro.param('tau_scale', lambda: torch.tensor(1.), constraint=constraints.positive)
        tau_a= pyro.param('tau_a', lambda: torch.ones(g), constraint=constraints.positive)
        tau_b = pyro.param('tau_b', lambda: torch.ones(g), constraint=constraints.positive)

        ln_w_loc = pyro.param('ln_w_loc', lambda: torch.randn((self.n_factor, g)))
        ln_w_scale = pyro.param('ln_w_scale', lambda: torch.ones(g), constraint=constraints.positive)

        # z_alpha = pyro.param('z_alpha', lambda: torch.ones((n, self.n_factor)), constraint=constraints.positive)
        z_loc = pyro.param('z_loc', lambda: torch.randn((n, self.n_factor)))
        z_scale = pyro.param('z_scale', lambda: torch.ones(self.n_factor), constraint=constraints.positive)

        with pyro.plate('S', self.n_clus[0]) as i:
            u = pyro.sample("u", dist.Normal(u_loc[i], u_scale).to_event(1))

        with pyro.plate('T', self.n_clus[1]) as i:
            v = pyro.sample("v", dist.Normal(v_loc[i], v_scale).to_event(1))

        with pyro.plate('G', g) as i:
            # tau = pyro.sample('tau', dist.HalfNormal(tau_loc[i], tau_scale))
            tau = pyro.sample('tau', dist.InverseGamma(tau_a[i], tau_b[i]))

        with pyro.plate('K', self.n_factor) as i:
            # w = pyro.sample('w', dist.Normal(ln_w_loc[i], ln_w_scale).to_event(1))
            w = pyro.sample('w', dist.LogNormal(ln_w_loc[i], ln_w_scale).to_event(1))

        with pyro.plate('N', n) as i:
            z = pyro.sample('z', dist.Normal(z_loc[i], z_scale).to_event(1))
            # z = pyro.sample('z', dist.Dirichlet(z_alpha[i]))




    def factorAnalysis(self, n_factor, learningRate = 0.02, steps = 500, savePath = None):
        print("\n## 5. Perform factor analysis on the cell-level expression changes ---------------------------")
        if savePath is None:
            savePath = self.savePath

        self.n_factor = n_factor
        ot_clq_sparse = self.getAlign_results('ot_clq_sparse')

        smoke_test = ('CI' in os.environ)
        pyro.enable_validation(True)
        pyro.set_rng_seed(1)
        # logging.basicConfig(format='%(message)s', level=logging.INFO)

        norm_data_pre = getCliqueNormData(self.data_pre)
        norm_data_pos = getCliqueNormData(self.data_pos)
        self.delta_x = torch.tensor(norm_data_pos[ot_clq_sparse['j'],:] - norm_data_pre[ot_clq_sparse['i'],:])

        self.trace = poutine.trace(self.model).get_trace()
        self.trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
        # print(self.trace.format_shapes())

        
        pyro.clear_param_store()
        
        # auto_guide = pyro.infer.autoguide.AutoNormal(self.model)
        adam = pyro.optim.Adam({"lr": learningRate})  # Consider decreasing learning rate.
        elbo = pyro.infer.Trace_ELBO()
        svi = pyro.infer.SVI(self.model, self.guide, adam, elbo)
        # svi = pyro.infer.SVI(self.model, auto_guide, adam, elbo)
        
        losses = []
        for step in range(steps if not smoke_test else 2):  # Consider running for more steps.
            loss = svi.step()
            losses.append(loss)
            # if step % 10 == 0:
            #     logging.info("Elbo loss {}: {}".format(step, loss))


        fig, ax = plt.subplots(figsize=(3.5, 2), tight_layout = True)
        ax.plot(losses)
        ax.set_xlabel("SVI step")
        ax.set_ylabel("ELBO loss")
        self.p_ELBO = fig
        fig.savefig(savePath + self.run_label + '_ELBO_loss.' + self.saveFigFormat, dpi=300)
        print('| - Saving figure:', self.run_label + '_ELBO_loss.' + self.saveFigFormat)

        self.FA_params = dict(pyro.get_param_store())
        self.getFA_params('u').to_csv(savePath + self.run_label + '_FA-U.txt', sep='\t', header=True, index=True)
        self.getFA_params('v').to_csv(savePath + self.run_label + '_FA-V.txt', sep='\t', header=True, index=True)
        self.getFA_params('z').to_csv(savePath + self.run_label + '_FA-Z.txt', sep='\t', header=True, index=True)
        self.getFA_params('w').to_csv(savePath + self.run_label + '_FA-W.txt', sep='\t', header=True, index=True)

        self.bool_FA = True

        f = open(savePath + self.run_label + '_' + 'object.pkl', 'wb')
        pickle.dump(self, f)
        f.close()




    def getFA_params(self, variable):
        '''
        Get the decomposed results of the expression changes by the Bayesian factor analysis model.
        
        Parameters
        ----------
        variable: 'str'
            The parameter name you want to extract. `W`, `Z`, `U`, or `V` are allowed.

        Returns
        -------
        The pandas DataFrame of the `variable`.
        '''

        if variable.upper() == 'U':
            cls_name = ['Cluster' + str(i) for i in range(self.n_clus[0])]
            para_value = pd.DataFrame(self.FA_params['u_loc'].data.cpu().numpy(), index = cls_name, columns = self.data_pre.var.index)
            return(para_value)
        elif variable.upper() == 'V':
            cls_name = ['Cluster' + str(i) for i in range(self.n_clus[1])]
            para_value = pd.DataFrame(self.FA_params['v_loc'].data.cpu().numpy(), index = cls_name, columns = self.data_pre.var.index)
            return(para_value)
        elif variable.upper() == 'W':
            ln_w_loc = self.FA_params['ln_w_loc'].data.cpu().numpy()
            ln_w_scale = self.FA_params['ln_w_scale'].data.cpu().numpy()
            w_mean = np.exp(ln_w_loc + (ln_w_scale ** 2) / 2)
            fa_name = ['Factor' + str(i+1) for i in range(self.n_factor)]
            para_value = pd.DataFrame(w_mean, index = fa_name, columns = self.data_pre.var.index)
            return(para_value)
        elif variable.upper() == 'Z':
            fa_name = ['Factor' + str(i+1) for i in range(self.n_factor)]
            clq_name = ['Clique_pair' + str(i+1) for i in range(self.delta_x.shape[0])]
            para_value = pd.DataFrame(self.FA_params['z_loc'].data.cpu().numpy(), index = clq_name, columns = fa_name)
            return(para_value)
        else:
            return(pd.DataFrame(self.FA_params[variable].data.cpu().numpy()))



    def subClusterPCA(self):
        norm_data = self.data_pre.raw.X[:, self.data_pre.var['highly_variable']].toarray()
        fates = self.data_pre.obs['fate'].unique()
        fates = fates[fates != 'NA']
        subcls_data_pre = pd.DataFrame([np.average(norm_data[np.where(self.data_pre.obs['fate'] == x)[0], :], axis=0) for x in fates], 
                                       index = ['Pre_'+x for x in fates], columns = self.data_pre.var['highly_variable'].index)
        
        norm_data = self.data_pos.raw.X[:, self.data_pos.var['highly_variable']].toarray()
        sources = self.data_pos.obs['source'].unique()
        sources = sources[sources != 'NA']
        subcls_data_pos = pd.DataFrame([np.average(norm_data[np.where(self.data_pos.obs['source'] == x)[0], :], axis=0) for x in sources], 
                               index = ['Pos'+x for x in sources], columns = self.data_pos.var['highly_variable'].index)

        subcls_data = pd.concat([subcls_data_pre, subcls_data_pos])





def createScStateDynamicsObj(sc_obj_comb, run_label, key, pre_name, pos_name, pre_colors=None, pos_colors=None, 
    cls_prefixes = ['S', 'T'], savePath = "", saveFigFormat = "png"):
    '''
    Creat a scStateDynamics object based on the scRNA-seq data at pre-timepoint and post-timepoint, and add the meta-information into it.

    Parameters
    ----------
    sc_obj_comb: 'AnnData'
        A scanpy object including the scRNA-seq data of both pre-timepoint and post-timepoint.
    key: 'str'
        The name of the column recording the sample source (time point information) in sc_obj_comb.
    pre_name: 'str'
        The name of the pre-timepoint data.
    pos_name: 'str'
        The name of the post-timepoint data.
    pre_colors:'list'
        A list of colors to map the cluster at pre-timepoint.
    pos_colors:'list'
        A list of colors to map the cluster at post-timepoint.
    cls_prefixes:'list'
        A list with length 2, containing the prefixes of the cluster labels at pre-timepoint and post-timepoint. The default is 'S' and 'T'.
    savePath: 'path-like str'
        A path to save the analysis results. The default is to save at current directory.
    saveFigFormat: 'str'
        The figure format suffix, such as 'png' or 'pdf'.

    Returns
    -------
    An initialized scStateDynamics object. 
    '''
    data_pre, data_pos = splitScObjects(sc_obj_comb, sam_key = key, sam_values = [pre_name, pos_name])
    scd = scStateDynamics(data_pre, data_pos, pre_name, pos_name, run_label, pre_colors, pos_colors, cls_prefixes, savePath, saveFigFormat)
    return(scd)