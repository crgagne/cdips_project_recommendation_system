
import numpy as np
from IPython.core.debugger import Tracer
import pandas as pd
import bayesian_changepoint_detection.online_changepoint_detection as oncd
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from functools import partial



def dot(K, L):
    if len(K) != len(L): return 0
    return sum(i[0]*i[1] for i in zip(K, L))

def similarity(item_1, item_2):
        return dot(item_1, item_2) / np.sqrt(dot(item_1, item_1)*dot(item_2, item_2))



class User:
    def __init__(self,user_id,user_profile,spu_fea,extra_inputs={}):
        self.user_id=user_id
        self.user_profile=user_profile
        self.spu_fea = spu_fea
        self.extra_inputs=extra_inputs

    def get_user_feature_trajectory(self):

        # get his trajectory
        trajectory = self.user_profile.loc[self.user_profile.user_id==self.user_id,]

        # remove buy item
        trajectory = trajectory.loc[trajectory.view_spu!=trajectory.buy_spu.as_matrix()[0]]

        n_features = len(self.spu_fea.features.as_matrix()[0])
        n_views = len(trajectory)

        # loop through sequence of views and store feature vect into matrix
        self.feature_by_time_matrix = np.empty((n_features,n_views))
        for vi,view_spu in enumerate(trajectory.view_spu):
                # load features for image
                self.feature_by_time_matrix[:,vi] = self.spu_fea.loc[self.spu_fea.spu_id==view_spu,'features'].as_matrix()[0] # return a 1-D np array

        # get spu sequence
        self.view_spu_by_time_list = trajectory.view_spu.as_matrix()

    def create_candidates(self):

        # get bought item
        self.buy_spu  = self.user_profile.loc[self.user_profile.user_id==self.user_id,'buy_spu'].as_matrix()[0]

        # find buy item categoriy
        self.buy_sn = self.user_profile.loc[self.user_profile['buy_spu']==self.buy_spu,'buy_sn'].as_matrix()[0] # should assert they are all the same

        # find all other items in the category: CAREFUL THIS IS A SUBSET
        spus_in_category_b = self.user_profile.loc[self.user_profile.buy_sn==self.buy_sn,'buy_spu'].unique()
        spus_in_category_v = self.user_profile.loc[self.user_profile.view_sn==self.buy_sn,'view_spu'].unique()
        self.spus_in_category = list(set(list(spus_in_category_b)+list(spus_in_category_v))) # take intersection

        # make sure buy item is in list
        assert self.buy_spu in self.spus_in_category
        self.item_score_in_category = pd.DataFrame(data = self.spus_in_category,columns=['spu'])

        # put bought or not
        self.item_score_in_category['buy']=np.zeros(len(self.item_score_in_category))
        self.item_score_in_category.loc[self.item_score_in_category.spu==self.buy_spu,'buy']=1
    ####
    def reduce_dimensions(self):
        if 'dr_params' in self.extra_inputs.keys():
            dr_params = self.extra_inputs['dr_params']
            method=dr_params['method']
            ndims=dr_params['ndims']
        else: # defaults
            ndims=5
            method='PCA'

        if method=='PCA':
            #print('reducing feature dimensions (pca)')
            # requires pca object in extra_inputs
            pca = self.extra_inputs['pca']
            feature_by_time_matrix_projected= pca.transform(self.feature_by_time_matrix.T) # not the transpose
            self.feature_by_time_matrix_reduced = feature_by_time_matrix_projected[:,0:ndims]

    def identify_change_points(self):
        #print('estimating change points')

        if 'cp_params' in self.extra_inputs.keys():
            cp_params = self.extra_inputs['cp_params']
            method = cp_params['method']
        else: # defaults
            method=='Online'


        if method=='Online': # online method
            R, maxes = oncd.online_changepoint_detection(self.feature_by_time_matrix_reduced,
                partial(oncd.constant_hazard, 250),
                oncd.MV_Norm(mu=np.zeros(self.feature_by_time_matrix_reduced.shape[1]),
                Sigma=5.0*np.diag(np.ones(self.feature_by_time_matrix_reduced.shape[1])),
                n=np.array([1.0])))

            diff_in_max = np.abs(np.diff(np.argmax(R,axis=0))) # looks for differences in most likely run lengths
            expected_run_len = np.dot(R.T,np.arange(len(R)))
            self.R = R
            self.diff_in_max = diff_in_max
            self.expected_run_len = expected_run_len
            # calculate change points
            self.change_points = np.zeros(self.feature_by_time_matrix_reduced.shape[0])
            self.change_points[diff_in_max>5]=1.0
        elif method=='Offline':
            Q, P, Pcp = offcd.offline_changepoint_detection(self.feature_by_time_matrix_reduced,partial(offcd.const_prior, l=(self.feature_by_time_matrix_reduced.shape[0]+1)),offcd.fullcov_obs_log_likelihood, truncate=-20)
            self.cp_prob = np.exp(Pcp).sum(0)
            self.change_points = self.cp_prob>0.7



    def rank_candidates(self,method='AverageFeatureSim'):

        #print(method)
        # calculate things used for each method
        self.pre_calculate(method)

        # loop through candidates and score
        #print('scoring candidates')
        for candidate_spu in self.spus_in_category:
             self.item_score_in_category.loc[self.item_score_in_category['spu']==candidate_spu,'score'] = self.recommendation_score(candidate_spu,method=method)

        # rank candidates by score
        self.item_score_in_category['rank']=self.item_score_in_category['score'].rank()

        assert len(self.item_score_in_category.loc[self.item_score_in_category.buy==1,'rank'].as_matrix())==1 # make sure buy item is only in there once

        # calculate things to aggregate across people
        self.user_buy_rank=self.item_score_in_category.loc[self.item_score_in_category.buy==1,'rank'].as_matrix()[0]
        self.no_ranks=self.item_score_in_category['rank'].max()


    def pre_calculate(self,method):

        # commmon precalculation tasks
        #print('getting feature sequence')
        self.get_user_feature_trajectory()

        if method=='AverageFeatureSim':
            # average
            self.avg_feature_vec = np.mean(self.feature_by_time_matrix,axis=1)

        if method=='AverageFeatureSim_AfterChangePoint':
            # reduce dimensionality
            self.reduce_dimensions()
            # idenfity change points
            self.identify_change_points()
            if np.max(self.change_points)==1.0:
                last_change_point = np.where(self.change_points)[0][-1]
                self.avg_feature_vec = np.mean(self.feature_by_time_matrix[:,int(last_change_point):-1],axis=1)
            else:
                self.avg_feature_vec = np.mean(self.feature_by_time_matrix,axis=1)
                #print('no change points')

        if method =='AverageFeatureSim_withinSegments':
            # reduce dimensionality
            self.reduce_dimensions()
            # idenfity change points
            self.identify_change_points()

            # split feature sequence by change-points (if no change-points, this will return the original array)
            subarrays = np.split(self.feature_by_time_matrix,np.where(self.change_points)[0],axis=1)

            # average each segment and store
            self.avg_feature_vec_segments = []
            for subarray in subarrays:
                self.avg_feature_vec_segments.append(np.mean(subarray,axis=1))



    def recommendation_score(self,candidate_spu,method='AverageFeatureSim'):

        if method=='Random':
            return(np.random.randn(1))

        if method=='AverageFeatureSim':

            # load nn features for candidate item
            features_candidate = self.spu_fea.loc[self.spu_fea.spu_id==candidate_spu,'features'].as_matrix()[0] # return a 1-D np array

            # calculate similarity of each candidate to average user feature
            return(similarity(self.avg_feature_vec,features_candidate))


        if method=='LastItemSim':

            # get last item
            features_lastitem = self.feature_by_time_matrix[:,-1]

            # get candidate item features
            features_candidate = self.spu_fea.loc[self.spu_fea.spu_id==candidate_spu,'features'].as_matrix()[0] # return a 1-D np array

            # calculate sim
            return(similarity(features_lastitem,features_candidate))

        if method=='Last5ItemSim':

            # get last item
            features_lastitem = np.mean(self.feature_by_time_matrix[:,-5:],axis=1)

            # get candidate item features
            features_candidate = self.spu_fea.loc[self.spu_fea.spu_id==candidate_spu,'features'].as_matrix()[0] # return a 1-D np array

            # calculate sim
            return(similarity(features_lastitem,features_candidate))


        if method =='AverageFeatureSim_AfterChangePoint':

            # get candidate item features
            features_candidate = self.spu_fea.loc[self.spu_fea.spu_id==candidate_spu,'features'].as_matrix()[0] # return a 1-D np array

            # calculate sim
            return(similarity(self.avg_feature_vec,features_candidate))

        if method =='AverageFeatureSim_withinSegments':

            # get candidate item features
            features_candidate = self.spu_fea.loc[self.spu_fea.spu_id==candidate_spu,'features'].as_matrix()[0] # return a 1-D np array

            # calculate similarity of each candidate to each segment
            similarities = np.array([])
            for avg_feature_segment in self.avg_feature_vec_segments:
                similarities = np.append(similarities,similarity(avg_feature_segment,features_candidate))

            #Tracer()()
            # consider maxing... or some other aggregation function.

            return(np.mean(similarities))


        if method =='Item-CF':

            # requires item similarity matrix
            item_item_similarity = self.extra_inputs['item_item_similarity']

            # for this candidate
            # loop through the viewed items and get similarty
            item_sim = np.empty(len(self.view_spu_by_time_list))
            for vi,view_spu in enumerate(self.view_spu_by_time_list):
                #calculate the similarity
                if candidate_spu in item_item_similarity:
                    item_sim[vi] = item_item_similarity[candidate_spu][view_spu]
                else:
                    item_sim[vi]=0.0
            return(item_sim.mean())
