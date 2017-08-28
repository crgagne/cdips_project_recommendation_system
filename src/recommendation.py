
import numpy as np
from IPython.core.debugger import Tracer
import pandas as pd
import bayesian_changepoint_detection.online_changepoint_detection as oncd
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from functools import partial

def get_user_buy_ranks(users_sample,user_profile,spu_fea,method,randomize_scores=False,extra_inputs={}):

    user_buy_ranks = np.empty(len(users_sample))
    no_ranks = np.empty(len(users_sample))
    for ui,user_id in enumerate(users_sample):
        print(ui)

        # rank items
        item_score_in_category = rank_candidates(user_id,user_profile,spu_fea,method=method,extra_inputs=extra_inputs,randomize_scores=randomize_scores)

        # get bought item rank and store into array
        user_buy_ranks[ui]=item_score_in_category.loc[item_score_in_category.buy==1,'rank'].as_matrix()[0]

        # get number of ranks per category
        no_ranks[ui]=item_score_in_category['rank'].max()

    return(user_buy_ranks,no_ranks,item_score_in_category)


def rank_candidates(user_id,user_profile,spu_fea,method='AverageFeatureSim',extra_inputs={},randomize_scores=False):
        '''
        Parameters:
        -----------
        user_id: str
        user_profile: DataFrame
            Can be sub-sampled

        Changes:
        ---------
        - migth use the buy item as the input.

        Return:
        -------
        ranked_candidates: DataFrame
            Returns a dictionary with (spu; rank) for each item in the candidate list
        '''
        # get bought item
        buy_spu = get_user_buy_spu(user_id,user_profile)

        # find buy item categoriy
        buy_sn = user_profile.loc[user_profile['buy_spu']==buy_spu,'buy_sn'].as_matrix()[0] # should assert they are all the same

        # find all other items in the category: CAREFUL THIS IS A SUBSET
        spus_in_category_b = user_profile.loc[user_profile.buy_sn==buy_sn,'buy_spu'].unique()
        spus_in_category_v = user_profile.loc[user_profile.view_sn==buy_sn,'view_spu'].unique()
        spus_in_category = list(set(list(spus_in_category_b)+list(spus_in_category_v))) # take intersection

        # make sure buy item is in list
        assert buy_spu in spus_in_category

        item_score_in_category = pd.DataFrame(data = spus_in_category,columns=['spu'])


        if randomize_scores:
            item_score_in_category['score'] = np.random.randn(len(item_score_in_category))
            #Tracer()()
        else:
            # for each item calculate the recommendation score (this might be similarity or some other)
            for candidate_spu in spus_in_category:
                 item_score_in_category.loc[item_score_in_category['spu']==candidate_spu,'score'] = recommendation_score(user_id,candidate_spu,user_profile,spu_fea,method=method,extra_inputs=extra_inputs)

        # rank candidates by score
        item_score_in_category['rank']=item_score_in_category['score'].rank()

        # put bought or not
        item_score_in_category['buy']=np.zeros(len(item_score_in_category))
        item_score_in_category.loc[item_score_in_category.spu==buy_spu,'buy']=1
        assert len(item_score_in_category.loc[item_score_in_category.buy==1,'rank'].as_matrix())==1 # make sure buy item is only in there once

        return(item_score_in_category)





def recommendation_score(user_id,candidate_spu,user_profile,spu_fea,method='AverageFeatureSim',extra_inputs={}):
    # Actually maybe don't do this per item.


    if method=='AverageFeatureSim':

        # get feature vec
        features_items = get_user_feature_trajectory(user_id,user_profile,spu_fea)
        # average
        avg_feature_vec = np.mean(features_items,axis=1)

        # load nn features for candidate item
        try:
            features_other = spu_fea.loc[spu_fea.spu_id==candidate_spu,'features'].as_matrix()[0] # return a 1-D np array
        except:
            features_other = np.ones(len(average_features))
            print('missing a candidates features ui:'+str(user_id)+' spu:'+str(candidate_spu))

        # calculate similarity of each candidate to average user feature
        sim = similarity(avg_feature_vec,features_other)

        return(sim)


    if method=='LastItemSim':

        # get feature vec
        features_items = get_user_feature_trajectory(user_id,user_profile,spu_fea)

        # get last item
        features_lastitem = features_items[:,-1]

        # get candidate item features
        try:
            features_candidate = spu_fea.loc[spu_fea.spu_id==candidate_spu,'features'].as_matrix()[0] # return a 1-D np array
        except:
            features_candidate = np.ones(len(features_lastitem))
            print('missing a candidates features ui:'+str(user_id)+' spu:'+str(candidate_spu))

        # calculate sim
        sim = similarity(features_lastitem,features_candidate)

        return(sim)

    if method =='AverageFeatureSim_AfterChangePoint':

        # get feature vec
        feature_by_time_matrix = get_user_feature_trajectory(user_id,user_profile,spu_fea)

        # reduce dimensionality
        feature_by_time_matrix_reduced =reduce_dimensions(feature_by_time_matrix,extra_inputs=extra_inputs)

        # idenfity change points
        out = identify_change_points(feature_by_time_matrix_reduced)
        change_points = out['change_points']
        if np.max(change_points)==1.0:
            last_change_point = np.where(change_points)[-1]
            avg_feature_vec = np.mean(feature_by_time_matrix[:,last_change_point:-1],axis=1)
        else:
            avg_feature_vec = np.mean(feature_by_time_matrix,axis=1)

        # get candidate item features
        try:
            features_candidate = spu_fea.loc[spu_fea.spu_id==candidate_spu,'features'].as_matrix()[0] # return a 1-D np array
        except:
            features_candidate = np.ones(len(features_lastitem))
            print('missing a candidates features ui:'+str(user_id)+' spu:'+str(candidate_spu))

        # calculate sim
        sim = similarity(avg_feature_vec,features_other)
        return(sim)

    #if method=='AverageFeatureSim_Weighted':

        # load user's average features that are weighted by classification problems weights (striped v not)

        # load candidates average features that are weighted by classification problem weights

    #if method=='Popularity':
        # return item based on popularity #


    #if method=='Classify_User'


def dot(K, L):
    if len(K) != len(L): return 0
    return sum(i[0]*i[1] for i in zip(K, L))

def similarity(item_1, item_2):
        return dot(item_1, item_2) / np.sqrt(dot(item_1, item_1)*dot(item_2, item_2))


def get_user_buy_spu(user_id,user_profile):
    '''Returns the FIRST buy_spu for each user'''
    trajectory = user_profile.loc[user_profile.user_id==user_id,]
    return(trajectory.buy_spu.as_matrix()[0])



def get_user_feature_trajectory(user_id,user_profile,spu_fea):

    # get his trajectory
    trajectory = user_profile.loc[user_profile.user_id==user_id,]

    # remove buy item
    trajectory = trajectory.loc[trajectory.view_spu!=trajectory.buy_spu.as_matrix()[0]]

    n_features = len(spu_fea.features.as_matrix()[0])
    n_views = len(trajectory)

    # loop through sequence of views and store feature vect into matrix
    feature_by_time_matrix = np.empty((n_features,n_views))
    for vi,view_spu in enumerate(trajectory.view_spu):
            # load features for image
            feature_by_time_matrix[:,vi] = spu_fea.loc[spu_fea.spu_id==view_spu,'features'].as_matrix()[0] # return a 1-D np array

    return(feature_by_time_matrix)

####
def reduce_dimensions(feature_by_time_matrix,extra_inputs={},method='PCA',ndim=5):
    if method=='PCA':
        # requires pca object in extra_inputs
        pca = extra_inputs['pca']
        feature_by_time_matrix_projected= pca.transform(feature_by_time_matrix.T) # not the transpose
        return(feature_by_time_matrix_projected[0:ndim,:])


def identify_change_points(feature_by_time_matrix):
        # offline method
        #Q, P, Pcp = offcd.offline_changepoint_detection(feature_by_time_matrix,
        #    partial(offcd.const_prior,
        #    l=(len(feature_by_time_matrix)+1)),
        #    offcd.gaussian_obs_log_likelihood,
        #    truncate=-40)

        # online method
        R, maxes = oncd.online_changepoint_detection(feature_by_time_matrix.T,
            partial(oncd.constant_hazard, 250),
            oncd.MV_Norm(mu=np.zeros(feature_by_time_matrix.shape[0]),
            Sigma=np.diag(np.ones(feature_by_time_matrix.shape[0])),
            n=np.array([1.0])))

        diff_in_max = np.abs(np.diff(np.argmax(R,axis=0))) # looks for differences in most likely run lengths
        expected_run_len = np.dot(R.T,np.arange(len(R)))

        # determine change point method
        change_points = np.zeros(feature_by_time_matrix.shape[1])
        change_points[diff_in_max>5]=1.0
        out = {}
        out['change_points']=change_points
        return(out)
