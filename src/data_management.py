
import glob
import numpy as np
import pandas as pd


def get_meta_data_from_sequence_data(folder):
    '''
    Parameters:
    -----------
    folder_path: str
        path of the input folder (should be the image view sequence folder)
        E.g. "data_img_sample_item_view_sequences"

    Return:
    -------
    df: DataFrame
        Returns a dataframe with the buyer id, view sequence etc. for all images in the view sequence folder
    '''
    filelist = glob.glob(folder+'*')
    for f,filee in enumerate(filelist):
        single_row = filee.split('/')[2].split('_')# split into list with user id, view number, etc.
        if f==0:
            tmp_array = single_row
        else:
            tmp_array = np.vstack((tmp_array,single_row))
    df = pd.DataFrame(tmp_array,columns=['user_id','view_position','spu','view_seconds','tag','jpg'])
    df['view_seconds']=df['view_seconds'].astype('int')
    df['view_position']=df['view_position'].astype('int')
    
    return(df)
