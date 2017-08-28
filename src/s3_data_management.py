

import boto
import sys

def push_results_to_s3(fname,fpath):
    '''
    Example: push_results_to_s3('../notebook_htmls/Notebook_Exploring_New_Data_v1.html',
    'Notebook_Exploring_New_Data_v1.html')

    Requires S3 key in user's home directory in a folder ~/.aws

    '''

    def percent_cb(complete, total):
        sys.stdout.write('.')
        sys.stdout.flush()

    s3_connection = boto.connect_s3()
    bucket = s3_connection.get_bucket('bishopbucket')
    key = boto.s3.key.Key(bucket, 'proj_cdips/Results/'+fname)
    key.set_contents_from_filename(fpath,
        cb=percent_cb, num_cb=10)
    s3_connection.close()
    print('')
    print('https://s3-us-west-2.amazonaws.com/bishopbucket/proj_cdips/Results/'+fname)
