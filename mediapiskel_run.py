import os
import pandas as pd

video_information = pd.read_csv('/media/hannah/hdd/lsf/mediapi/mediapi-skel/video_information.csv', decimal=',')

videos = [str(v).zfill(5) for v in video_information['video']]
heights = list(video_information['height'])
widths = list(video_information['width'])
fps = list(video_information['fps'])

for i in range(len(videos)):
    if not os.path.exists('/media/hannah/hdd/lsf/mediapi/mediapi-skel/clean_op_data/'+videos[i]):
        os.makedirs('/media/hannah/hdd/lsf/mediapi/mediapi-skel/clean_op_data/'+videos[i])

    cmd = 'python clean_op_data.py ' + \
          '--openpose_folder ' + '/media/hannah/hdd/lsf/mediapi/mediapi-skel/skeleton_keypoints/' + videos[i] +'/ ' + \
        '--output_folder ' + '/media/hannah/hdd/lsf/mediapi/mediapi-skel/clean_op_data/'+videos[i] + '/ ' +\
        '--max_number_signers 2 ' + \
        '--convert_to_25fps True ' + \
        '--fps ' + str(fps[i]) + ' ' + \
        '--height ' + str(heights[i]) + ' ' + \
        '--width ' + str(widths[i]) + ' ' + \
        '--hand_motion_threshold 0.0005 '+ \
        '--hand_size_threshold 0.01 '+ \
        '--tracking_threshold 1 '+ \
        '--min_length_segment 0.25'



    os.system(cmd)

