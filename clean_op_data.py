import yaml
import os
from utils import *
from scipy.signal import savgol_filter
import pickle
from sklearn.impute import KNNImputer
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to .yaml config file', default=None)
parser.add_argument('--openpose_folder', type=str, help='Folder containing openpose files *_keypoints.json',
                    default='skeleton_keypoints/')
parser.add_argument('--output_folder', type=str, help='Folder to save results',
                    default='output/')
parser.add_argument('--max_number_signers', type=int, default=1, help='Number of signers to keep')
parser.add_argument('--fps', type=float, default=25., help='Framerate of video')
parser.add_argument('--convert_to_25fps', default=False, help='Convert 30fps to 25fps')
parser.add_argument('--height_video', type=int, default=1080, help='Pixel height of video')
parser.add_argument('--width_video', type=int, default=1920, help='Pixel width of video')
parser.add_argument('--hand_motion_threshold', type=float, default=0.0005,
                    help='minimum variance of y-axis movement of dominant wrist per frame as % of height of video')
parser.add_argument('--hand_size_threshold', type=float, default=0.01,
                    help='minimum average y-axis height of dominant hand as % of height of video')
parser.add_argument('--tracking_threshold', type=float, default=1.,
                    help='tracking_threshold/fps is the maximum distance a person can move between two frames')
parser.add_argument('--min_length_segment', type=float, default=0.25,
                    help='The minimum length of segment in seconds')

args = parser.parse_args()
args_list = vars(args)

if args_list['config'] is not None:
    print('WARNING: In case of conflict, .yaml arguments override command line arguments')
    with open(args_list['config']) as file:
        args_list_yml = yaml.load(file, Loader=yaml.FullLoader)

    args_list = {**args_list, **args_list_yml} ### the second list overwrites the first

print(args_list)

### arguments ###
openpose_folder = args_list['openpose_folder']
output_folder = args_list['output_folder']
max_number_signers = args_list['max_number_signers']
fps = args_list['fps']
height_video = args_list['height_video']
width_video = args_list['width_video']

convert_to_25fps = args_list['convert_to_25fps']

hand_motion_threshold = args_list['hand_motion_threshold']
hand_size_threshold = args_list['hand_size_threshold']

tracking_threshold = args_list['tracking_threshold']
min_length_segment = args_list['min_length_segment']

#### get list of openpose files from folder files
list_op_files = []
for file in os.listdir(openpose_folder):
    if file.endswith('keypoints.json'):
        list_op_files.append(file)
list_op_files = sorted(list_op_files)

#### get dictionary of all people in OP files and track people
full_dict = get_full_dict(list_op_files=list_op_files, data_location=openpose_folder, width=width_video, height=height_video, fps=fps, tolerance=tracking_threshold)

#### get scenes = group of frames with same people in consecutive frames
scene_changes = [0]
for fno in range(len(full_dict)):
    temp_sc = 1
    for pno in range(len(full_dict[fno]) - 1):  ## minus 1 due to adding number of people in dict
        if not np.isnan(full_dict[fno][pno]['next_person']): ## if there is a next person for someone in frame
            temp_sc = 0
    if temp_sc == 1:
        scene_changes.append(fno)

scene_changes = np.array(scene_changes)
scene_changes_diff = np.diff(scene_changes)
scenes = []
for i in range(len(scene_changes)-1):
    if scene_changes_diff[i]>fps*min_length_segment: ### don't follow if less than min length segment
        scenes.append([scene_changes[i], scene_changes[i+1]])

print('scenes ', scenes)
### loop over scenes
for s in scenes:

    ### get a list of people and frames in scene
    list_frames_people = []
    for fno in range(s[0],s[1]):
        for pno in range(len(full_dict[fno]) - 1):  ## minus 1 due to adding number of people in dict
            list_frames_people.append([fno, pno])

    data_list = []
    stats_list = []

    ##### while there are still entries in the dictionary
    while len(list_frames_people) > 0:
        # print('len list frames ', len(list_frames_people))
        fno = list_frames_people[0][0]
        pno = list_frames_people[0][1]
        if fno < (len(list_op_files) - 1):
            follow_person = []
            end_follow = 0
            while end_follow != 1:
                # print(fno, pno)
                if ([fno, pno] in list_frames_people):
                    list_frames_people.remove([fno, pno])

                data_numpy = np.zeros((3, 127, 1))
                pose = full_dict[fno][pno]['pose']
                score = full_dict[fno][pno]['score']
                data_numpy[0, :, 0] = pose[0::2]
                data_numpy[1, :, 0] = pose[1::2]
                data_numpy[2, :, 0] = score

                follow_person.append((fno, pno, data_numpy))

                if np.isnan(full_dict[fno][pno]['next_person']):
                    end_follow == 1
                    break
                else:
                    pno = full_dict[fno][pno]['next_person']
                    fno += 1

            ### check statistics and drop else
            ### make numpy df
            comb_data_numpy = []
            for j in range(len(follow_person)):
                comb_data_numpy.append(follow_person[j][2])
            comb_data_numpy = np.array(comb_data_numpy)

            #### length of sequence
            len_seq = comb_data_numpy.shape[0]
            height, movement = compute_size_movement(comb_data_numpy)

            if (len_seq >= min_length_segment*fps) & (height >= hand_size_threshold) & (movement >= hand_motion_threshold):
                ### add to list

                fnos = [j[0] for j in follow_person]
                pnos = [j[1] for j in follow_person]

                data_list.append([fnos, pnos, comb_data_numpy])

                ### height*movement is most likely signer measure
                stats_list.append([len_seq, height, movement, height*movement])
        else:
            break

    if len(data_list)>0:
        # sort by stats
        stats_list_rank = np.argsort(np.array([-stats_list[i][-1] for i in range(len(stats_list))]))
        data_list = [data_list[i] for i in stats_list_rank]

        ### clean op data by interpolating using then with savgol filter
        imputer = KNNImputer(n_neighbors=fps // 11, weights="uniform") ## n neighbours is basically = 2
        for i in range(len(data_list)):
            for j in range(data_list[i][2].shape[1]):
                for k in range(data_list[i][2].shape[2]):
                    if j!=2:
                        ### make all score 0s NA
                        mask = [1 if not (data_list[i][2][:,2,k,:]==0)[l] else np.nan for l in range(data_list[i][2].shape[0])]
                        if (1 in mask): ## if not all NA
                            data_list[i][2][:,j,k,:] = imputer.fit_transform(data_list[i][2][:,j,k,:]*np.array(mask).reshape(-1,1))

                    data_list[i][2][:,j,k,0] = savgol_filter(data_list[i][2][:,j,k,0],
                                                                     window_length=13,
                                                                     polyorder=2,
                                                                     mode='mirror')

        ### stack data. Data list contained ranked sequences of people. Fit people into the top non-zero slot until max signers
        ### is reached
        fnos = np.arange(s[0],s[1]+1)
        pnos = -np.ones((len(fnos),max_number_signers))

        data_numpy = np.zeros((len(fnos),
                               data_list[0][2].shape[1],
                               data_list[0][2].shape[2],
                               max_number_signers))

        for i in range(len(data_list)):
            start_frame = int(data_list[i][0][0] - fnos[0])
            end_frame = int(data_list[i][0][0] - fnos[0] + data_list[i][2].shape[0])
            for j in range(max_number_signers):
                if np.max(data_numpy[start_frame:end_frame,:,:,j])==0: ### if all zeros then add, else go on to next signer slot
                    data_numpy[start_frame:end_frame, :, :, j] = data_list[i][2][:,:,:,0]
                    pnos[start_frame:end_frame, j] = data_list[i][1]
                    break

        #### trip upper and lower bounds of kept scene
        pnos_trim = [i for i in range(0, pnos.shape[0]) if np.max(pnos, axis=1)[i] != -1]
        pnos_bounds = np.arange(min(pnos_trim), max(pnos_trim) + 1)

        fnos = fnos[pnos_bounds]
        pnos = pnos[pnos_bounds]
        data_numpy = data_numpy[min(pnos_trim):(max(pnos_trim) + 1), :, :, :]

        ### convert to 25 fps at end
        if (fps>=28 and fps <=32 and convert_to_25fps==True):
            for i in range(len(data_list)):
                conv_data_list_red = np.zeros((len([1 for n in range(data_numpy.shape[0]-1) if n % 6 != 5]),
                                               data_numpy.shape[1],
                                               data_numpy.shape[2],
                                               data_numpy.shape[3]))
                for j in range(3):
                    for k in range(127):
                        if (j!=2):
                            conv_data_list_red[:, j, k, 0] = convert_30_25_fps(data_numpy[:, j, k, 0],
                                                                                    scores = data_numpy[:, 2, k, 0],
                                                                                    labels = False)
                        else:
                            conv_data_list_red[:, j, k, 0] = convert_30_25_fps(data_numpy[:, j, k, 0],
                                                                                    scores = None,
                                                                                    labels = False)
                data_numpy = conv_data_list_red

        ## data numpy has shape time*3=(x,y,confidence)*127 keypoints*number signers
        final_data = [fnos, pnos, data_numpy]

        ### save using first frame number, first person numbr and length of sequence in # frames
        print('saving '+output_folder + str(fnos[0]) + '_' + str(
                data_numpy.shape[0]) + '_data.pkl')
        pickle.dump(final_data, open(
            output_folder + str(fnos[0]) + '_' + str(
                data_numpy.shape[0]) + '_data.pkl', 'wb'))

