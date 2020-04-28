import json
import numpy as np
from tqdm import tqdm

def op_clean_data(op_data, person_number, width, height):
    body25points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 17, 18]

    op_data_rs = np.array(op_data['people'][person_number]['face_keypoints_2d']).reshape((70, 3))
    op_data_rs = np.row_stack((op_data_rs, np.array(op_data['people'][person_number]['pose_keypoints_2d']).reshape((25, 3))[body25points,:]))
    op_data_rs = np.row_stack((op_data_rs, np.array(op_data['people'][person_number]['hand_left_keypoints_2d']).reshape((21, 3))))
    op_data_rs = np.row_stack((op_data_rs, np.array(op_data['people'][person_number]['hand_right_keypoints_2d']).reshape((21, 3))))

    op_data_rs[:, 0] = op_data_rs[:, 0] / int(width)
    op_data_rs[:, 1] = op_data_rs[:, 1] / int(height)

    return np.around(op_data_rs, decimals=3)

def tracking_openpose(op_data_1, width, height, fps, data_location, list_op_files, starting_file_number, person_number, length_of_segment, tolerance=1, forwards=True):
    if forwards == True:
        next = 1
    else:
        next = -1

    ### create dict
    data_dict = {}
    data_dict["data"] = []

    for it in range(1,length_of_segment+1):

        op_data_1_rs = op_clean_data(op_data_1, person_number, width, height)

        ### create json
        pose_dict = {}
        pose_dict["skeleton"] = []
        pose_dict["skeleton"].append({
            "pose": list(op_data_1_rs[:,0:2].flatten()),
            "score": list(op_data_1_rs[:,2]),
            "person_number": person_number
        })
        data_dict["data"].append({
            "frame_index": it,
            "skeleton": pose_dict["skeleton"]
        })

        ### open next op file if not last iteration
        if it<length_of_segment:
            with open(data_location + list_op_files[starting_file_number + next*it]) as json_file:
                op_data_2 = json.load(json_file)

            number_of_people = len(op_data_2['people'])
            if number_of_people==0:
                return data_dict

            distance = []
            for nopeo in range(number_of_people):

                op_data_2_rs = op_clean_data(op_data_2, nopeo, width, height)

                ### compute difference between the two
                op_data_diff = np.column_stack((op_data_1_rs, op_data_2_rs))
                op_data_diff = op_data_diff[[70,71,78]]
                op_data_diff = op_data_diff[(op_data_diff!=0).any(axis=1)]

                if len(op_data_diff)==0:
                    return data_dict

                distance.append(np.median(np.linalg.norm((op_data_diff[:,0:2]-op_data_diff[:,3:5]), axis=1)))

            if (min(distance) < tolerance / fps):
                person_number = distance.index(min(distance))
                op_data_1 = op_data_2
            else:
                return data_dict

    return data_dict


def get_full_dict(list_op_files, data_location, width, height, fps, tolerance):
    print('Getting full dictionary')
    full_dict = {}
    for starting_file_number in tqdm(range(len(list_op_files) - 1)):
        ### open op file for initialisation
        with open(data_location + list_op_files[starting_file_number]) as json_file:
            op_data_1 = json.load(json_file)
            no_people_opdata = len(op_data_1['people'])
        if starting_file_number not in full_dict.keys():
            full_dict[starting_file_number] = {}
        if starting_file_number + 1 not in full_dict.keys():
            full_dict[starting_file_number + 1] = {}
        full_dict[starting_file_number]['number_of_people'] = no_people_opdata
        for person_number in range(no_people_opdata):
            data_dict = tracking_openpose(op_data_1, width, height, fps, data_location, list_op_files, starting_file_number,
                                          person_number, length_of_segment=2, tolerance=tolerance, forwards=True)

            if len(data_dict) >= 1:
                ## add current person if not already added
                if person_number not in list(full_dict[starting_file_number].keys())[1:]:
                    full_dict[starting_file_number][person_number] = {}
                    full_dict[starting_file_number][person_number]['pose'] = data_dict['data'][0]['skeleton'][0][
                        'pose']
                    full_dict[starting_file_number][person_number]['score'] = data_dict['data'][0]['skeleton'][0][
                        'score']
                    full_dict[starting_file_number][person_number]['next_person'] = np.nan

                if len(data_dict['data']) > 1:  ### add next person
                    next_person_number = data_dict['data'][1]['skeleton'][0]['person_number']
                    full_dict[starting_file_number + 1][next_person_number] = {}
                    full_dict[starting_file_number + 1][next_person_number]['pose'] = data_dict['data'][1]['skeleton'][0]['pose']
                    full_dict[starting_file_number + 1][next_person_number]['score'] = data_dict['data'][1]['skeleton'][0]['score']
                    full_dict[starting_file_number + 1][next_person_number]['next_person'] = np.nan

                    full_dict[starting_file_number][person_number]['next_person'] = next_person_number

    return full_dict


def compute_size_movement(comb_data_numpy):
    ### size Time * 3 * 127 * 1
    comb_data_numpy = np.moveaxis(comb_data_numpy, [0, 1, 2, 3], [1, 0, 2, 3])
    ### size 3 * T * 127 * 1
    #### measure size and variation in hand movement
    ### make dataframe of hands
    lh_y = comb_data_numpy[1, :, 85:106, 0]
    lh_score = comb_data_numpy[2, :, 85:106, 0]
    rh_y = comb_data_numpy[1, :, 106:127, 0]
    rh_score = comb_data_numpy[2, :, 106:127, 0]

    max_lh_y = []
    min_lh_y = []
    max_rh_y = []
    min_rh_y = []
    wrist_lh_y = []
    wrist_rh_y = []
    for t in range(lh_y.shape[0]):
        if len(lh_y[t][lh_score[t] > 0.1]) > 0:
            max_lh_y.append(np.max(lh_y[t][lh_score[t] > 0.1]))
            min_lh_y.append(np.min(lh_y[t][lh_score[t] > 0.1]))
        if len(rh_y[t][rh_score[t] > 0.1]) > 0:
            max_rh_y.append(np.max(rh_y[t][rh_score[t] > 0.1]))
            min_rh_y.append(np.min(rh_y[t][rh_score[t] > 0.1]))
        if lh_score[t][0] > 0.05:
            wrist_lh_y.append(lh_y[t][0])
        if rh_score[t][0] > 0.05:
            wrist_rh_y.append(rh_y[t][0])

    #### height of hands
    lh_height = 0
    rh_height = 0
    lh_movement = 0
    rh_movement = 0
    if len(max_lh_y) > 0:
        lh_height = np.median(np.array(max_lh_y) - np.array(min_lh_y))
    if len(max_rh_y) > 0:
        rh_height = np.median(np.array(max_rh_y) - np.array(min_rh_y))
    if len(wrist_lh_y) > 0:
        lh_movement = np.var(np.array(wrist_lh_y))
    if len(wrist_rh_y) > 0:
        rh_movement = np.var(np.array(wrist_rh_y))

    height = max(lh_height, rh_height)
    movement = max(lh_movement, rh_movement)
    return height, movement


def convert_30_25_fps(vector, scores = None, labels = True):
    if scores is None:
        vector = [vector[0]]+[(5-i%6)/5*vector[i]+(i%6)/5*vector[i+1] for i in range(1,len(vector)-1)]
    else:
        vector = [vector[0]]+[(5-i%6)/5*vector[i]*int(scores[i]!=0)
                              +int(scores[i]!=0)*int(scores[i+1]==0)*(i%6)/5*vector[i]
                                +int(scores[i]==0)*int(scores[i+1]!=0)*(5-i%6)/5*vector[i+1]
                              +(i%6)/5*vector[i+1]*int(scores[i+1]!=0) for i in range(1,len(vector)-1)]
    vector = [vector[i] for i in range(len(vector)) if i % 6 != 5]
    if labels==True:
        vector = [int(round(vector[i])) for i in range(len(vector))]
    return np.array(vector)