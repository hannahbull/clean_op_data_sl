# Clean OpenPose 2D keypoints for Sign Language Analysis

This code enables cleaning of OpenPose 2D keypoints for analysis of sign language videos. 

The key cleaning operations are: 

- Keeping only the upper body coordinates (hips, waist and above)
- Rescaling $x$ and $y$ coordinates to between 0 and 1
- Tracking of people across frames based of a maximum movement threshold `tracking_threshold`
- Segmentation of the video into scenes (breaks in tracking)
- Keeping only scenes of minimal length `min_length_segment` and containing likely signers (people with hand size above a threshold `hand_size_threshold` and wrist movement of the dominant hand above a threshold `hand_motion_threshold`)
- Keeping the top `max_number_signers` signers in each scene, where the most likely signers are ranked by hand size times variation of wrist movement of the dominant hand
- KNN imputation of missing data (where the OpenPose score=0)
- Smoothing with a Savitzky-Golay filter
- Conversion from 30 frames per second to 25 frames per second (optional)

# Input

Run OpenPose 2D with face, hands and body-25 keypoints for a video containing sign language. Save the *keypoints.json files in a folder. This is the argument `openpose_folder`.

# Output

Frame numbers, person numbers and numpy array of the skeleton keypoints for each scene in the video. The numpy array is of dimension ($t$, 3, 127, `max_number_signers`), where $t$ is the length of the scene in number of frames. The second axis corresponds to the $x$, $y$ and scores. The third axis is the number of keypoints of the head, hands and upper body. These are saved in `output_folder`.

# Example 

```python clean_op_data.py --config 'config.yaml'```

# References

OpenPose: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)


