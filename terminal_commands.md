# Terminal commands to run to train a DLC network
```python
# conda install -c conda-forge cudnn (just once)

ipython

import deeplabcut

video_folder = 'D:\\Dropbox (UCL)\\DAQ\\upstairs_rig\\videos_for_DLC_training\\'

deeplabcut.create_new_project('opto', 'philip', [video_folder + video_name for video_name in ['cam0.avi', 'cam1.avi', 'cam2.avi', 'cam3.avi', 'cam4.avi', 'cam5.avi', 'cam6.avi'], working_directory='D:\\data\\DLC_nets', copy_videos=True, multianimal=False)

config_path = 'D:\\data\\DLC_nets\\opto-philip-2021-07-26\\config.yaml'
# -> modify the config file as necessary, adding the desired bodyparts, and resnet: resnet_101 and default_net_type: resnet_101

deeplabcut.extract_frames(config_path, mode='automatic', algo='uniform', userfeedback=False, crop=False)

# deeplabcut.extract_frames(config_path, 'manual') # to choose frames manually

deeplabcut.label_frames(config_path)

deeplabcut.check_labels(config_path, visualizeindividuals=False)

deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')
# -> modify the pose_cfg.yaml file in the /train folder as necessary, changing the pretrained network location for example

deeplabcut.train_network(config_path, gputouse=0, keepdeconvweights=false)
# to verify that this is using the GPU, ensure that GPU-Util increases after vs. before running this command, by navigating to C:\Program File\NVIDIA Corporation\NVSMI in the command prompy and typing nvidia-smi   

deeplabcut.evaluate_network(config_path,Shuffles=[1], plotting=True)

# -> move onto the analysis framework to verify that the tracking looks good for your purposes. If not, extract (outlier) frames from the escape clips, refine the labels, and run a new training iteration.
```