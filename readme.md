# Requirements
You can find all the requirements in the requirements.txt file

# Data preprocessing

To embed images and text in the CLIP embedding space you can use CLIP_data.ipynb, otherwise you can download the already preprocessed data at [link](https://epflch-my.sharepoint.com/:f:/r/personal/luca_salvador_epfl_ch/Documents/DeepLearning_Final?csf=1&web=1&e=d0y9gb).
These files should be put in a folder named data/hateful_memes.

If you want to run CLIP_data.ipynb make sure to put the non processed data that you can find at this [link](https://epflch-my.sharepoint.com/:f:/r/personal/luca_salvador_epfl_ch/Documents/DeepLearning_Final?csf=1&web=1&e=d0y9gb) under data_nonprocessed in the working directory.

# Training and inference
Our model are all trained on 20 epochs (the result sent are run on 5 epochs just to show the code runs). The training can be done in the main.ipynb. Otherwise the model weights are saved and can be found [here](https://epflch-my.sharepoint.com/:f:/r/personal/luca_salvador_epfl_ch/Documents/DeepLearning_Final?csf=1&web=1&e=d0y9gb) and can be used for inference. These weights should be put in a folder named model/.

# ACGAN
The ACGAN.ipynb was our inital choice but we have seen it not working very well. The code is auxiliary but not the main focus of the project
