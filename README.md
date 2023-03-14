# SfM and NeRF

## Phase 1: SfM
The code for SfM is in folder `Phase1`. The intermediate output images can be found in `wrapper.ipynb`. 

### How to run

1. Download data [here](https://drive.google.com/file/d/1DLdCpX5ojtSN4RjYZ2UwpV2fAJn3sX_k/view)
2. Navigate to Phase1 folder
3. Choice 1: follow the steps in `wrapper.ipynb`  
Choice 2: 
```
python Wrapper.py
```

## Phase 2: NeRF

### How to run
1. Download lego dataset [here](https://drive.google.com/drive/folders/1lrDkQanWtTznf48FCaW5lX9ToRdNDF1a)
2. Navigate to Phase2 folder
3. To train model 
```
python Wrapper.py --mode train
```
To test model
```
python Wrapper.py --mode test --checkpoint_path $CHECKPOINT_PATH$
```