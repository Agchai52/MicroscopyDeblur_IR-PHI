# Deep Image Restoration for Infrared Photothermal Heterodyne Imaging (IR-PHI)
Abstract: TODO

## Prerequisites
- Python 3.7
- Pytorch, torch>=0.4.1, torchvision>=0.2.1
- To run the code, please install required packages by the following command
```
pip install -r requirements.txt
```
## Build virtual environment using Anaconda (Optional)
- Install Anaconda [link](https://conda.io/projects/conda/en/latest/user-guide/install/windows.html)
- Create a new conda virtualenv called env3 with Python 3.7:
```
conda create --name env3 python=3.7
```
- Install required packages:
```
conda install --name env3 -c conda-forge --file requirements.txt
```
- Activate environment:
```
conda activate env3
```
- Deactivate environment:
```
conda deactivate
```
## Process real images
- Activate environment:
```
conda activate env3
```
- Put images in a folder, e.g. 'dataset/real_images/'
- Choose output folder, e.g. 'output/'
- Run model on CPUs:
```
python main.py --phase test_real --input_dir dataset/real_images/ --out_dir output/ --gpu -1
```
- Run model on a GPU, e.g. with gpu_id=0:
```
python main.py --phase test_real --input_dir dataset/real_images/ --out_dir output/ --gpu 0
```


## Train the model
### Generate the dataset
1. Generate train and test dataset
```
python generate_dataset.py --phase train --num_imgs 2000
python generate_dataset.py --phase test --num_imgs 500
```
2. Rename the dataset as "dataset" (Optional)
3. To generate names of all the train and test data, run the file "readDatasetNames.py" (Optional)
```
python readDatasetNames.py
```
```
python main.py --phase train --epoch 10 --gpu 0
```

### Test the model
```
python main.py --phase test --gpu 0
```