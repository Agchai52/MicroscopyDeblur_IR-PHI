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
- Download code shared in this directory
```
git clone https://github.com/Agchai52/MicroscopyDeblur_IR-PHI.git
```
- Download trained weights and unzip it in the current directory [link](https://drive.google.com/file/d/1uqcObC60L4aSXaadLPJQzuD9IHZe98Ny/view?usp=sharing)

- Activate environment (for Anaconda user):
```
conda activate env3
```
- Activate environment (for Virtualenvwrapper user):
```
workon env3
```

- Put `.png` or `.jpg` images in a folder, e.g. `dataset/real_images/`
- Choose output folder, e.g. `output/`
- Run the model on CPUs:
```
python main.py --phase test_real --input_dir dataset/real_images/ --output_dir output/ --gpu -1
```
- Run the model on a GPU, e.g. with `gpu_id=0`:
```
python main.py --phase test_real --input_dir dataset/real_images/ --output_dir output/ --gpu 0
```


## Train and Test the model
### Generate the dataset
1. Generate train and test dataset
```
python generate_dataset.py --phase train --num_imgs 2000
python generate_dataset.py --phase test --num_imgs 500
```
2. Rename the dataset as "dataset" (Optional)

### Train the model
- Remove old files and trained weights
```commandline
rm -r checkpoint/ test/ valid/ logfile* plot_* *_record.txt
```
- Train a new model
```
python main.py --phase train --epoch 10 --gpu 0
```

### Test the model
```
python main.py --phase test --gpu 0
```