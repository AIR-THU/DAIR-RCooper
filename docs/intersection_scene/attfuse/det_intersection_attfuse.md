# README for Intersection_Attfuse

## Environment setup
> The codebase is built upon [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). We have modified certain portions of the code to better suit our task requirements.

To set up environment, you can follow the instructions [here](https://opencood.readthedocs.io/en/latest/md_files/installation.html) to install OpenCOOD. The code is tested on pytorch v1.8.0 and cudatoolkit v11.1.

<b>Notes</b>: Don't run "python setup.py develop" in original opencood. Turn to our /codes/opencood_plugin to install modified opencood.
```bash
cd codes/opencood_plugin
python setup.py develop
```

## Data Preparation
#### a.Download
Download RCopper dataset here.
#### b.Convert to OPV2V format using the provided dataset converter.
modify the dataset path in codes/dataset_convertor/converter_config.py, and run the following commond:
```bash
cd codes/dataset_converter
python rcooper2opv2v.py
```

## Inference
We provide the inference script in codes/scripts/det_inference_intersection_attfuse.sh.

To evaluate the pretrained model, do the following steps:

1. Download the pretrained checkpoint and make sure the path parameters in .sh file are all right.

2. Run the command

    ```bash
    cd codes
    bash scripts/det_inference_intersection_attfuse.sh
    ```

## Training
We provide the training script in codes/scripts/det_train_intersection_attfuse.sh

To train a new model, do the following steps:

1. Setup the model parameters in a YAML file. We provide an example in configs/intersection_attfuse.yaml

2. Make sure the 'MODEL_CONFIG_DIR' in codes/scripts/det_train_intersection_attfuse.sh is right, and run the command

    ```bash
    cd codes
    bash scripts/det_train_intersection_attfuse.sh
    ```

The training logs are saved in codes/opencood_plugin/opencood/logs.


## Acknowledgment
The codebase is build upon [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). Thanks for their great contributions.
