# README for Intersection_Late

## Environment setup
> The codebase is build upon [V2V4Real](https://github.com/ucla-mobility/V2V4Real). Code is modified for supporting the following tracking task.

To set up the codebase environment, do the following steps:

1. Create conda environment (python >= 3.7)

    ```bash
    conda create -n v2v4real python=3.7
    conda activate v2v4real
    ```

2. Pytorch Installation (>= 1.12.0 Required)

    ```bash
    conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
    ```

3. spconv 2.x Installation

    ```bash
    pip install spconv-cu113
    ```

4. Install other dependencies

    ```bash
    cd codes/v2v4real_plugin
    pip install -r requirements.txt
    python setup.py develop
    ```

5. Install bbx nms calculation cuda version

    ```bash
    cd codes/v2v4real_plugin
    python opencood/utils/setup.py build_ext --inplace
    ```

## Dataset setup
To setup the dataset, convert the RCopper to V2V4Real format using the provided dataset converter.

Setup the dataset path in codes/dataset_convertor/converter_config.py, and complete the conversion.
```bash
cd codes/dataset_converter
python rcooper2vvreal.py
```

## Inference
We provide the inference script in codes/scripts/det_inference_intersection_late.sh.

To evaluate the pretrained model, do the following steps:

1. Download the pretrained checkpoint and make sure the path parameters in .sh file are all right.

2. Run the command

    ```bash
    cd codes
    bash scripts/det_inference_intersection_late.sh
    ```

## Training
We provide the training script in codes/scripts/det_train_intersection_late.sh

To train a new model, do the following steps:

1. Setup the model parameters in a YAML file. We provide an example in configs/intersection_late.yaml

2. Make sure the 'MODEL_CONFIG_DIR' in codes/scripts/det_train_intersection_late.sh is right, and run the command

    ```bash
    cd codes
    bash scripts/det_train_intersection_late.sh
    ```

The training logs are saved in codes/v2v4real_plugin/opencood/logs.


## Acknowledgment
The codebase is build upon [V2V4Real](https://github.com/ucla-mobility/V2V4Real). Thanks for their great contributions.
