# README for Tracking

## Declaration and acknoledgement
We follow the classic tracking pipeline "tracking by detection".

The code is based on [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)  and our former work [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X). Sincere appreciation for their great contributions.


## Environment setup

To set up the codebase environment, do the following steps:

1. Create conda environment (python == 3.8)

    ```bash
    conda create -n rcoopertrk python=3.8
    conda activate rcoopertrk
    ```

2. Install dependencies

    ```bash
    pip install -r code/ab3dmot_plugin/requirements.txt
    ```

## Dataset setup
To setup the dataset, convert the RCopper to DAIR-V2X format using the provided dataset converter.

Setup the dataset path in codes/dataset_convertor/converter_config.py, and complete the conversion.

```bash
cd codes/dataset_converter
python rcooper2dair.py
```

## Tracking and evaluation

Modify the dataset path and the name in codes/scripts/tracking.sh (First several lines).

Then, run the following scripts.

```bash
cd codes
bash scripts/tracking.sh
```

