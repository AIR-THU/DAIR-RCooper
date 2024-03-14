# Dataset conversion

To facilitate the evaluation of cooperative perception methods on RCooper. We provide the format converter from RCooper to other popular public cooperative perception datasets. After the conversion, researchers can directly evaluate the methods using several opensourced frameworks.

We now support the following conversions:

* [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
* [V2V4Real](https://github.com/ucla-mobility/V2V4Real)
* [OPV2V](https://github.com/ucla-mobility/OpenCOOD)


### RCooper 2 V2V4Real

Setup the dataset path in codes/dataset_convertor/converter_config.py, and complete the conversion.
```bash
cd codes/dataset_converter
python rcooper2vvreal.py
```

### RCooper 2 OPV2V

Setup the dataset path in codes/dataset_convertor/converter_config.py, and complete the conversion.
```bash
cd codes/dataset_converter
python rcooper2opv2v.py
```

### RCooper 2 DAIR-V2X

Setup the dataset path in codes/dataset_convertor/converter_config.py, and complete the conversion.
```bash
cd codes/dataset_converter
python rcooper2dair.py
```