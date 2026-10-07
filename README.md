# Ubiquitination Sites Prediction
This is the official repository of "**A Benchmark for Machine Learning based Ubiquitination Sites Prediction from Human Protein Sequences**" paper.

## To Do
- [x] Add end to end training codes.
- [x] Add hybrid training codes.
- [x] Add preprocessing codes.
- [x] Add end to end demo code.
- [ ] Add hybrid demo code.
- [x] Add end to end datasets.
- [ ] Add hybrid datasets.
- [x] Update readme to show how to use the demo codes.
- [ ] Support to get raw sequences as the input of demo codes. 
- [ ] Add benchmark dataset and its description.

## Requirements

```
Python 3.10-3.12
PyTorch 2.13.0 / torchvision 0.28.0
 ```

## Install
For testing the project install the corresponding `requirements.txt` files in 
your environment. 

If you want to use python environment:

1. Create a python environment: `python3 -m venv <env_name>`.
2. Activate the environment you have just created: `source <env_name>/bin/activate`.
3. Install a matching official PyTorch build, then the demo dependencies from the repository root:

```sh
# CPU environment for offline compatibility checks:
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r demo/requirements.txt
python -m unittest discover -s tests -v
```

For GPU use, select an [official PyTorch 2.13.0 CUDA wheel channel](https://pytorch.org/get-started/previous-versions/)
such as `cu126` instead of `cpu`, with a compatible NVIDIA driver. The old CUDA 11.8
pins contain known vulnerabilities and must not be restored. Use a fresh environment
when moving from the original stack. The dependency file covers the demo; transformer
training scripts need their additional libraries separately.

The LSTM architecture and state-dict format are unchanged. Checkpoints are loaded
with `weights_only=True` onto CPU before copying into the model. The small test uses
synthetic tensors and a local in-memory checkpoint; it does not train or download
models or datasets. GPU performance and full pretrained-model evaluation have not
been validated for this dependency upgrade.

## Demo
To do inference you have to prepare your windowed dataset and change the end_to_end_config.yaml. 
Then cd to the demo directory, `cd ./demo` and run the following command:

`python inference_end_to_end.py`

The result will be saved in the **save_path** directory of the yaml config file.

you have to convert your sequences to **fixed size** sequences, i.e., window size, similar to the provided file in
`data/test data/processed/window/55.csv`.


