# Ubiquitination Sites prediction
This is the official repository of "**A Benchmark for Machine Learning based Ubiquitination Sites Prediction from Human Protein Sequences**" paper.

## To Do
- [x] Add end to end training codes.
- [x] Add hybrid training codes.
- [x] Add preprocessing codes.
- [x] Add demo code.
- [x] Add end to end datasets.
- [ ] Add hybrid datasets.
- [ ] Add how to use 
- [x] Update readme to show how to use the demo codes.
- [ ] Support to get raw sequence as the input of demo codes. 

## Requirements

```
python 3.10
pytorch 2.1.0+cuda118
 ```

## Install
For testing the project install the corresponding `requirements.txt` files in 
your environment. 

If you want to use python environment:

1. Create a python environment: `python3 -m venv <env_name>`.
2. Activate the environment you have just created: `source <env_name>/bin/activate`.
3. install dependencies inside it: `pip3 install -r requirements.txt`.

## Demo
To do inference you have to prepare your windowed dataset and change the end_to_end_config.yaml. 
Then cd to the demo directory, `cd ./demo` and run the following command:

`python inference_end_to_end.py`

The result will be saved in the **save_path** directory of the yaml config file.

you have to convert your sequences to **fixed size** sequences, i.e., window size, similar to the provided file in
`data/test data/processed/window/55.csv`.


