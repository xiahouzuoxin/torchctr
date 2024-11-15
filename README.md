# About

I primarily used TensorFlow for large-scale recommendation tasks when in big company, but PyTorch could be more efficient for smaller tasks in a smaller company.

This directory aims to train a Click-Through Rate (CTR) model using PyTorch. It's a simple example, seeking to keep everything minimal. 

> **Tips**:
> 1. Apply polars instead of pandas to process data. I first implemented the pandas version of FeatureTransformer, but it's very slow when datasize> 20 millon.
> 2. Apply parquets instead of pickle files to save data samples. It's really saved my memory.

## Supported features:

* Both numerical and categorical input features
  * Categorical: automatic vocabulary extraction, low-frequency filtering, dynamic embedding, hash embedding
  * Numerical: standard or 0-1 normalization, automatic discretization, automatic update of statistical number for standard or 0-1 normalization if new data is fed in
* Variable-length sequence feature support, if there's order in the sequence, please put the latest data before the oldest data as it may pads at the end of the sequence
* Sequence features support weights by setting the weight column
* Implemented a common [Trainer](./torchctr/trainer.py) for training pytorch models, and save/load the results
* Basic FastAPI for [Model API Serving](./torchctr/serving/serve.py)

Not supported:

- Distribution training, as target of this tool is for small companies. And I used polars to speed up the data processing.

# Install

```
pip install git+https://github.com/xiahouzuoxin/torchctr
```

And follow the [example](./examples/train_amazon_hf_dataset.ipynb) to use.

# [Model API Serving](./torchctr/serving/serve.py)

An simple serving without latency considered.

1. [Optional] According to your model and data processing, maybe need create a new ServingModel like [BaseServingModel](./torchctr/serving/model_def.py)
2. Set up the service:
    - Debuging: Given service name and model path from command line
      ```
      cd $torchctr_root
      python -m torchctr.serving.serve --name [name] --path [path/to/model or path/to/ckpt] --serving_class BaseServingModel
      ```
    - Production: write the command line parameters to `serving_models` variable in [torchctr/serving/serve.py](torchctr/serving/serve.py)

3. Test the service: reference `test_predict` in [example](./examples/train_amazon_hf_dataset.ipynb)
