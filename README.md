# Purpuse

I primarily used TensorFlow for large-scale recommendation tasks when in big company, but PyTorch could be more efficient for smaller tasks in a smaller company.

This directory aims to train a Click-Through Rate (CTR) model using PyTorch. It's a simple example, seeking to keep everything minimal. While the model is straightforward, the data preprocessing pipeline is more complex due to a variety of inputs.

Supported features include:

* Both numerical and categorical input features
  * Categorical: automatic vocabulary extraction, low-frequency filtering, dynamic embedding, hash embedding
  * Numerical: standard or 0-1 normalization, automatic discretization, automatic update of statistical number for standard or 0-1 normalization if new data is fed in
* Variable-length sequence feature support, if there's order in the sequence, please put the latest data before the oldest data as it may pads at the end of the sequence
* Sequence features support weights by setting the weight column
* Implemented [DataFrameDataset](./torchctr/dataset.py) for straightforward training with input data of pandas/polars DataFrame format
* Implemented a common [Trainer](./torchctr/trainer.py) for training pytorch models, and save/load the results
* Basic FastAPI for [Model API Serving](./torchctr/serving/serve.py)

Not supported:

- Distribution training, as target of this tool is for small companies

# Install

```
pip install git+https://github.com/xiahouzuoxin/torchctr
```

# [Example](./examples/train_amazon.ipynb)

1. Using [DataFrameDataset](./torchctr/dataset.py) to load the raw data as pytorch Dataset format
2. Create a model definition file in `torchctr/models`, and implement the model by inherit from nn.Module but with some extra member methods,
    - required:
      - training_step
      - validation_step
    - optional:
      - configure_optimizers
      - configure_lr_scheduler
3. Using [Trainer](./torchctr/trainer.py) to train the model
4. Serving the model by [Model API Serving](./torchctr/serving/serve.py)

# [Model API Serving](./torchctr/serving/serve.py)

1. [Optional] According to your model and data processing, maybe need create a new ServingModel like [BaseServingModel](./torchctr/serving/model_def.py)
2. Set up the service:
    - Debuging: Given service name and model path from command line
      ```
      cd $torchctr_root
      python -m torchctr.serving.serve --name [name] --path [path/to/model or path/to/ckpt] --serving_class BaseServingModel
      ```
    - Production: write the command line parameters to `serving_models` variable in [torchctr/serving/serve.py](torchctr/serving/serve.py)

3. Test the service: reference `test_predict` in [example](./examples/train_amazon.ipynb)

# Related Dataset

- https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- https://tianchi.aliyun.com/dataset/56?spm=a2c22.12282016.0.0.27934197fn5Vdv
