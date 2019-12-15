# NeuralFeatureExtractor

## Supported
- [pytorch MNIST](https://github.com/pytorch/examples/tree/master/mnist)

## Example

The example extrat features of size [196] from [pytorch MNIST example](https://github.com/pytorch/examples/tree/master/mnist)

Trained model is stored under [pretrained_model](https://github.com/ljj7975/NeuralFeatureExtractor/tree/master/pretrained_model)

```
python main.py --model_config config/mnist/model_config.json --data_config config/mnist/data_config.json --num_gpu 1
```

Generated samples are stored under `generated/<model name>`
