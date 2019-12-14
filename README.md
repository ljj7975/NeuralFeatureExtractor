# NeuralFeatureExtractor

## Supported
- [pytorch MNIST](https://github.com/pytorch/examples/tree/master/mnist)

## Example

The example extrat features of size [196] from [pytorch MNIST example](https://github.com/pytorch/examples/tree/master/mnist)

Trained model is stored under [pretrained_model](https://github.com/ljj7975/NeuralFeatureExtractor/tree/master/pretrained_model)

```
python main.py --model MnistModel --data_loader MnistDataLoader --trained_model pretrained_model/mnist_cnn.pt --data_folder <path to dataset>
```

Generated samples are stored under `generated/<model name>`
