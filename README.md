# NeuralFeatureExtractor

## Usgae

Generating sample MNIST features (feature size : [196])

```
// train a base model
python training_script/train_mnist.py --data_folder <path to dataset> --save-model

// generate samples
python main.py --model MnistModel --data_loader MnistDataLoader --trained_model mnist_cnn.pt --data_folder <path to dataset>
```

Samples will be stored under `generated/<model name>`