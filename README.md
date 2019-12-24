# NeuralFeatureExtractor

NeuralFeatureExtractor (NFE) enables feature extraction from trained PyTorch model


## Supported models & datasets

current version supports following models:
- [pytorch MNIST CNN example](https://github.com/pytorch/examples/tree/master/mnist)
- [Keyword Spotting: Honk](https://github.com/castorini/honk) & [Google Speech Command dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)
- [CNN example for MNIST](https://github.com/pytorch/examples/tree/master/mnist) & [Facial Expression Research Group Database (FERG-DB)](https://grail.cs.washington.edu/projects/deepexpr/ferg-db.html)

Details can be found [below](https://github.com/ljj7975/NeuralFeatureExtractor#supported-feature-extraction)


## Supported file types
- csv

To support new file type, simply write a class that inherits [FileHandler](https://github.com/ljj7975/NeuralFeatureExtractor/blob/master/file_handler/file_handler.py)


## How to extract features for new dataset using a new model

### Step 1: Link PyTorch model

In order to add a new model, copy and paste your model to [model](https://github.com/ljj7975/NeuralFeatureExtractor/tree/master/model) folder and modify the `forward` function to return the intermediate representation

The new model must be added to [\_\_init\_\_.py](https://github.com/ljj7975/NeuralFeatureExtractor/blob/master/model/__init__.py) as well.

Then, the trained model must be stored under [pretrained_model](https://github.com/ljj7975/NeuralFeatureExtractor/tree/master/pretrained_model) folder

### Step 2: Link DataLoader

Next, copy and paste your DataLoader to [data_loader](https://github.com/ljj7975/NeuralFeatureExtractor/tree/master/data_loader) folder

### Step 3: Generate config files

NFE requires two configurations
- `model_config` : configuration for the Model
- `data_config` : configuration for the DataLoader

Samples can be found from [config](https://github.com/ljj7975/NeuralFeatureExtractor/tree/master/config) folder

### Step 4: Extract features

The following command generates features and stored them at `generated/<model name>`
```
python main.py --model_config <model_config> --data_config <data_config>
```

## Supported feature extraction

### [MNIST](https://github.com/pytorch/examples/tree/master/mnist)

target classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
feature size : 196

```
python main.py --model_config config/mnist/model_config.json --data_config config/mnist/data_config.json
```


### [Keyword Spotting: Honk](https://github.com/castorini/honk) & [Google Speech Command dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)

Relevant packages can be installted by running `pip install -r per_model_requirements/kws_res_model.txt`

The dataset must be downloaded prior to feature extraction.
Please refer to [official Google Speech Command dataset page](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)

target classes: ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
feature size : 196

```
python main.py --model_config config/kws/model_config.json --data_config config/kws/data_config.json
```

### [CNN example for MNIST](https://github.com/pytorch/examples/tree/master/mnist) & [Facial Expression Research Group Database (FERG-DB)](https://grail.cs.washington.edu/projects/deepexpr/ferg-db.html)

Use the same CNN example for MNIST by loading FERG images with grey scale and reduce dimensions to [1, 28, 28]

target classes:

- anger: 0
- disgust: 1
- fear: 2
- joy: 3
- sadness: 4

feature size : 196

```
python main.py --model_config config/ferg/model_config.json --data_config config/ferg/data_config.json
```
