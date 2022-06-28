# Zero Shot Learning w/ CLIP

---
## Table of contents
- [1. Introduction](#introduction)
- [2. Repo structure](#repo-structure)
- [3. Implementation details](#implementation-details)
- [4. Monitoring integration](#monitoring-integration)
- [5. Quickstart code](#quickstart-code)
- [6. License](#license)
---

## Introduction
Self-supervised methods allow deep learning architectures to inheritate natural patterns of unstructured data. This way, we leverage the study of particular downstream tasks, which require heavy amounts of data (a strong limitation), to the analysis of more *intelligent* generalist models. As mentioned in [this Meta article](https://ai.facebook.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/):

> Self-supervised learning enables AI systems to learn from orders of magnitude more data, which is important to recognize and understand patterns of more subtle, less common representations of the world. Self-supervised learning has long had great success in advancing the field of natural language processing (NLP), including the 
Collobert-Weston 2008 model, Word2Vec, GloVE, fastText and, more recently, BERT, RoBERTa, XLM-R and others. Systems pretrained this way yield considerably higher performance than when solely trained in a supervised manner.

In this repository, we introduce an implementation of [CLIP model](https://openai.com/blog/clip/), together with some useful tips for model training. As one can read from the paper's abstract,

> State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks.

For pedagogical purposes, our training scripts will be limited to the usage of [Common Objects in COntext (COCO) dataset](https://cocodataset.org/#home). 


## Repo structure

Since all scripts do have the same requirements and source folder, a unified structure for all of them is used, as illustrated below:

<details>
<summary>
Click here to find out!
</summary>

    ├── input                        # Configuration files, datasets,...
    │   ├── model_config.json        # Configuration file for model architecture
    │   ├── training_config.json     # Configuration file for training (batch_size, learning_rate,...)
    │   └── wandb_config.json        # Credentials for Weights and Biases API usage
    │
    ├── scripts                      # Building blocks of the repo
    │   ├── pretrain.py              # Pretraining, including both freeze and unfreeze backbones
    │   └── inference_demo.py        # StreamLit application built with OpenAI pretrained CLIP model (WIP)
    │
    ├── src                          # Main methods to build scripts code
    │   ├── callbacks.py             # Contains W&B logging
    │   ├── dataset.py               # Method that structures and transforms data
    │   ├── fitter.py                # Training, validation and storing loop wrapper
    │   ├── loss.py                  # Custom function to meet our needs during training
    │   ├── model.py                 # Core script containing the architecture of the model
    │   ├── setup.py                 # Helper methods to shorten main script length and make it more readable
    │   └── utils.py                 # Helper methods to control reproducibility
    │
    └── requirements.txt             # Libraries to be used and their versions
</details>



## Implementation details

CLIP learns visual patterns from text supervision. Taking into account model architecture and dataset choice, the next considerations were followed:

* As COCO dataset provides five text captions per image, a random caption is chosen to facilitate model generalisation. Plus, images are provided with a strong data augmentation pipeline to the same end.
* Both vision and text pretrained backbones from [HuggingFace :hugs:](https://huggingface.co/) are used as feature extractors, followed by dense blocks to standardise shape. To combine both components efficiently, a first stage of pretraining with frozen feature extractors to solely train dense blocks is carried out and configured via the file `freeze_training_config.json`. A full training of the network is performed afterwards, and can be configured via the file `training_config.json`.

A visual description of the implementation is shown now:

![CLIPImage](https://github.com/openai/CLIP/blob/main/CLIP.png)


## Monitoring integration
This experiment has been integrated with Weights and Biases to track all metrics, hyperparameters, callbacks and GPU performance. You only need to adapt the parameters in the `wandb_config.json` configuration file to keep track of the model training and evaluation. An example is shown [here](https://wandb.ai/azm630/ZSL_CLIP).

## Quickstart code
You can reproduce this experiment by running the following code snippet:

```console
// Clone repo
git clone https://github.com/hedrergudene/ZSL_CLIP.git
// Change working directory
cd ZeroShotLearning_CLIP/
// Set up COCO dataset
mkdir input/MSCOCO
wget http://images.cocodataset.org/zips/train2017.zip -P input/MSCOCO \
     && mkdir input/MSCOCO/images \
     && unzip input/MSCOCO/train2017.zip -d input/MSCOCO/images \
     && rm input/MSCOCO/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip -P input/MSCOCO/MSCOCO \
     && unzip input/MSCOCO/val2017.zip -d input/MSCOCO/images \
     && rm input/MSCOCOval2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P input/MSCOCO \
     && unzip input/MSCOCO/annotations_trainval2017.zip -d input/MSCOCO \
     && rm input/MSCOCO/annotations_trainval2017.zip
// Install requirements
pip install -r requirements.txt
// Run pretraining
mv scripts/pretrain.py pretrain.py
python scripts/pretrain.py
```

Make sure you have previously updated `wandb_config.json` file with your own credentials.

## License
Released under [MIT](/LICENSE) by [@hedrergudene](https://github.com/hedrergudene).
