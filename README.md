# Speaker Emotion Recognition

### Single-stream Speaker Emotion Recognition
Our aim is to learn a discriminative embedding that is amenable to an emotion recognition task. Following Fig. shows the proposed emotion recognition framework leveraging \textit{pretrain-then-transfer} learning paradigm. First, it trains a speaker recognition network on a large-scale dataset with an aim to obtain a robust feature extractor. Afterwards, voice embeddings are extracted using this pre-trained network to train a simple two-layer multilayer perceptron with a softmax to classify speaker emotions.

![Emotion recognition network](images/voice.jpg)

### Pre-trained Audio Network
The network leverages a ‘thin-ResNet’ trunk architecture, end-to-end trainable on speaker recognition task. To pre-train this network, we have utilized large scale speaker recognition datasets, VoxCeleb 1 and 2, which help in generalization on downstream speaker emotion recognition task.

The code we used is released by authors and is publicly available [here](https://github.com/WeidiXie/VGG-Speaker-Recognition)![GitHub stars](https://img.shields.io/github/stars/WeidiXie/VGG-Speaker-Recognition.svg?logo=github&label=Stars)

### Training

```
python main.py --train_file 'voxFeats/train.pkl' --test_file 'voxFeats/test.pkl' --batch_size 35 --input_emb_size 512 -epochs 50
```
