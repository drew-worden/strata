<br/>
<p align="center">
      <img src="/assets/logo.svg" height="156">
</p>
<br/>
A custom-built large language model, built to outperform GPT-3 level models while being significantly smaller and easier to train.
<p align="center">
    <img src="https://img.shields.io/badge/AUTHOR-Drew_Worden-333333.svg?style=for-the-badge&labelColor=000">
    <img src="https://img.shields.io/badge/LICENSE-MIT-333333.svg?style=for-the-badge&labelColor=000">
</p>

# Description
Strata is a large language model intended for experimentation and academic purposes. It is not advisable to use it in production settings.
The main goal of Strata is to provide GPT-3 and above level performance while reducing the model size (parameters), compute requirements, size of the token training set, and training time. This is to make it more accessible to researchers and students who may not have access to the resources required to train a model like GPT-3.
The model is written in PyTorch and is designed to be easy to understand and modify. It is intended to be a starting point for those interested in experimenting with large language models. The model is explicitly created without the use of off-the-shelf libraries like Hugging Face's `transformers` module and OpenAI's `tiktoken` module to increase understanding of the underlying mechanisms.

# Features
- [x] Transformer architecture written using only PyTorch
- [ ] Custom BPE tokenizer supporting 86,400 unique tokens
- [ ] Configuration class for easy hyperparameter tuning
- [ ] Well-defined and customizable training loop
- [x] Logger for easy tracking of training progress
- [ ] Checkpointing for easy resuming of training
- [ ] Evaluation tests for comparing model performance
- [x] Data loader for easy loading of training data
- [ ] Optimizer utility for experimenting with different optimizer settings
- [ ] Transfer script for easy transfer of model to training environment
- [ ] Associated paper for more details on the model architecture and training process

# Associated Paper
The associated paper for Strata is currently in progress and will be linked here once it is complete.

[//]: # (Please refer to the associated paper for more details on the model architecture and the training process.)

# License
[MIT License](LICENSE)
