# DL-Final-Project
A group repository to store the code and documents for CS 7643's Final Project


 - ./proposal has our Project Proposal
 - ./initial-test-code has a python script for a simple inference with a LLM loaded with the transformers library, and code to run it on the Georgia Tech GPUs (PACE ICE computing cluster)
 - ./environments has some yml conda envs to help you setup, however it is probably easier to just setup a clean PyTorch environment (ideally with GPU support w/ CUDA). Then to install transformers.

## Experiments

The experiments are in two python notebooks. As mentioned above, you need a Python PyTorch environment, with enough VRAM and RAM depending on the batch sizes etc. You need to install HuggingFace's transformers library [https://huggingface.co/docs/transformers/en/installation] and that should suffice. 

The notebook for finetuning is annotated and self-explanatory, and is used to fine-tune SmolLM2 to reach almost 90% accuracy on IMDb binary sentiment analysis.

The other notebook is for zero-shot / few-shot SmolLM2 benchmarking.