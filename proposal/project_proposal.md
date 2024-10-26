# Team Name
Small NLP

# Project Title
Lightweight Sentiment Analysis: Small LLMs vs. Big Models and Traditional Techniques

# Project summary
```
(4-5+ sentences). Fill in your problem and background/motivation (why do you want to solve it? Why is it interesting?). This should provide some detail (don’t just say “I’ll be working on object detection”)
```




# What you will do 
``` 
(Approach, 4-5+ sentences) - Be specific about what you will implement and what existing code you will use. Describe what you actually plan to implement or the experiments you might try, etc. Again, provide sufficient information describing exactly what you’ll do. One of the key things to note is that just downloading code and running it on a dataset is not sufficient for a description or a project! Some thorough implementation, analysis, theory, etc. have to be done for the project.
```




# Resources / Related Work & Papers 
``` 
(4-5+ sentences). What is the state of art for this problem? Note that it is perfectly fine for this project to implement approaches that already exist. This part should show you’ve done some research about what approaches exist.
```

Link to some benchmarks for models on the dataset we are using: 
* https://paperswithcode.com/sota/sentiment-analysis-on-imdb




# Datasets 
IMDb Movie Reviews for Sentiment Analysis:
* https://ai.stanford.edu/~amaas/data/sentiment/

Other links to the dataset:
* https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews 
* https://paperswithcode.com/dataset/imdb-movie-reviews

# List your Group members.
Jaden Zwicker




# Example Proposal
[Ed Discussion Info Thread](https://edstem.org/us/courses/60909/discussion/5248321)


```
Team: Next Move

Project Title: Motion Prediction

Project Summary: 

The ability to forecast human motion is useful for a myriad of applications including robotics, self-driving cars, and animation. Typically we consider this a generative modeling task, where given a seed motion sequence, a network learns to generate/synthesize a sequence of plausible human poses. This task has seen much progress for shorter horizon forecasting through traditional sequence modeling techniques; however, longer horizons suffer from pose collapse. This project aims to explore recent approaches that can better capture long-term dependencies and generate longer horizon sequences.

Approach:

 Based on our preliminary research, there are multiple approaches to address the 3D Motion Prediction problem. We want to start by collecting and analyzing varying approaches; e.g. Encoder-Recurrent-Decoder (ERD), GCN, and Spatio-Temporal Transformer. We expect to reproduce [1] and baseline other approaches.

 As a stretch goal, we want to explore possible directions to improve these papers. One avenue is to augment the data to provide multiple views of the same motion and ensure prediction consistency.

 Another stretch goal is to come up with a new metric and loss terms (e.g. incorporating physical constraints) to improve benchmarks.

Resources/Related Work:

Older studies such as [2],[6] use LSTM based recurrent neural networks. However, it has been shown that Attention [5] offers an effective approach to dealing with long sequences. In particular, [1] implements a transformer model using attention for human prediction with an accuracy of XX. Alternative approaches such as [1],[4] explicitly model the spatial aspect of motion modeling. These approaches seem to work better than Attention when the data is limited. To benchmark our results, we will use [9] which is a dataset with 1000 samples of varying sequence lengths.

[1] “A Spatio-temporal Transformer for 3D HumanMotion Prediction”, Aksan et al.

[2] “Recurrent Network Models for Human Dynamics”, Fragkiadaki et al.

[3] “Learning Dynamic Relationships for 3D Human Motion Prediction”, Cui et al.

[4] “Convolutional Sequence to Sequence Model for Human Dynamics”, Zhang et al

[5] “Attention is all you need”, Vaswani et al.

[6] “On human motion prediction using recurrent neural networks”, Martinez et al.

[7] “Structured Prediction Helps 3D Human Motion Modelling”, Aksan et al.

[8] “Learning Trajectory Dependencies for Human Motion Prediction”, Mao et al.

[9] “AMASS: Archive of Motion Capture as Surface Shapes”, Mahmood et al.

Datasets:

AMASS https://amass.is.tue.mpg.de/

Team Members:

Eren Jaeger

Armin Arlert

Mikasa Ackerman
```