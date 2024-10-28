# Team Name
Small NLP

# Project Title
Lightweight Sentiment Analysis: Small LLMs vs. Big Models and Traditional Techniques

# Project summary
With the recent rise of extremely powerful LLMs which perform sentiment analysis at record high scores there is also a greater need for considerable amounts of compute to run such models. The deep learning models which excel at this NLP task usually require cloud compute or a local GPU cluster which users may have to outsource to. As such this project seeks to evaluate and compare the performance of a fine-tuned very small deep learning model which will be able to run on a lot more hardware then the leading LLMs such as BERT or GPT. 

The smaller model will be fine-tuned to IMDb’s movie review dataset to assess its performance in sentiment analysis. We will also compare it to the computationally intensive models. As well as evaluating the small deep learning models performance versus industry grade models it will also be compared to traditional ML techniques, and standard non-transformer deep learning networks. Some of the other models for evaluation include: CNNs, LSTM, LSTM-CNN, Logistic Regression, Naive Bayes, and Random Forest learners. These benchmarks of performance will be attained from the existing literature for this well tested dataset.

With the research into how a smaller sized model can hopefully outperform the traditional ML techniques while achieving reasonable scores when compared to the larger models we hope to show how a locally runnable model can still perform sentiment analysis at a high level. This can show ways to reduce the deployment and computation costs of the larger models with sufficient hyperparameter tuning of the smaller deep learning model. We can also perhaps show that there are diminishing returns when using more and more complex NLP models for tasks that aren't too hard (like most sentiment analysis)


# What you will do 
``` 
(Approach, 4-5+ sentences) - Be specific about what you will implement and what existing code you will use. Describe what you actually plan to implement or the experiments you might try, etc. Again, provide sufficient information describing exactly what you’ll do. One of the key things to note is that just downloading code and running it on a dataset is not sufficient for a description or a project! Some thorough implementation, analysis, theory, etc. have to be done for the project.
```

We will first setup SmolLM, a publicy available 135M-parameter LLM on Georgia Tech's PACE ICE GPUs. Then we will setup code to establish the baseline performance of this model used for sentiment analysis. We will then try to use few-shot learning and fine-tuning to improve the model's performance on sentiment analysis, using a dataset from IMDB movie reviews. We will perform an in-depth analysis explaining what model parameters are important, which can be tuned and why, and how the fine-tuning process can improve over our baseline performance. We will also assess what performance can be reached with the same evaluation procedure, when replacing the LLM to another model such as the ones mentioned in the project summary.



# Resources / Related Work & Papers 
``` 
(4-5+ sentences). What is the state of art for this problem? Note that it is perfectly fine for this project to implement approaches that already exist. This part should show you’ve done some research about what approaches exist.
```

For state-of-the-art sentiment analysis, transformer-based models like BERT, RoBERTa, and GPT-3 have demonstrated remarkable performance due to their ability to capture contextual information and semantic nuances. These models leverage pre-trained embeddings fine-tuned on vast datasets, resulting in a deep understanding of sentiment at both the word and sentence levels. 
 Other transformer-based approaches, such as T5 and DeBERTa, utilize enhanced architecture to improve the capture of syntactic and semantic structures, further boosting performance. Notably, SOTA models now integrate task-specific tuning or domain adaptation for optimal results in specific applications, like social media sentiment analysis. These models frequently surpass traditional benchmarks and achieve top scores on popular sentiment datasets, including SST-2 and IMDB. The advantage of complex models is that for the harder examples (such as people using sarcasm), a real in-depth understanding of humans and their use of language can be required.

For non-transformer-based methods, RNNs (e.g., LSTMs and GRUs) and CNNs were previously popular due to their effectiveness in sequential text processing, capturing dependencies in text through memory and convolutional filters. Though less context-aware than transformers, these architectures still perform well when combined with pre-trained word embeddings (like GloVe or Word2Vec), offering competitive sentiment analysis performance at lower computational costs.

These 5 papers are very relevant to our project :

1. Link to some benchmarks for models on the IMDB dataset we are using: [link](https://paperswithcode.com/sota/sentiment-analysis-on-imdb)


2. This paper talks about traditional methods and their performance on the dataset: [link](../documentation/Analyzing_Sentiment_using_IMDb_Dataset.pdf)

3. Another paper on deep nns run on the same dataset (CNN, LSTM, and LSTM-CNN): [link](../documentation/Performance_Analysis_of_Different_Neural_Networks_for_Sentiment_Analysis_on_IMDb_Movie_Reviews.pdf)

4. Very similar research to paper 3: [link](../documentation/ssrn-3403985.pdf)

5. Paper doing traditional ML methods for our dataset: [link](../documentation/Sentiment_Analysis_of_IMDb_Movie_Reviews__A_comparative_study_on_Performance_of_Hyperparameter-tuned_Classification_Algorithms.pdf)"



- [1] paperswithcode.com/sota/sentiment-analysis-on-imdb
- [2] Tripathi, Sandesh, et al. "Analyzing sentiment using IMDb dataset."
- [3] Haque, Md Rakibul, Salma Akter Lima, and Sadia Zaman Mishu. "Performance analysis of different neural networks for sentiment analysis on IMDb movie reviews."
- [4] Ali, Nehal Mohamed, Marwa Mostafa Abd El Hamid, and Aliaa Youssif. "Sentiment analysis for movies reviews dataset using deep learning models."
 - [5] Ghosh, Ayanabha. "Sentiment analysis of IMDb movie reviews: A comparative study on performance of hyperparameter-tuned classification algorithms."


# Datasets 
IMDb Movie Reviews for Sentiment Analysis:
* https://ai.stanford.edu/~amaas/data/sentiment/

Other links to the dataset:
* https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews 
* https://paperswithcode.com/dataset/imdb-movie-reviews

# List your Group members.
 - Jaden Zwicker
 - Brieuc Popper
 - Houshmand Abbaszadeh




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
