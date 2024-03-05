# Project Overview

This project processes 50K IMDb reviews, each one with a sentiment of 0 or 1, then it trains several different models, depending on our choice (mostly our GPU limit from Google Colab). We compare them, then train a bidirectional LSTM, and compare again (it's the best).

## Preprocessing  

![Preprocessing Steps](https://github.com/YanivGabay/NLP_EX2/assets/154590609/e86b6aa4-7f77-4100-ac8d-c7d4cd596d5b)

The basic instructions were followed with an exception for step 2 - (remove stop-words). After examining the data, we saw that many words important for sentiment analysis are considered stop words, for example, the word "bad" was one of them. Therefore, we found:

![Afinn Library Usage](https://github.com/YanivGabay/NLP_EX2/assets/154590609/166ce8cb-8f57-4d98-9d03-6a7bcced2f2e)

Afinn is a cool library that can grade us stop words from -5 to 5 (bad till neutral till good sentiment).

![Afinn Example](https://github.com/YanivGabay/NLP_EX2/assets/154590609/d96821af-3306-48e1-a247-7ca953d7f26e)

## Results:

![Results](https://github.com/YanivGabay/NLP_EX2/assets/154590609/dd038d2e-6116-43ad-9fba-dd34219c29a2)

This way, we filtered the stop words. Also, words like "not" are considered neutral, as they are "flippers" that flip the meaning, so we didn't remove some flippers as well.

![Flippers Example](https://github.com/YanivGabay/NLP_EX2/assets/154590609/4e0947f1-a33d-4113-a14b-91e22ff3486c)

![Additional Example](https://github.com/YanivGabay/NLP_EX2/assets/154590609/f93a7d3b-8ab3-409f-9d73-4075f7ac5de2)

It's crucial to view the entire dataset and explore it to catch any anomalies, such as non-ASCII characters, odd words, etc. Of course, perfection is unattainable, but throughout our work, we exported the data after each alteration to ensure quality.

## WORD POS:

![WORD POS Example](https://github.com/YanivGabay/NLP_EX2/assets/154590609/53c8b1e0-8111-4a36-b0ac-1453e98c965e)

Some auxiliary verbs like "was" changed meaning after lemmatization, "was" was turning into "wa," which isn't even a word. We also wanted to preserve words like "didn't," "wasn't," and most uses of "not" with auxiliary verbs. So we used:

![POS Tagging](https://github.com/YanivGabay/NLP_EX2/assets/154590609/8f7009cf-23d8-46be-b657-d357f5ad2e58)

So:

![POS Tagging Example](https://github.com/YanivGabay/NLP_EX2/assets/154590609/6566b828-c14b-43b4-a04e-2282400b6e4a)

## Preparing For Training:

![Preparing for Training](https://github.com/YanivGabay/NLP_EX2/assets/154590609/e327f946-017e-4da8-b5a5-205847f9a71c)
![Additional Training Prep](https://github.com/YanivGabay/NLP_EX2/assets/154590609/91c1655c-0b1e-4aff-b0a6-12277280936d)

## Training:

![Training Process](https://github.com/YanivGabay/NLP_EX2/assets/154590609/17cc33ca-a803-42ca-b62d-1769402aa9b2)

We tried 3 different optimizers; Adam gave the best results, with the variant of dim size (128|256), and we tried many different dropouts for different configs.

![Training Configurations](https://github.com/YanivGabay/NLP_EX2/assets/154590609/d0763e2e-0e6f-4995-bdb5-5f738ee205aa)

For each config, we can create a model, with or without GloVe, with unfreeze without unfreeze options. For each model, we will print its train loss test loss over epochs graph, and confusion matrix.

We added GloVe embedding, with the option to unfreeze changes during training which adapts the weights to our vocabulary.

## BILSTM:

![BILSTM Model](https://github.com/YanivGabay/NLP_EX2/assets/154590609/2011f739-9c67-4399-8e70-2123298936a1)

Our highest epoch count was 20 - yielding about 90% accuracy, an improvement. There's little overfitting, and we believe with longer training and better GPU resources, we can achieve higher accuracies.

## Final Results:

- Results for experiment 1: 88.21%
- Results for experiment 2: 88.43%
- Results for experiment 3: 88.13%
- Results for BiDir LSTM: 90.23%

The standard networks reached a peak of 88.4% (experiment_2). Thus, in the end, BiLSTM is superior to regular networks, although they performed much better than expected. With more training time and access to more powerful GPUs to manipulate the BiLSTM network and play with more parameters, embeddings, and large dictionaries, we anticipate reaching much higher accuracies.

Preprocessing proved to be highly significant, and constantly reviewing the data after each processing step was crucial in achieving our results.
