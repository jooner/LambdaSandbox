<Project Lambda Sandbox>

This is a copy of the homegrown codebase for implementing a basic LSTM for text generation.
Initially, it simply compared euclidean distances between words that were embedded using GloVe.
Next, it tried to generate next words solely based on GloVe's embeddings by returning the closest word.

The next step was to use Tolstoy's novels, Anna Karenina and War and Peace, to train a vanilla double-layered LSTM.
This was done initially by training the network to return a vector that matches the dimensions of the embedding we used (i.e. 50, 100, 300)
However, we faced problems in dealing with unknown words. This was tackled by arbitrarily adding noise to the previously seen word and estimating where the next word should be. This was a naive way of dealing with the problem.

Ultimately, the problem was that we needed to traverse through the entire encoding dictionary to find the next word. That is, once the layer produced an output of, say, 50 dimensions, we needed to measure the euclidean distance from that point to all word embeddings and find the one with the minimum distance. This was problematic because (1) semantically, minimum distance may not translate to "appropriate next word"--it may well be the former word, or an opposite word, or something entirely different. (2) computationally, this was extremely expensive, especially if we increased the dimensionality of the word embedding, and the vocabulary size.

Alternatively, we chose to use only the top 22,000 words that appeared in our training set, and to tokenize those as one-hot vectors. The input remained the same (the n-dimensional embedding), but the output was given as a 22,000-dimensional vector, that was treated with a softmax layer. This drastically increased the training time, but drastically decreased the testing time, once the layers were trained. It also eliminates the trickiness of dealing with euclidean distance and endeavoring to justify its use in a semantic context.

We still face many hurdles in word-embedding. The next improvement would be to add in a character-based convolutional network to concatenate onto unknown word tokens that will effectively grasp proper nouns, as well as similarity of syntax for past-present-future tenses, adjective-adverb, etc.

The next step of this project is to go beyond text generation, and to apply it for Q&A by boosting the memory retained by the network through a novel architecture.

Stay Tuned!  