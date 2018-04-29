import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

#############because the data has some problem, so I haven't use it###################
#download the data
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)

    ##if the file is wanted, then print the right word
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?'
        )
    return filename

##download the data
# filename = maybe_download('text8.zip', 31344016)
########################################################################################


filename = "/home/cks/PycharmProjects/tensorflow/mylearn/pratice/Word2Vec/short.txt"

#unpack the data zip
def read_data(filename):
    with open(filename) as f:
        data = tf.compat.as_str(f.read()).split()

    return data

##get the unpacking data
words = read_data(filename)
print('Data size', len(words))


#get the top50000 as the vocabulary
#and then intercept the words
#mark the other word which is not the top50000 as 0
##set the vocabulary size
vocabulary_size = 5000
##intercept the words and mark the word
def build_dataset(words):
    ###get the most common words -- top50000(vocabulary_size)
    ####I guest this operation is set the word type which will be read
    count = [['UNK', -1]]
    ####get the top50000(vocabulary_size)
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    ###mark the count
    ####mark the words which is the top50000 as it's level and save at dictionary
    ####the count is ranked
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    ###mark all words
    ####mark the words which is the top50000 as it's level and save at data
    ####mark the words which is not the top50000 as 0
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    ###static the other words'(non-top50000) number
    count[0][1] = unk_count
    ###reverse the dictionary from(key, value) to (value, key)
    ####the zip function read the test.py(but i think the most useful way is read the origin code)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

#get the data, count, dictionary, reverse_dictionary
##data:  list--(level) of all words
##count: list--(top50000_word, [0]:unk_count [1-n]---?????? ) -- count[:5] -> [['UNK', 0], ('the', 55), ('of', 45), ('and', 27), ('to', 21)]
##dictionary:   dict()--(word:level) of top50000 words
##reverse_dictionary:   dict()--(level:word) of top50000 words
data, count, dictionary, reverse_dictionary = build_dataset(words)

#delete the origin words' list for saving the space
del words
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in  data[:10]])

#generate the train sample
data_index = 0

##batch_size: ok, batch_size is batch's szie
##num_skips: the number of sample which is generated of one word
##skip_window is the farthest distance of word's connect
##Attendtion: this function's parameter must accord with these limit
    ###num_skips <= skip_window x 2
    ###batch_size = k x num_skips ( k belong to Z)
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    ###init the batch array
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    ###init the labels array
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    ###init the max space's size
    span = 2 * skip_window + 1
    ###init a deque which size is span
    ####the buffer is saving the target word and the other words which is has the relation with the target word
    buffer = collections.deque(maxlen=span)

    ###push span element to buffer from data(level list) even though the data size is not enough(but this situation is not possible occur)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    ###the first loop
    for i in range(batch_size // num_skips):
        ###define the target equal skip_window, and consider the skip_window(th) word is the target word
        target = skip_window
        ###define the word's number(identifier) which is need to avoid of word's list, the first number is target word's number
        targets_to_avoid = [ skip_window ]

        ###generate the (buffer[skip_window], buffer[target]),
        #### the buffer[skip_window] as the feature, but it's the origin target word
        #### the buffer[target] as the label, but it's the new "target word" which is randow index from buffer
        for j in range(num_skips):
            ####generate the random index of buffer as the new "target word" which as the label
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            ####because this word will be use, so , we push it into the avoid list, we'll not use it at this loop again
            targets_to_avoid.append(target)
            ####get the origin target as the feature, and push it into the batch
            batch[i * num_skips + j] = buffer[skip_window]
            ####get the random "target word" as the label, and push it into the labels
            labels[i * num_skips + j, 0] = buffer[target]
        ###push the new word into the buffer, and the first word of the buffer will be delete
        buffer.append(data[data_index])
        ###update the data_index which is the number of next word which will be push into the buffer
        data_index = (data_index + 1) % len(data)
    return batch, labels

#get the batch and appropriate labels
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->' , labels[i, 0],
          reverse_dictionary[labels[i, 0]])

#set the parameter
##the batch_size
batch_size = 128
##the embedding size, the word to be the [embedding, 1] data
embedding_size = 128
##the farthest distance of the word which target word can connect
skip_window = 1
##the number of samples of target word
num_skips = 2

##the word's number of validate
valid_size = 16
##the number of frequecest word
valid_window = 100
##get 16(valid_size) words from frequecest word list(100)
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
##set the noise words' number
num_sampled = 64


#define the Word2Vec net Struction
##create the tf.Graph(ok, i don's what is it)
graph = tf.Graph()
with graph.as_default():
    ###create the train datas' placeholder
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    ###create the train labels' placeholder
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    ###change the valid_example to tf.constant
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    ###limit the caculate operation in cpu
    with tf.device('/cpu:0'):
        ####use the tf.random_uniform random generate the word vector(embeddings)
        ####the vocabulary_size is 5000
        ####the number of the vector's dimension is 128
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
        )
        ###use the tf.nn.nembedding_lookup to search the train_inputs' embed
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        ###init the nce_weight[vocabulary_size, embedding_size]
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0/math.sqrt(embedding_size))
        )
        ####init the nce_biases[vocabulary_size] ??????????????? isn't it should be embedding_size?? why is vocabulary_size?
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    ##use the tf.nn.nce_loss function to caculate the loss of words' vector(embedding) in train data
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))

    ##set the optimizer
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    ##Caculate the L2 norm
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    ##get the standardized normalized_embeddings
    normalized_embeddings = embeddings / norm
    ##use the tf.nn.nembedding_lookup to search the valid_dataset' embed
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset
    )
    ##Caculate the similarity of valid_embeddings' word with all words
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True
    )
    ##init the all model's parameters
    init = tf.global_variables_initializer()


#the run time
num_steps = 100001
with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window
        )
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step ", step, ":", average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i,:]).argsort()[1:top_k+1]
                log_str = "Nearest to % s:" % valid_word

                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

#this is the plot function
def plot_with_labels(low_dim_embs, labels, filename = 'tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18,18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy = (x, y),
                     xytext = (5, 2),
                     textcoords = 'offset points',
                     ha = 'right',
                     va = 'bottom')

    plt.savefig(filename)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
