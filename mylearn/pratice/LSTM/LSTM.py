import time
import numpy as np
import tensorflow as tf
import reader

# define the data input class
class PTBInput(object):

    # init
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        ## num_steps is the unrolled steps of LSTM
        self.num_steps = num_steps = config.num_steps
        ## caculate the size of each epoch
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        ## use the reader.ptb_producer get the features(input_data), and the labels(targets)
        ### this targets is a defined tensor
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name = name
        )

# define the language model
class PTBModel(object):

    # init
    ## is_trainging is the training sign
    ## input_
    def __init__(self, is_training, config, input_):
        self._input = input_
        ## get the batch_size
        batch_size = input_.batch_size
        ## get the num_steps (num_steps is the unrolled steps of LSTM)
        num_steps = input_.num_steps
        ## get the hidden_size( i think i needn't say what is this angin)
        size = config.hidden_size
        ## get the size of vocabulary list
        vocab_size = config.vocab_size

        ## set the LSTM unit
        def lstm_cell():
            ### size = hidden_size
            return tf.contrib.rnn.BasicLSTMCell(
                size, forget_bias = 0.0, state_is_tuple = True
            )

        ## set the lstm_cell function to attn_cell
        attn_cell = lstm_cell
        ## if training and the config.keep_prob < 1
        ### then add the Dropout layer to the lstm_cell
        if is_training and config.keep_prob < 1:
            def  attn_cell():
                #### let the lstm_cell()'s output as the input for DropoutWrapper
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob = config.keep_prob
                )
        ## pile the lstm_cell (i think here should be attn_cell) num_layers times
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)],
            state_is_tuple = True
        )
        ## set the LSTM cell's state as zero
        self._initial_state = cell.zero_state(batch_size, tf.float32)


        ## limit the caculate operation in cpu
        with tf.device("/cpu:0"):
            ### embedding the data
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=tf.float32
            )
            ### get the embed of input_.input_data from embedding
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        ## if is_training and keep_prob < 1, add the dropout
        ### the dropout(inputs, keep_prob) is mean to get (keep_prob / 1) x 100% data from input
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            ### set the loop size is num_steps
            for time_step in range(num_steps):
                ### at the sencond loop , use the tf.get_variable_scope().reuse_variables() set the reuse variables
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                ### input the (inputs, state) to the LSTM cell
                ### get the result(cell_output) and the updated state
                #### input[a, b, c]
                    ##### a: the index of sample in one batch
                    ##### b: the index of word in one sample
                    ##### c: the dimension number of the word's vector
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                ### add the result to the output list
                outputs.append(cell_output)

        ## change the outputs struct to [outputs/size, size]
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        ## init the weight
        softmax_w = tf.get_variable(
            "softwax_w", [size, vocab_size], dtype=tf.float32
        )
        ## init the bias
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        ## get the result by logits = wx + b
        logits = tf.matmul(output, softmax_w) + softmax_b
        ## caculate the loss
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(input_.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)]
        )

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        ## set the lr as the untrainable Variable
        self._lr = tf.Variable(0.0, trainable=False)
        ## get the all trainable Variable
        tvars = tf.trainable_variables()
        ## set the max_L = max_grad_norm to control the gradient
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        ## set the optimizer(ok, i say this function too much time)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        ## create the train operation
        self._train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        ## create the palceholder for lr to control learning rate
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name = "new_learning_rate"
        )
        ## set the _new_lr to the _lr
        self._lr_update = tf.assign(self._lr, self._new_lr)
    # we can control learning rate at outside
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict = {self._new_lr: lr_value})


    @property
    def input(self):
        return self._input


    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

# init the SamllConfig, ok, this is according with the modelize programing
class SmallConfig(object):
    ## init the weigth of the net
    init_scale = 0.1
    ## set the learning rate
    learning_rate = 1.0
    ## set the gradient's max norm
    max_grad_norm = 5
    ## set the Stackingable layers
    num_layers = 2
    ## LSTM gradient backpropagation's step
    num_steps = 20
    ## the number of hidden node
    hidden_size = 200
    ## initial learning rate could train times
    max_epoch = 4
    ## altogether cound train times
    max_max_epoch = 13
    ## dropout layer retain proportion
    keep_prob = 1.0
    ## this is the learning rate's decay speed
    lr_decay = 0.5
    ## batch_size is batch's size
    batch_size = 20
    ## vocabulary's size
    vocab_size = 10000
# the same with the small
class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_szie = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
# the same with the small
class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_eppch = 14
    max_max_epoch = 35
    keep_prob = 0.35
    lr_decay = 1/1.15
    batch_size = 20
    vocab_size = 10000

class TestConfig(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_eppch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, eval_op = None, verbose = False):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    ## create the fetches to get the run results
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state
    }

    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        ### get the LSTM's all state to feed_dict
        #### enumerate(["a", "b", "c"]) ==> [(0, "a"), (1, "b"), (2, "c")]
        #### ok, now, I don's konw what is the c or h
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        ### run and get the result
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0/model.input.epoch_size, np.exp(costs/iters),
                   iters * model.input.batch_size / (time.time() - start_time)))
    return np.exp(costs / iters)

# get the train data, valid data and test data by reader.ptb
raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data

# define train model's config as SamllConfig
config = SmallConfig()
# define test model's config as SamllConfig
## the test model's config must same with train model's config
## but the test model's batch_size and the num_steps must be 1
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

# create a default Graph
with tf.Graph().as_default():
    ## create a initializer for initing all parameter between (-config.init_scale) and (config.init_scale)
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    ## create the Train model
    ### tf.name_scope is use for Managing namespaces
    with tf.name_scope("Train"):
        ### create the train_input by PTBInput
        #### the config is SamllConfig
        #### the train_data is train data
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
        ### tf.Variable_scope is use for Realizing variable sharing with tf.get_Variable()
        with tf.variable_scope("Model", reuse = None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, input_=train_input)

    ## create the Valid model
    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name='ValidInput')
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, input_=valid_input)

    ## create the Test model
    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data,
                              name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config, input_=test_input)


    ## use the tf.train.Supervisor create a training manager
    sv = tf.train.Supervisor()
    with sv.managed_session() as session:
        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i+1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose = True)

            print("Epoch: %d Train Perplexity: %.3f" % (i+1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid)
            print("Epoch: %d Valid Perplexity: %.3f" % (i+1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest)
        print("Test Perplexity: %.3f" % test_perplexity)


