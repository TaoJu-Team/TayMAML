import tensorflow as tf


class D3DQNBrain:
    name = 'D3DQN'

    def __init__(self, features, actions, w_initializer, b_initializer, name, inputs, n_l1):
        self.actions = actions
        self.features = features
        self.w = w_initializer
        self.b = b_initializer
        self.name = name
        self.input = inputs
        self.n_l1 = n_l1

    def __call__(self):
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [self.features, self.n_l1], initializer=self.w, collections=self.name)
            b1 = tf.get_variable('b1', [1, self.n_l1], initializer=self.b, collections=self.name)
            l1 = tf.nn.relu(tf.matmul(self.input, w1) + b1)

        with tf.variable_scope('Value'):
            w2 = tf.get_variable('w2', [self.n_l1, 1], initializer=self.w, collections=self.name)
            b2 = tf.get_variable('b2', [1, 1], initializer=self.b, collections=self.name)
            self.V = tf.matmul(l1, w2) + b2

        with tf.variable_scope('Advantage'):
            w2 = tf.get_variable('w2', [self.n_l1, self.actions], initializer=self.w, collections=self.name)
            b2 = tf.get_variable('b2', [1, self.actions], initializer=self.b, collections=self.name)
            self.A = tf.matmul(l1, w2) + b2

        with tf.variable_scope('Q'):
            out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))
        return out


class BrainSmall(tf.keras.Model):
    name = 'D3DQN'

    def __init__(self, state_size, num_VMs):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(12, activation=tf.nn.relu)
        self.dense_2 = tf.keras.layers.Dense(15, activation=tf.nn.relu)
        self.dense_3 = tf.keras.layers.Dense(15, activation=tf.nn.relu)
        self.dense_4 = tf.keras.layers.Dense(12, activation=tf.nn.relu)
        self.value = tf.keras.layers.Dense(1, activation=tf.nn.relu)
        self.advantage = tf.keras.layers.Dense(num_VMs, activation=tf.nn.relu)
        self.dense_5 = tf.keras.layers.Dense(1)

    def call(self, state):
        state = self.dense_1(state)
        state = self.dense_2(state)
        state = self.dense_3(state)
        state = self.dense_4(state)
        value = self.value(state)
        advantage = self.advantage(state)
        state = value + (advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True))
        state = self.dense_5(state)

        return tf.expand_dims(tf.squeeze(state, axis=-1), axis=0)
