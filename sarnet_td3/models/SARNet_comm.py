import collections

import numpy as np
import tensorflow as tf

import sarnet_td3.common.ops as ops

RRLCellTuple = collections.namedtuple("RRLCellTuple", ("memory"))


class RRLCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """
    Input:
        Query: Query from the agent "i" recurrent cell
        [batchSize, query_units]

        Keys: Current representation of the observation from each agent
        [batchSize, n, query_units]

        numAgents: number of agents
        [batchSize]

        batchSize: Tensor Scalar
        [batchSize]

        Values: Representation of observation of all agents
        [batchSize, n, mem_units]
    """

    def __init__(self, train, args, reuse=None):
        self.args = args
        self.num_agents = self.args.num_adversaries

        self.dropouts = {}
        self.dropouts["memory"] = self.args.memory_dropout
        self.dropouts["read"] = self.args.read_dropout
        self.dropouts["write"] = self.args.write_dropout

        self.train = train
        self.reuse = reuse

        self.none = tf.zeros((1, 1), dtype=tf.float32)

        self.attn_scale = np.sqrt(self.args.query_units)

    """Cell State Size"""

    @property
    def state_size(self):
        return RRLCellTuple(self.args.value_units)

    """Cell output size. No outputs used for now"""

    @property
    def output_size(self):
        return self.args.value_units

    """
    The Control Unit

    Input:
    queryInput: external input to the control unit (RNN output of specific agent)
    [batchSize, query_units]

    Keys: Observation embeddings from all agents
    [batchSize, n, query_units]

    num_agents: Total number of agents in the reasoning operation

    query: previous query control hidden state value
    [batchSize, query_units]

    Returns:
    New control state
    [batchSize, #agents]
    """

    def control(self, query, keys, reuse=None):
        with tf.compat.v1.variable_scope("control", reuse=reuse):
            dim = self.args.query_units
            interactions = tf.expand_dims(query, axis=-2) * keys
            # Multiplies Q * K and reduces across the agent dimension
            # Input: [batch, #agents, dim] -> [batch, #agents] through a linear transformation
            if not self.args.tar_attn:
                logits = ops.inter2logits(interactions, dim)
            else:
                logits = ops.inter2logits(interactions, dim, sumMod="SUM")
            attention = tf.nn.softmax(logits / self.attn_scale)

        return attention

    """
    The Read Unit

    Input:
    valueBase: [?, n, mem_size]
    memory: [?, mem_size]
    query: [?, query_units]

    Returns:
    Information: [?, mem_size]    
    """

    def read(self, valueBase, memory, query, reuse=None):
        with tf.compat.v1.variable_scope("read", reuse=reuse):
            # memory dropout
            newMemory = memory
            if self.args.memory_dropout:
                newMemory = tf.nn.dropout(memory, self.dropouts["memory"])

            # Convert memory dim from [batch, dim] to [batch, #agents, dim]
            newMemory = tf.expand_dims(newMemory, axis=-2)
            newMemory = tf.zeros_like(valueBase) + newMemory

            interactions = newMemory * valueBase

            # Perform Linear{(Memory * Value) + Value}
            if self.args.FeedInteractions:
                interactions = tf.add(interactions, valueBase)
                interactions = ops.linear(interactions, self.args.value_units, self.args.value_units, name="interactAfterAdd", reuse=reuse)
            else:
                # Perform Linear(Memory * Value) + Value
                interactions = tf.add(ops.linear(interactions, self.args.value_units, self.args.value_units, name="interactBeforeAdd", reuse=reuse), valueBase, name="ValueMemSUM")

            # Query: [batch, #agents], Inter: [batch, #agents, dim]
            # Output: [batch, dim]
            readInformation = ops.att2Smry(query, interactions)
            dim = self.args.value_units
            if self.args.FeedOldMemory:
                dim += dim
                readInformation = tf.concat([readInformation, memory], axis=-1)
            # read dropout
            if self.args.read_dropout:
                readInformation = tf.nn.dropout(readInformation, self.dropouts["read"])
            readInformation = ops.linear(readInformation, dim, self.args.value_units, name="finalMemory", reuse=reuse)

            if self.args.memoryBN:
                newMemory = tf.contrib.layers.batch_norm(readInformation, decay=self.args.bnDecay, center=self.args.bnCenter,
                                                         scale=self.args.bnScale, is_training=self.train,
                                                         updates_collections=None, reuse=reuse)
            else:
                newMemory = readInformation

            return newMemory

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__
        with tf.compat.v1.variable_scope(scope, reuse=self.reuse):
            memory = state
            query, keys, values = inputs

            # Reshape keys/values to [agent, batch, dim] to [batch, agent, dim]
            keys = tf.stack(keys, axis=-2)
            values = tf.stack(values, axis=-2)
            ## Control unit Output: [batch, #agents]
            newAttn = self.control(query, keys)

            ## Read Unit [batch, dim]
            info = self.read(values, memory, newAttn)

            newState = info

            return newAttn, newState

