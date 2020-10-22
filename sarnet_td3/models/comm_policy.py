import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from sarnet_td3.models.SARNet_comm import RRLCell
import sarnet_td3.common.ops as ops


class CommActorNetwork():
    def __init__(self, is_train, args, reuse=None):
        self.args = args
        self.is_train = is_train
        self.reuse = reuse
        if self.args.QKV_act:
            self.QKVAct = tf.nn.relu
        else:
            self.QKVAct = None

        self.attn_scale = np.sqrt(self.args.query_units)

    def _gru(self, reuse):
        # return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=self.args.gru_units)
        # return tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=args.gru_units)
        return tf.contrib.rnn.GRUCell(num_units=self.args.gru_units, reuse=reuse)

    def _lstm(self, reuse):
        return tf.contrib.rnn.LSTMCell(num_units=self.args.gru_units, reuse=reuse)

    def _sarnet_comm(self, reuse):
        return RRLCell(self.is_train, self.args, reuse=reuse)

    # Embedding Network common to all networks
    """
    Args
    Input: 
    Encoder state if a recurrent network - [batch, time, gru_dim]
    Obs dim - [batch, time, obs_dim]
    """
    def obs_encoder(self, x, state, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            out = x
            if self.args.recurrent:
                if self.args.encoder_model == "GRU":
                    enc = self._gru(reuse)
                elif self.args.encoder_model == "LSTM":
                    enc = self._lstm(reuse)
                out = tf.expand_dims(out, axis=-2)
                out, state = tf.compat.v1.nn.dynamic_rnn(enc, out, initial_state=state, time_major=False, scope="rnn_encoder")
            else:
                out = layers.fully_connected(out, num_outputs=self.args.encoder_units, activation_fn=tf.nn.relu, scope="mlp_encoder", reuse=reuse)
                state = out

            return out, state

    def action_enc(self, input, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            out = layers.fully_connected(input, num_outputs=self.args.action_units, activation_fn=tf.nn.relu, scope="actproj0", reuse=reuse)
            return out

    def action_pred(self, input, num_outputs, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            out = layers.fully_connected(input, num_outputs=num_outputs, activation_fn=None, scope="actproj1", reuse=reuse)
            return out

    def get_query_vec(self, x, scope, reuse, comm_type="sarnet"):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            if self.args.nheads > 1 and not comm_type == "tarmac":
                query_out = []
                for i in range(self.args.nheads):
                    query_out.append(layers.fully_connected(x, num_outputs=self.args.query_units, activation_fn=self.QKVAct, scope="query_encoder"+str(i), reuse=reuse))
            else:
                query_out = layers.fully_connected(x, num_outputs=self.args.query_units, activation_fn=self.QKVAct, scope="query_encoder", reuse=reuse)

            return query_out

    def get_key_val_vec(self, x, msg_i, scope, reuse, comm_type="sarnet"):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x_val = x
            if self.args.FeedMsgToValueProj and not comm_type == "tarmac":
                x_val = tf.concat([x_val, msg_i], axis=-1)
            # Input [batch, dim]
            if self.args.nheads > 1 and not comm_type == "tarmac":
                key_out = []
                value_out = []
                for i in range(self.args.nheads):
                    key_out.append(layers.fully_connected(x, num_outputs=self.args.key_units, activation_fn=self.QKVAct, scope="key_encoder"+str(i), reuse=reuse))
                    value_out.append(layers.fully_connected(x_val, num_outputs=self.args.value_units, activation_fn=self.QKVAct, scope="value_encoder"+str(i), reuse=reuse))
            else:
                key_out = layers.fully_connected(x, num_outputs=self.args.key_units, activation_fn=self.QKVAct, scope="key_encoder", reuse=reuse)
                value_out = layers.fully_connected(x_val, num_outputs=self.args.value_units, activation_fn=self.QKVAct, scope="value_encoder", reuse=reuse)

        return key_out, value_out

    def compute_attn_dotprod(self, query, key_n, value_n, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            query = tf.expand_dims(query, axis=1)
            att_smry = query * key_n
            att_smry = ops.inter2logits(att_smry, self.args.query_units, sumMod="SUM")  # Get dot product
            att_smry = tf.nn.softmax(att_smry) / self.attn_scale  # (batch_size, #Agents)
            # Now do interaction with the value (batch_size, #Agents) * (batch_size, #Agents, value_dim)
            message = tf.einsum('bn,bnd->bd', att_smry, value_n)
            _obs_act_in = tf.contrib.layers.batch_norm(message, decay=self.args.bnDecay, center=self.args.bnCenter,
                                                       scale=self.args.bnScale, is_training=self.is_train,
                                                       updates_collections=None, scope="batch_norm_obs", reuse=reuse)
            return message, att_smry

    def commnet_com(self, x_n, n, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            message = tf.add_n(x_n) / n
            return message

    def ic3_gating_com(self, state_t1, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            _gate = layers.fully_connected(state_t1, num_outputs=1, activation_fn=tf.nn.softmax, scope="gate_encoder", reuse=reuse)
            _comm_gate_out = tf.multiply(state_t1, _gate)

            return _comm_gate_out

    """ Communication Architectures: CommNet, SARNet, TarMAC, IC3Net"""
    # Input is defined as the following:
    # x_n: {Indexes: [0 ~ N-1] - Observation; [N ~ 2*N-1] - GRU State; [2*N ~ 3*N-1] - GRU State 2; [3*N ~ 4*N-1] - Message_tp}

    def sarnet(self, x_n, num_outputs, p_index, n, n_start, n_end, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Gather inputs fed for the graph
            obs_n_in = [x_n[i] for i in range(n_start, n_end)]
            h_n_in = [x_n[i + n] for i in range(n_start, n_end)]
            c_n_in = [x_n[i + 2 * n] for i in range(n_start, n_end)]
            msg_n_in = [x_n[i + 3 * n] for i in range(n_start, n_end)]

            # Collect _encoder status from all agents
            obs_enc_out_n = []
            _key_enc = []
            _value_enc = []

            # Aggregate hidden states from other agents, compute message, and then use it to
            # augment observation for encoder
            if self.args.SARplusIC3:
                saric3_comm_n = []
                for i in range(n_start, n_end):
                    _gating_comm = self.ic3_gating_com(h_n_in[i], scope="sarplusic3_gate", reuse=tf.compat.v1.AUTO_REUSE)
                    saric3_comm_n.append(_gating_comm)

            # Weight sharing between the encoder layers of all agents
            for i in range(n_start, n_end):
                # Encode observation
                # Pre encode
                obs_enc_in_i = obs_n_in[i]
                if self.args.pre_encoder:
                    obs_enc_in_i = layers.fully_connected(obs_enc_in_i, num_outputs=self.args.gru_units, activation_fn=tf.nn.relu, scope="sarnet_pre_enc", reuse=tf.compat.v1.AUTO_REUSE)

                if self.args.FeedOldMemoryToObsEnc:
                    obs_enc_in_i = tf.concat([msg_n_in[i], obs_enc_in_i], axis=-1)

                if self.args.SARplusIC3:
                    # Get all gating actions for all agents except "i" to compute mean
                    saric3_comm_j = saric3_comm_n[:i] + saric3_comm_n[i+1:]
                    # Compute messages mean of pre-scaled hidden states
                    message_t1_i = self.commnet_com(saric3_comm_j, n_end - n_start - 1, scope="sarplusic3_mean", reuse=tf.compat.v1.AUTO_REUSE)
                    obs_enc_in_i = tf.concat([obs_enc_in_i, message_t1_i], axis=-1)

                # Encode observation with a recurrent model
                if self.args.encoder_model == "LSTM":
                    obs_enc_state_i = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c_n_in[i], h_n_in[i])
                else:  # GRU
                    obs_enc_state_i = h_n_in[i]

                _enc_out, _enc_state = self.obs_encoder(obs_enc_in_i, obs_enc_state_i, scope="sarnet_enc", reuse=tf.compat.v1.AUTO_REUSE)

                # Populate encoder output for recurrent cases
                _enc_out = tf.squeeze(_enc_out, axis=-2)
                obs_enc_out_n.append(_enc_out)
                if i is p_index:
                    _enc_state_p_idx = _enc_state

                # Generate nhead keys and values from all agents
                _key_proj, _value_proj = self.get_key_val_vec(_enc_out, msg_n_in[i], scope="sarnet_KV_Projs", reuse=tf.compat.v1.AUTO_REUSE)
                _key_enc.append(_key_proj)
                _value_enc.append(_value_proj)
            # Generate nhead-queries only from action agent
            _query_enc = self.get_query_vec(obs_enc_out_n[p_index - n_start], scope="sarnet_QProj", reuse=tf.compat.v1.AUTO_REUSE)

            # Reasoning Cell
            self.sarnet_cell = self._sarnet_comm(reuse=tf.compat.v1.AUTO_REUSE)
            sarnet_input = (_query_enc, _key_enc, _value_enc)

            attn_output, memory_state = self.sarnet_cell(sarnet_input, x_n[p_index + int(3 * n)])
            # Merge new reasoned message and observation encoding
            _obs_act_in = obs_enc_out_n[p_index - n_start]

            if self.args.bNorm_state:
                _obs_act_in = tf.contrib.layers.batch_norm(_obs_act_in, decay=self.args.bnDecay, center=self.args.bnCenter,
                                             scale=self.args.bnScale, is_training=self.is_train, updates_collections=None, scope="batch_norm_obs", reuse=tf.compat.v1.AUTO_REUSE)

            _enc_input_2 = tf.concat([_obs_act_in, memory_state], axis=-1)
            if self.args.TwoLayerEncodeSarnet:
                _enc_input_2 = self.action_enc(_enc_input_2, scope="sarnet_act_enc", reuse=tf.compat.v1.AUTO_REUSE)
            action = self.action_pred(_enc_input_2, num_outputs, scope="sarnet_act", reuse=tf.compat.v1.AUTO_REUSE)

            return action, _enc_state_p_idx, memory_state, attn_output

    def tarmac(self, x_n, num_outputs, p_index, n, n_start, n_end, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Input x_n -> 0~n: obs, n~2n: GRU, 2n~3n: Memory
            # n: Total number of agents, n_comm: Total number of agents in the communication
            # Gather inputs fed for the graph
            obs_n_in = [x_n[i] for i in range(n_start, n_end)]
            h_n_in = [x_n[i + n] for i in range(n_start, n_end)]
            c_n_in = [x_n[i + 2 * n] for i in range(n_start, n_end)]
            msg_n_in = [x_n[i + 3 * n] for i in range(n_start, n_end)]

            # Collect _encoder status from all agents
            obs_enc_out_n = []
            _key_enc = []
            _value_enc = []

            if self.args.TARplusIC3:
                _gating_comm_n = []
                for i in range(n_start, n_end):
                    _gating_comm = self.ic3_gating_com(h_n_in[i], scope="tarplusic3_gate", reuse=tf.compat.v1.AUTO_REUSE)
                    _gating_comm_n.append(_gating_comm)

            # Start from first agent of one type to the last. (e.g. 0 - num_adv-1) or (num_adv to n)
            for i in range(n_start, n_end):
                # Encode observation
                obs_enc_in_i = obs_n_in[i]
                if self.args.pre_encoder:
                    obs_enc_in_i = layers.fully_connected(obs_enc_in_i, num_outputs=self.args.gru_units, activation_fn=tf.nn.relu, scope="tarmac_pre_enc", reuse=tf.compat.v1.AUTO_REUSE)

                if self.args.TARplusIC3:
                    # Get all gating actions for all agents except "i" to compute mean
                    taric3_comm_j = _gating_comm[:i] + _gating_comm[i + 1:]
                    message_t1_i = self.commnet_com(taric3_comm_j, n_end - n_start - 1, scope="tarplusic3_mean", reuse=tf.compat.v1.AUTO_REUSE)
                    obs_enc_in_i = tf.concat([obs_enc_in_i, message_t1_i, msg_n_in[i]], axis=-1)  # Concat message, mean hidden state (IC3) and observation encoding
                else:
                    obs_enc_in_i = tf.concat([obs_enc_in_i, msg_n_in[i]], axis=-1)  # Concat message and observation encoding

                # Encode observation with a recurrent model
                if self.args.encoder_model == "LSTM":
                    obs_enc_state_i = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c_n_in[i], h_n_in[i])
                else:  # GRU
                    obs_enc_state_i = h_n_in[i]

                _enc_out, _enc_state = self.obs_encoder(obs_enc_in_i, obs_enc_state_i, scope="tarmac_enc", reuse=tf.compat.v1.AUTO_REUSE)
                _enc_out = tf.squeeze(_enc_out, axis=-2)
                obs_enc_out_n.append(_enc_out)

                # Generate Keys and Values from all agents
                _key_proj, _value_proj = self.get_key_val_vec(_enc_out, msg_n_in, scope="tarmac_keyval", reuse=tf.compat.v1.AUTO_REUSE, comm_type = "tarmac")
                _key_enc.append(_key_proj)
                _value_enc.append(_value_proj)

            # Since TarMAC doesn't use nhead attention layers, we just use the first value from the list
            # Stack all keys and values of all agents to a single tensor
            _value_enc = tf.stack(_value_enc, axis=-2)
            _key_enc = tf.stack(_key_enc, axis=-2)

            # Generate Query only from action agent
            _query_enc = self.get_query_vec(obs_enc_out_n[p_index - n_start], scope="tarmac_query", reuse=tf.compat.v1.AUTO_REUSE, comm_type="tarmac")

            # Generate attention and message, attention only used for benchmarking results
            message_t1, attn = self.compute_attn_dotprod(_query_enc, _key_enc, _value_enc, scope="tarmac_attn", reuse=tf.compat.v1.AUTO_REUSE)

            obs_enc_p_idx = obs_enc_out_n[p_index - n_start]

            action = self.action_pred(obs_enc_p_idx, num_outputs, scope="tarmac_act", reuse=tf.compat.v1.AUTO_REUSE)

            return action, _enc_state, message_t1, attn

    def ic3net(self, x_n, num_outputs, p_index, n, n_start, n_end, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Gather inputs fed for the graph
            obs_n_in = [x_n[i] for i in range(n_start, n_end)]
            h_n_in = [x_n[i + n] for i in range(n_start, n_end)]
            c_n_in = [x_n[i + 2 * n] for i in range(n_start, n_end)]
            msg_n_in = [x_n[i + 3 * n] for i in range(n_start, n_end)]

            # Action Agent at index 0
            # Collect _encoder status from all agents
            # obs_enc_n = []
            _gating_comm_n = []
            for i in range(n_start, n_end):
                # Weight sharing between the encoder layers of all agents
                #  Communication: Gating decision is made from the hidden layer at time-step 't-1', i.e. x_n[i+n],
                if p_index is not i + n_start:
                    _gating_comm = self.ic3_gating_com(h_n_in[i], scope="ic3_gate", reuse=tf.compat.v1.AUTO_REUSE)
                    _gating_comm_n.append(_gating_comm)

            # Average all scaled hidden states from the gating mechanism, received from the other agents
            message_t1_n = self.commnet_com(_gating_comm_n, n_end - n_start - 1, scope="mean_ic3net", reuse=tf.compat.v1.AUTO_REUSE)

            # Pre - encode observation
            obs_enc_in_i = obs_n_in[p_index - n_start]
            if self.args.pre_encoder:
                obs_enc_in_i = layers.fully_connected(obs_enc_in_i, num_outputs=self.args.gru_units,
                                                    activation_fn=tf.nn.relu, scope="ic3_pre_enc", reuse=tf.compat.v1.AUTO_REUSE)

            obs_enc_in_i = tf.concat([obs_enc_in_i, message_t1_n], axis=-1)

            # Encode observation with a recurrent model
            if self.args.encoder_model == "LSTM":
                obs_enc_state_i = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c_n_in[p_index - n_start], h_n_in[p_index - n_start])
            else:  # GRU
                obs_enc_state_i = h_n_in[p_index - n_start]

            obs_enc_out_pidx, obs_enc_state_pidx = self.obs_encoder(obs_enc_in_i, obs_enc_state_i, scope="ic3net_gru_enc", reuse=tf.compat.v1.AUTO_REUSE)
            obs_enc_out_pidx = tf.squeeze(obs_enc_out_pidx, axis=-2)

            action = self.action_pred(obs_enc_out_pidx, num_outputs, scope="ic3net_act", reuse=tf.compat.v1.AUTO_REUSE)

            return action, obs_enc_state_pidx, message_t1_n, _gating_comm_n

    def commnet(self, x_n, num_outputs, p_index, n, n_start, n_end, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Gather inputs fed for the graph
            obs_n_in = [x_n[i] for i in range(n_start, n_end)]
            h_n_in = [x_n[i + n] for i in range(n_start, n_end)]
            c_n_in = [x_n[i + 2 * n] for i in range(n_start, n_end)]
            msg_n_in = [x_n[i + 3 * n] for i in range(n_start, n_end)]
            # Action Agent at index 0
            # Collect _encoder status from all agents
            _message_comm_j = []
            for i in range(n_start, n_end):
                #  Communication: Average of all hidden states is made from the hidden layer at time-step 't-1'
                if p_index is not i + n_start:
                    _message_comm_j.append(h_n_in[i])
            message_t1_n = self.commnet_com(_message_comm_j, n_end - n_start - 1, scope="mean_commnet", reuse=tf.compat.v1.AUTO_REUSE)

            # Encode observation
            obs_enc_in_pidx = obs_n_in[p_index - n_start]
            if self.args.pre_encoder:
                obs_enc_in_pidx = layers.fully_connected(obs_enc_in_pidx, num_outputs=self.args.gru_units, activation_fn=tf.nn.relu, scope="comm_pre_enc", reuse=tf.compat.v1.AUTO_REUSE)

            obs_enc_in_pidx = tf.concat([obs_enc_in_pidx, message_t1_n], axis=-1)

            # Encode observation with a recurrent model
            if self.args.encoder_model == "LSTM":
                obs_enc_state_i = (c_n_in[p_index - n_start], h_n_in[p_index - n_start])
            else:  # GRU
                obs_enc_state_i = h_n_in[p_index - n_start]

            _enc_out, _enc_state = self.obs_encoder(obs_enc_in_pidx, obs_enc_state_i, scope="commnet_enc", reuse=tf.compat.v1.AUTO_REUSE)
            _enc_out = tf.squeeze(_enc_out, axis=-2)

            action = self.action_pred(_enc_out, num_outputs, scope="comm_act", reuse=tf.compat.v1.AUTO_REUSE)

            return action, _enc_state, message_t1_n, message_t1_n

    def ddpg(self, x_n, num_outputs, p_index, n, n_start, n_end, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Gather inputs fed for the graph
            obs_n_in = [x_n[i] for i in range(n_start, n_end)]
            h_n_in = [x_n[i + n] for i in range(n_start, n_end)]
            c_n_in = [x_n[i + 2 * n] for i in range(n_start, n_end)]
            msg_n_in = [x_n[i + 3 * n] for i in range(n_start, n_end)]

            _enc_input = obs_n_in[p_index - n_start]
            if self.args.pre_encoder:
                _enc_input = layers.fully_connected(_enc_input, num_outputs=self.args.gru_units,
                                                    activation_fn=tf.nn.relu, scope="ddpg_pre_enc", reuse=tf.compat.v1.AUTO_REUSE)
            # Encode observation with a recurrent model
            if self.args.encoder_model == "LSTM":
                obs_enc_state_i = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c_n_in[p_index - n_start], h_n_in[p_index - n_start])
            else:  # GRU
                obs_enc_state_i = h_n_in[p_index - n_start]
            _enc_out, _enc_state = self.obs_encoder(_enc_input, obs_enc_state_i, scope="ddpg_enc", reuse=tf.compat.v1.AUTO_REUSE)
            _enc_out = tf.squeeze(_enc_out, axis=-2)

            action = self.action_pred(_enc_out, num_outputs, scope="ddpg_act", reuse=tf.compat.v1.AUTO_REUSE)

            return action, _enc_state, _enc_out, _enc_out
