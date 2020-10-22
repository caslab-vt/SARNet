import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

from sarnet_td3.models.SARNet_comm import RRLCell
import sarnet_td3.common.ops as ops


class CommActorNetworkTD3():
    def __init__(self, is_train, args, reuse=None):
        self.args = args
        self.is_train = is_train
        self.reuse = reuse
        if self.args.QKV_act:
            self.QKVAct = tf.nn.relu
        else:
            self.QKVAct = None

        self.attn_scale = np.sqrt(self.args.query_units)

    def _gru(self, reuse, proj=False, num_outputs=None):
        # return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=self.args.gru_units)
        # return tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=args.gru_units)
        if proj:
            gru_cell = tf.contrib.rnn.GRUCell(num_units=self.args.gru_units, reuse=reuse, name="rnn_encoder")
            tf.contrib.rnn.InputProjectionWrapper(gru_cell, self.args.gru_units, activation='relu', reuse=reuse)
            return tf.contrib.rnn.OutputProjectionWrapper(gru_cell, num_outputs, reuse=reuse)
        else:
            return tf.contrib.rnn.GRUCell(num_units=self.args.gru_units, reuse=reuse, name="rnn_encoder")

    def _lstm(self, reuse, proj=False, num_outputs=None):
        if proj:
            lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.args.gru_units, reuse=reuse, name="rnn_encoder")
            proj = tf.contrib.rnn.InputProjectionWrapper(lstm_cell, self.args.gru_units, activation=tf.nn.relu, reuse=reuse)
            return tf.contrib.rnn.OutputProjectionWrapper(proj, num_outputs, reuse=reuse)
        else:
            return tf.contrib.rnn.LSTMCell(num_units=self.args.gru_units, reuse=reuse, name="rnn_encoder")

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
                # out = tf.expand_dims(out, axis=0)
                out, state = enc(out, state)
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

    def value_pred(self, input, num_outputs, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            out = layers.fully_connected(input, num_outputs=num_outputs, activation_fn=None, scope="valueproj", reuse=reuse)
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

    def get_key_val_vec(self, x, scope, reuse, comm_type="sarnet", msg_t_i=None):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            x_val = x
            # Input [batch, dim]
            if self.args.nheads > 1 and not comm_type == "tarmac":
                key_out = []
                value_out = []
                for i in range(self.args.nheads):
                    key_out.append(layers.fully_connected(x, num_outputs=self.args.key_units, activation_fn=self.QKVAct, scope="key_encoder"+str(i), reuse=reuse))
                    value_out.append(layers.fully_connected(x_val, num_outputs=self.args.value_units, activation_fn=self.QKVAct, scope="value_encoder"+str(i), reuse=reuse))

            elif self.args.FeedMsgToValueProj and comm_type == "sarnet":
                x = tf.concat([x, msg_t_i], axis=-1)
                key_out = layers.fully_connected(x, num_outputs=self.args.key_units, activation_fn=self.QKVAct,
                                                 scope="key_encoder", reuse=reuse)
                value_out = layers.fully_connected(x_val, num_outputs=self.args.value_units, activation_fn=self.QKVAct,
                                                   scope="value_encoder", reuse=reuse)
            else:
                key_out = layers.fully_connected(x, num_outputs=self.args.key_units, activation_fn=self.QKVAct, scope="key_encoder", reuse=reuse)
                value_out = layers.fully_connected(x_val, num_outputs=self.args.value_units, activation_fn=self.QKVAct, scope="value_encoder", reuse=reuse)

        return key_out, value_out

    def compute_attn_dotprod(self, query, key_n, value_n, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            query = tf.expand_dims(query, axis=1)
            att_smry = query * key_n
            # if not self.args.sar_attn:
            att_smry = ops.inter2logits(att_smry, self.args.query_units, sumMod="SUM")  # Get dot product
            # else:
            # att_smry = ops.inter2logits(att_smry, self.args.query_units)
            att_smry = tf.nn.softmax(att_smry / self.attn_scale)   # (batch_size, #Agents)
            # Now do interaction with the value (batch_size, #Agents) * (batch_size, #Agents, value_dim)
            message = tf.einsum('bn,bnd->bd', att_smry, value_n)
            # _obs_act_in = tf.contrib.layers.batch_norm(message, decay=self.args.bnDecay, center=self.args.bnCenter,
            #                                            scale=self.args.bnScale, is_training=self.is_train,
            #                                            updates_collections=None, scope="batch_norm_obs", reuse=reuse)
            return message, att_smry

    def commnet_com(self, x_n, n, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            message = tf.add_n(x_n) / n
            return message

    def ic3_gating_com(self, state_t1, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            _gate = layers.fully_connected(state_t1, num_outputs=1, activation_fn=tf.nn.softmax, scope="gate_encoder", reuse=reuse)
            _comm_gate_out = tf.multiply(state_t1, _gate)
            # Last minute change
            # return _comm_gate_out
            return _comm_gate_out, _gate

    """ Communication Architectures: CommNet, SARNet, TarMAC, IC3Net"""
    # Input is defined as the following:
    # x_n: {Indexes: [0 ~ N-1] - Observation; [N ~ 2*N-1] - GRU State; [2*N ~ 3*N-1] -
    # GRU State 2; [3*N ~ 4*N-1] - Message_tp}

    def sarnet(self, x_n, num_outputs, p_index, n, n_start, n_end, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Gather inputs fed for the graph
            # Required as a list of [#agents, [time, batch, dim]]
            obs_n_in = [x_n[i] for i in range(n)]
            # Single time step only with for the initial state or message
            h_n_in = [x_n[i + n] for i in range(n)]
            c_n_in = [x_n[i + 2 * n] for i in range(n)]
            msg_n_in = [x_n[i + 3 * n] for i in range(n)]
            # attn_pidx = tf.zeros(shape=[None, self.args.value_units])
            outputs_act = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            outputs_attn = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            # Control the size of the loop with the time dimension of the observation space
            time_steps = tf.shape(obs_n_in[p_index])[0]
            def condition(stp, *args):
                return stp < time_steps

            def body(stp, h_n_t, c_n_t, msg_n_t, outputs_attn_pidx, outputs_):
                # Collect _encoder status from all agents
                obs_enc_out_n_t = []
                obs_enc_state_n_t1 = []
                _key_enc = []
                _value_enc = []
                _query_enc = []

                """Internal Computation from loop inputs - hidden_state"""
                # Aggregate hidden states from other agents, compute message, and then use it to
                # augment observation for encoder
                if self.args.SARplusIC3:
                    saric3_comm_n = []
                    for i in range(n_start, n_end):
                        _gating_comm = self.ic3_gating_com(h_n_t[i], scope="sarplusic3_gate", reuse=tf.compat.v1.AUTO_REUSE)
                        saric3_comm_n.append(_gating_comm)

                # Weight sharing between the encoder layers of all agents
                for i in range(n_start, n_end):
                    # Encode observation
                    # Pre encode
                    obs_enc_in_i = obs_n_in[i][stp]
                    """Computation from observation state, fed directly step-wise"""
                    if self.args.pre_encoder:
                        obs_enc_in_i = layers.fully_connected(obs_enc_in_i, num_outputs=self.args.gru_units,
                                                              activation_fn=tf.nn.relu, scope="sarnet_pre_enc",
                                                              reuse=tf.compat.v1.AUTO_REUSE)
                    """Computation from observation state, fed directly step-wise"""
                    if self.args.FeedOldMemoryToObsEnc:
                        obs_enc_in_i = tf.concat([msg_n_t[i], obs_enc_in_i], axis=-1)
                    """Computation from observation state, fed directly step-wise"""
                    if self.args.SARplusIC3:
                        # Get all gating actions for all agents except "i" to compute mean
                        saric3_comm_j = saric3_comm_n[:i] + saric3_comm_n[i + 1:]
                        # Compute messages mean of pre-scaled hidden states
                        ic3_msg_n_t = self.commnet_com(saric3_comm_j, n_end - n_start - 1, scope="sarplusic3_mean", reuse=tf.compat.v1.AUTO_REUSE)
                        obs_enc_in_i = tf.concat([obs_enc_in_i, ic3_msg_n_t], axis=-1)

                    """Internal Computation from loop inputs - hidden_state"""
                    # Encode observation with a recurrent model
                    if self.args.encoder_model == "LSTM":
                        obs_enc_state_i = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c_n_t[i], h_n_t[i])
                    else:  # GRU
                        obs_enc_state_i = h_n_t[i]

                    _enc_out, _enc_state = self.obs_encoder(obs_enc_in_i, obs_enc_state_i, scope="sarnet_enc", reuse=tf.compat.v1.AUTO_REUSE)

                    """
                    Update the encoder states for each agent for the next time-step
                    """
                    if self.args.encoder_model == "LSTM":
                        c_n_t[i], h_n_t[i] = _enc_state
                    else:
                        h_n_t[i] = _enc_state

                    # Populate encoder output for recurrent cases
                    # _enc_out = tf.squeeze(_enc_out, axis=0)
                    obs_enc_out_n_t.append(_enc_out)
                    obs_enc_state_n_t1.append(_enc_state)

                    _key_proj, _value_proj = self.get_key_val_vec(_enc_out, scope="sarnet_KV_Projs", reuse=tf.compat.v1.AUTO_REUSE, msg_t_i=msg_n_t[i])
                    _key_enc.append(_key_proj)
                    _value_enc.append(_value_proj)
                    _query_proj = self.get_query_vec(_enc_out, scope="sarnet_QProj", reuse=tf.compat.v1.AUTO_REUSE)
                    _query_enc.append(_query_proj)

                # Now generate messages for each agent
                for i in range(n_start, n_end):
                    # Reasoning Cell
                    self.sarnet_cell = self._sarnet_comm(reuse=tf.compat.v1.AUTO_REUSE)
                    sarnet_input = (_query_enc[i], _key_enc, _value_enc)
                    attn_i, msg_n_t[i] = self.sarnet_cell(sarnet_input, msg_n_t[i])
                    if i == (p_index - n_start):
                        attn_pidx = attn_i
                # Merge new reasoned message and observation encoding
                _obs_act_in = obs_enc_out_n_t[p_index - n_start]

                if self.args.bNorm_state:
                    _obs_act_in = tf.contrib.layers.batch_norm(_obs_act_in, decay=self.args.bnDecay,
                                                               center=self.args.bnCenter,
                                                               scale=self.args.bnScale, is_training=self.is_train,
                                                               updates_collections=None, scope="batch_norm_obs",
                                                               reuse=tf.compat.v1.AUTO_REUSE)

                act_in_pred = tf.concat([_obs_act_in, msg_n_t[p_index]], axis=-1)
                if self.args.TwoLayerEncodeSarnet:
                    act_in_pred = self.action_enc(act_in_pred, scope="sarnet_act_enc", reuse=tf.compat.v1.AUTO_REUSE)
                action = self.action_pred(act_in_pred, num_outputs, scope="sarnet_act", reuse=tf.compat.v1.AUTO_REUSE)
                outputs_attn_pidx = outputs_attn_pidx.write(stp, attn_pidx)
                outputs_ = outputs_.write(stp, action)

                return stp + 1, h_n_t, c_n_t, msg_n_t, outputs_attn_pidx, outputs_

            _, h_n_out, c_n_out, msg_n_out, attn_pidx, out = tf.while_loop(condition, body, loop_vars=[tf.Variable(0), h_n_in, c_n_in, msg_n_in, outputs_attn, outputs_act], parallel_iterations=1)

            out = out.stack()  # This will throw out a [time, out_dim_list]
            attn_pidx = attn_pidx.stack()
            attn_pidx = attn_pidx[-1]  # Get just the last time step for the attention

            action_traj = out
            msg_pidx_out = msg_n_out[p_index]

        return action_traj, (h_n_out[p_index], c_n_out[p_index]), msg_pidx_out, attn_pidx

    def tarmac(self, x_n, num_outputs, p_index, n, n_start, n_end, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Gather inputs fed for the graph
            # Required as a list of [#agents, [time, batch, dim]]
            obs_n_in = [x_n[i] for i in range(n)]

            # Single time step only with for the initial state or message
            h_n_in = [x_n[i + n] for i in range(n)]
            c_n_in = [x_n[i + 2 * n] for i in range(n)]
            msg_n_in = [x_n[i + 3 * n] for i in range(n)]

            outputs_ = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            outputs_attn = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            # Control the size of the loop with the time dimension of the observation space
            time_steps = tf.shape(obs_n_in[p_index])[0]

            def condition(stp, *args):
                return stp < time_steps

            def body(stp, h_n_t, c_n_t, msg_n_t, outputs_attn_pidx, outputs_):
                # Collect _encoder status from all agents
                obs_enc_out_n = []
                _key_enc = []
                _value_enc = []
                _query_enc = []

                # Start from first agent of one type to the last. (e.g. 0 - num_adv-1) or (num_adv to n)
                for i in range(n_start, n_end):
                    # Encode observation
                    obs_enc_in_i = obs_n_in[i][stp]
                    if self.args.pre_encoder:
                        obs_enc_in_i = layers.fully_connected(obs_enc_in_i, num_outputs=self.args.gru_units, activation_fn=tf.nn.relu, scope="tarmac_pre_enc", reuse=tf.compat.v1.AUTO_REUSE)

                    obs_enc_in_i = tf.concat([obs_enc_in_i, msg_n_t[i]], axis=-1)  # Concat message and observation encoding

                    # Encode observation with a recurrent model
                    if self.args.encoder_model == "LSTM":
                        obs_enc_state_i = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c_n_t[i], h_n_t[i])
                    else:  # GRU
                        obs_enc_state_i = h_n_t[i]

                    _enc_out, _enc_state = self.obs_encoder(obs_enc_in_i, obs_enc_state_i, scope="tarmac_enc", reuse=tf.compat.v1.AUTO_REUSE)
                    obs_enc_out_n.append(_enc_out)

                    if self.args.encoder_model == "LSTM":
                        c_n_t[i], h_n_t[i] = _enc_state
                    else:
                        h_n_t[i] = _enc_state

                    # Generate Keys and Values from all agents
                    _key_proj, _value_proj = self.get_key_val_vec(_enc_out, scope="tarmac_keyval", reuse=tf.compat.v1.AUTO_REUSE, comm_type = "tarmac")
                    _key_enc.append(_key_proj)
                    _value_enc.append(_value_proj)

                    # Generate Query only from action agent
                    _query_proj = self.get_query_vec(_enc_out, scope="tarmac_query", reuse=tf.compat.v1.AUTO_REUSE, comm_type="tarmac")
                    _query_enc.append(_query_proj)

                # Stack all keys and values of all agents to a single tensor
                _value_enc = tf.stack(_value_enc, axis=-2)
                _key_enc = tf.stack(_key_enc, axis=-2)

                # Updates messages for every agent for the next timestep
                for i in range(n_start, n_end):
                    # Generate attention and message, attention only used for benchmarking results
                    msg_n_t[i], attn_i = self.compute_attn_dotprod(_query_enc[i], _key_enc, _value_enc, scope="tarmac_attn", reuse=tf.compat.v1.AUTO_REUSE)
                    if i == (p_index - n_start):
                        attn_pidx = attn_i

                action = self.action_pred(obs_enc_out_n[p_index - n_start], num_outputs, scope="tarmac_act", reuse=tf.compat.v1.AUTO_REUSE)

                outputs_ = outputs_.write(stp, action)
                outputs_attn_pidx = outputs_attn_pidx.write(stp, attn_pidx)

                return stp + 1, h_n_t, c_n_t, msg_n_t, outputs_attn_pidx, outputs_

            # Run the While Loop
            _, h_n_out, c_n_out, msg_n_out,  attn_pidx, out = tf.while_loop(condition, body, loop_vars=[tf.Variable(0), h_n_in, c_n_in, msg_n_in, outputs_attn, outputs_], parallel_iterations=1)
            out = out.stack()  # This will throw out a [time, out_dim_list]
            attn_pidx = attn_pidx.stack()
            attn_pidx = attn_pidx[-1]  # Get just the last time step for the attention

            action_traj = out
            msg_pidx_out = msg_n_out[p_index]

            return action_traj, (h_n_out[p_index], c_n_out[p_index]), msg_pidx_out, attn_pidx

    def ic3net(self, x_n, num_outputs, p_index, n, n_start, n_end, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Gather inputs fed for the graph
            # Required as a list of [#agents, [time, batch, dim]]
            obs_n_in = [x_n[i] for i in range(n)]

            # Single time step only with for the initial state or message
            h_n_in = [x_n[i + n] for i in range(n)]
            c_n_in = [x_n[i + 2 * n] for i in range(n)]
            msg_n_in = [x_n[i + 3 * n] for i in range(n)]

            outputs_ = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
            outputs_attn = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            # Control the size of the loop with the time dimension of the observation space
            time_steps = tf.shape(obs_n_in[p_index])[0]

            def condition(stp, *args):
                return stp < time_steps

            def body(stp, h_n_t, c_n_t, msg_n_t, outputs_attn_pidx, outputs_):
                _gating_comm_n = []
                message_t1_n = []
                obs_enc_out_n = []
                for i in range(n_start, n_end):
                    # Weight sharing between the encoder layers of all agents
                    #  Communication: Gating decision is made from the hidden layer at time-step 't-1', i.e. x_n[i+n],
                    # Last minute change
                    # _gating_comm = self.ic3_gating_com(h_n_t[i], scope="ic3_gate", reuse=tf.compat.v1.AUTO_REUSE)
                    _gating_comm, _gate = self.ic3_gating_com(h_n_t[i], scope="ic3_gate", reuse=tf.compat.v1.AUTO_REUSE)
                    _gating_comm_n.append(_gating_comm)
                    if i == (p_index - n_start):
                        attn_pidx = _gate

                for i in range(n_start, n_end):
                    # Average all scaled hidden states from the gating mechanism, received from the other agents
                    message_t1_n.append(self.commnet_com(_gating_comm_n[:i] + _gating_comm_n[i+1:], n_end - n_start - 1, scope="mean_ic3net", reuse=tf.compat.v1.AUTO_REUSE))

                    # Pre - encode observation
                    obs_enc_in_i = obs_n_in[i][stp]
                    if self.args.pre_encoder:
                        obs_enc_in_i = layers.fully_connected(obs_enc_in_i, num_outputs=self.args.gru_units,
                                                              activation_fn=tf.nn.relu, scope="ic3_pre_enc",
                                                              reuse=tf.compat.v1.AUTO_REUSE)

                    obs_enc_in_i = tf.concat([obs_enc_in_i, message_t1_n[i]], axis=-1)

                    # Encode observation with a recurrent model
                    if self.args.encoder_model == "LSTM":
                        obs_enc_state_i = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c_n_t[i], h_n_t[i])
                    else:  # GRU
                        obs_enc_state_i = h_n_t[i]

                    _enc_out_i, _enc_state_i = self.obs_encoder(obs_enc_in_i, obs_enc_state_i,
                                                                            scope="ic3net_gru_enc",
                                                                            reuse=tf.compat.v1.AUTO_REUSE)
                    obs_enc_out_n.append(_enc_out_i)

                    if self.args.encoder_model == "LSTM":
                        c_n_t[i], h_n_t[i] = _enc_state_i
                    else:
                        h_n_t[i] = _enc_state_i

                action = self.action_pred(obs_enc_out_n[p_index - n_start], num_outputs, scope="tarmac_act", reuse=tf.compat.v1.AUTO_REUSE)

                outputs_ = outputs_.write(stp, action)
                outputs_attn_pidx = outputs_attn_pidx.write(stp, attn_pidx)

                return stp + 1, h_n_t, c_n_t, msg_n_t, outputs_attn_pidx, outputs_

            # Run the While Loop
            _, h_n_out, c_n_out, msg_n_out, attn_pidx, out = tf.while_loop(condition, body, loop_vars=[tf.Variable(0), h_n_in, c_n_in, msg_n_in, outputs_attn, outputs_], parallel_iterations=1)
            out = out.stack()  # This will throw out a [time, out_dim_list]
            attn_pidx = attn_pidx.stack()
            attn_pidx = attn_pidx[-1]  # Get just the last time step for the attention

            action_traj = out
            msg_pidx_out = msg_n_out[p_index]

            return action_traj, (h_n_out[p_index], c_n_out[p_index]), msg_pidx_out, attn_pidx

    def commnet(self, x_n, num_outputs, p_index, n, n_start, n_end, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Gather inputs fed for the graph
            # Required as a list of [#agents, [time, batch, dim]]
            obs_n_in = [x_n[i] for i in range(n)]

            # Single time step only with for the initial state or message
            h_n_in = [x_n[i + n] for i in range(n)]
            c_n_in = [x_n[i + 2 * n] for i in range(n)]
            msg_n_in = [x_n[i + 3 * n] for i in range(n)]

            outputs_ = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

            # Control the size of the loop with the time dimension of the observation space
            time_steps = tf.shape(obs_n_in[p_index])[0]

            def condition(stp, *args):
                return stp < time_steps

            def body(stp, h_n_t, c_n_t, msg_n_t, outputs_):
                # Collect all messages and get a mean of it
                obs_enc_out_n = []
                for i in range(n_start, n_end):
                    message_t1_i = self.commnet_com(h_n_t[:i] + h_n_t[i+1:], n_end - n_start - 1, scope="mean_commnet", reuse=tf.compat.v1.AUTO_REUSE)

                    obs_enc_in_i = obs_n_in[i][stp]
                    if self.args.pre_encoder:
                        obs_enc_in_i = layers.fully_connected(obs_enc_in_i, num_outputs=self.args.gru_units,
                                                                 activation_fn=tf.nn.relu, scope="comm_pre_enc",
                                                                 reuse=tf.compat.v1.AUTO_REUSE)
                    obs_enc_in_pidx = tf.concat([obs_enc_in_i, message_t1_i], axis=-1)

                    # Encode observation with a recurrent model
                    if self.args.encoder_model == "LSTM":
                        obs_enc_state_i = (c_n_t[i], h_n_t[i])
                    else:  # GRU
                        obs_enc_state_i = h_n_t[i]

                    enc_out_i, enc_state_i = self.obs_encoder(obs_enc_in_pidx, obs_enc_state_i, scope="commnet_enc", reuse=tf.compat.v1.AUTO_REUSE)
                    if self.args.encoder_model == "LSTM":
                        c_n_t[i], h_n_t[i] = enc_state_i
                    else:
                        h_n_t[i] = enc_state_i

                    obs_enc_out_n.append(enc_out_i)

                action = self.action_pred(obs_enc_out_n[p_index - n_start], num_outputs, scope="tarmac_act", reuse=tf.compat.v1.AUTO_REUSE)

                outputs_ = outputs_.write(stp, action)

                return stp + 1, h_n_t, c_n_t, msg_n_t, outputs_

            # Run the While Loop
            _, h_n_out, c_n_out, msg_n_out, out = tf.while_loop(condition, body, loop_vars=[tf.Variable(0), h_n_in, c_n_in, msg_n_in, outputs_], parallel_iterations=1)
            out = out.stack()  # This will throw out a [time, out_dim_list]

            action_traj = out
            msg_pidx_out = msg_n_out[p_index]

            return action_traj, (h_n_out[p_index], c_n_out[p_index]), msg_pidx_out, msg_pidx_out

    def ddpg(self, x_n, num_outputs, p_index, n, n_start, n_end, scope, reuse):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Gather inputs fed for the graph
            obs_n_in = [x_n[i] for i in range(n)]  # Shape is [nagents, [time, batch, dim]]
            h_n_in = [x_n[i + n] for i in range(n)]
            c_n_in = [x_n[i + 2 * n] for i in range(n)]
            msg_n_in = [x_n[i + 3 * n] for i in range(n)]

            _enc_input = obs_n_in[p_index]
            # if self.args.pre_encoder:
            #     _enc_input = layers.fully_connected(_enc_input, num_outputs=self.args.gru_units,
            #                                         activation_fn=tf.nn.relu, scope="ddpg_pre_enc", reuse=tf.compat.v1.AUTO_REUSE)
            # Encode observation with a recurrent model
            if self.args.encoder_model == "LSTM":
                obs_enc_state_i = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c_n_in[p_index], h_n_in[p_index])
            else:  # GRU
                obs_enc_state_i = h_n_in[p_index]

            with tf.compat.v1.variable_scope("ddpg_enc", reuse=tf.compat.v1.AUTO_REUSE):
                if self.args.recurrent:
                    if self.args.encoder_model == "GRU":
                        enc = self._gru(tf.compat.v1.AUTO_REUSE, proj=True, num_outputs=num_outputs)
                    elif self.args.encoder_model == "LSTM":
                        enc = self._lstm(tf.compat.v1.AUTO_REUSE, proj=True, num_outputs=num_outputs)

            action_traj, _enc_state = tf.compat.v1.nn.dynamic_rnn(enc, _enc_input, initial_state=obs_enc_state_i, time_major=True, scope="rnn_encoder")

            return action_traj, _enc_state, msg_n_in[p_index], msg_n_in[p_index]
