import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import sarnet_td3.common.ops as ops

def mlp_model(input, num_agents, args, scope, reuse=False, p_index=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        input = [input[i] for i in range(int(2 * num_agents))]
        out = tf.concat(input, axis=1)
        out = layers.fully_connected(out, num_outputs=args.critic_units, activation_fn=tf.nn.relu, scope="cmlp0", reuse=reuse)
        out = layers.fully_connected(out, num_outputs=int(args.critic_units/2), activation_fn=tf.nn.relu, scope="cmlp1", reuse=reuse)
        out = layers.fully_connected(out, num_outputs=1, activation_fn=None, scope="cmlp2", reuse=reuse)

        return out, out


def _gru(args, reuse, maac=False):
    # return tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(num_units=args.critic_units)
    # return tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=args.gru_units)
    if not maac:
        gru_cell = tf.contrib.rnn.GRUCell(num_units=args.critic_units, reuse=reuse, name="critic_encoder")
        proj = tf.contrib.rnn.InputProjectionWrapper(gru_cell, args.critic_units, reuse=reuse, activation=tf.nn.relu)
        proj = tf.contrib.rnn.OutputProjectionWrapper(proj, 1, reuse=reuse)
    else:
        gru_cell = tf.contrib.rnn.GRUCell(num_units=args.critic_units, reuse=reuse, name="critic_encoder")
        proj = gru_cell
    return proj

# Embedding Network common to all agents
"""
Args
Input: 
Encoder state if a recurrent network - [batch, time, gru_dim]
Obs dim - [batch, time, obs_dim]
"""

def rnn_model(input, num_agents, args, scope, reuse=False, p_index=None):
    # input = [obs, act], state=GRU else None
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        x_n = [input[i] for i in range(int(2 * num_agents))]
        state = input[int(2 * num_agents)]
        out = tf.concat(x_n, axis=-1)  # Concatenate the action/obs space for rnn encode
        # if args.pre_encoder:
        #     out = layers.fully_connected(x, num_outputs=args.critic_units, activation_fn=tf.nn.relu, scope="cmlp0", reuse=reuse)
        gru = _gru(args, reuse=reuse)
        # out = tf.expand_dims(out, axis=-2) # Changed from -2 to 0, to account for CudnnRNN cells (time-major only)
        out, state = tf.nn.dynamic_rnn(gru, out, initial_state=state, time_major=True, scope="cmlp_rnn")
        # timesteps = tf.shape(state)[0]
        # state_list_time = tf.map_fn(fn=lambda k: state[k], elems=tf.range(timesteps), dtype=tf.float32)
        #
        # condition = lambda stp, x: stp < timesteps
        #
        #
        # out = layers.fully_connected(state, num_outputs=1, activation_fn=None, scope="cmlp1", reuse=reuse)
        return out, state

# Computes n-head attention for critic - MAAC
def maac_attn_nhead(x, p_index, num_agents, args, scope, reuse=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        message_out = []
        for i in range(args.nheads):
            key_out = []
            value_out = []
            for a in range(num_agents):
                # print(" Agent" + str(a))
                if a is p_index:
                    query_out = layers.fully_connected(x[a], num_outputs=args.query_units, activation_fn=None,
                                                        scope="query_encoder" + str(i) + str(a), reuse=tf.compat.v1.AUTO_REUSE)
                else:
                    key_out.append(layers.fully_connected(x[a], num_outputs=args.key_units, activation_fn=None,
                                                          scope="key_encoder" + str(i) + str(a), reuse=tf.compat.v1.AUTO_REUSE))
                    value_out.append(layers.fully_connected(x[a], num_outputs=args.value_units, activation_fn=None,
                                                            scope="value_encoder" + str(i) + str(a), reuse=tf.compat.v1.AUTO_REUSE))

            query = tf.expand_dims(query_out, axis=1)
            key_out = tf.stack(key_out, axis=-2)
            att_smry = query * key_out
            att_smry = ops.inter2logits(att_smry, args.query_units, sumMod="SUM")  # Get dot product
            att_smry = tf.nn.softmax(att_smry / np.sqrt(args.query_units))  # (batch_size, #Agents)
            # Now do interaction with the value (batch_size, #Agents) * (batch_size, #Agents, value_dim)
            value_out = tf.stack(value_out, axis=-2)
            message = tf.einsum('bn,bnd->bd', att_smry, value_out)
            message_out.append(message)
        message_out = tf.concat(message_out, axis=-1)

        return message_out


def maac_rnn_model(x_n, n, args, scope, reuse=False, p_index=None):
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Gather inputs fed for the graph
        # Required as a list of [#agents, [time, batch, dim]]
        obs_n_in = [x_n[i] for i in range(n)]
        act_n_in = [x_n[i+n] for i in range(n)]
        state = x_n[int(2 * n)]

        outputs_ = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, name="QProjs")

        # Control the size of the loop with the time dimension of the observation space
        time_steps = tf.shape(obs_n_in[p_index])[0]
        def condition(stp, *args):
            return stp < time_steps

        def body(stp, qh_i_t, outputs_):
            # Collect _encoder status from all agents
            obs_enc_out_n = []
            _key_enc = []
            _value_enc = []
            _query_enc = []

            # Start from first agent of one type to the last. (e.g. 0 - num_adv-1) or (num_adv to n)
            for i in range(n):
                # Encode observation
                enc_in_i = tf.concat([obs_n_in[i][stp], act_n_in[i][stp]], axis=-1)
                obs_enc_out_n.append(layers.fully_connected(enc_in_i, num_outputs=args.gru_units,
                                                          activation_fn=tf.nn.relu, scope="maac_pre_enc" + str(i),
                                                          reuse=tf.compat.v1.AUTO_REUSE))

            # Calculate attention from n-heads
            critic_message = maac_attn_nhead(obs_enc_out_n, p_index, n, args, scope="maac_attn", reuse=tf.compat.v1.AUTO_REUSE)

            obs_enc_in_i = tf.concat([obs_enc_out_n[p_index], critic_message], axis=-1)
            enc = _gru(args, reuse=tf.compat.v1.AUTO_REUSE, maac=True)
            _enc_out_i, qh_i_t = enc(obs_enc_in_i, qh_i_t)

            # project to Q-value
            q = layers.fully_connected(_enc_out_i, num_outputs=1, activation_fn=None, scope="Qproj", reuse=tf.compat.v1.AUTO_REUSE)
            # print(stp)
            outputs_ = outputs_.write(stp, q)

            return stp + 1, qh_i_t, outputs_

        # Run the While Loop
        _, h_n_out, out = tf.while_loop(condition, body, loop_vars=[tf.Variable(0, dtype=tf.int32), state, outputs_], parallel_iterations=1)
        out = out.stack()  # This will throw out a [time, batch, out_dim_list]

    return out, h_n_out

#<tf.Variable 'MAAC-SP10/DDPG_ADV/adv_agent/target_q_func01/Variable:0' shape=() dtype=int32_ref>
#<tf.Variable 'MAAC-SP10/DDPG_ADV/adv_agent/q_func01/Variable:0' shape=() dtype=int32_ref>
