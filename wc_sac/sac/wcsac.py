#Portions of the code are adapted from Safety Starter Agents and Spinning Up, released by OpenAI under the MIT license.
#!/usr/bin/env python

from functools import partial
import numpy as np
import tensorflow as tf
if tf.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf_v1
    tf_v1.disable_eager_execution()
    tf = tf_v1
import time
from wc_sac.utils.logx import EpochLogger
from wc_sac.utils.mpi_tf import sync_all_params, MpiAdamOptimizer
from wc_sac.utils.mpi_tools import mpi_fork, mpi_sum, proc_id, mpi_statistics_scalar, num_procs


from gym.envs.registration import register
from scipy.stats import norm

EPS = 1e-8

'''
config1 = {
        'placements_extents': [-1.5, -1.5, 1.5, 1.5],
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'goal_size': 0.3,
        'goal_keepout': 0.305,
        'goal_locations': [(1.1, 1.1)],
        'observe_goal_lidar': True,
        'observe_hazards': True,
        'constrain_hazards': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': 1,
        'hazards_size': 0.7,
        'hazards_keepout': 0.705,
        'hazards_locations': [(0, 0)]
        }


register(id='StaticEnv-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config1})

config2 = {
        'placements_extents': [-1.5, -1.5, 1.5, 1.5],
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'goal_size': 0.3,
        'goal_keepout': 0.305,
        'observe_goal_lidar': True,
        'observe_hazards': True,
        'constrain_hazards': True,
        'lidar_max_dist': 3,
        'lidar_num_bins': 16,
        'hazards_num': 3,
        'hazards_size': 0.3,
        'hazards_keepout': 0.305
        }

register(id='DynamicEnv-v0',
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': config2})
'''

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(64,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def cvar_from_mean_var(cost_mean, cost_var, alpha):
    """
    由抢占率（cost）的均值、方差闭式计算高斯分布下的CVaR
    核心公式：CVaR_α(ρ) = μ_ρ + (σ_ρ · φ(Φ⁻¹(1-α))) / α
    其中：ρ~N(μ_ρ, σ_ρ²)，φ为标准正态PDF，Φ⁻¹为标准正态分位数逆函数

    参数：
        cost_mean: 抢占率均值 E[ρ]，形状 (batch,)
        cost_var : 抢占率方差 Var(ρ)，形状 (batch,)
        alpha: CVaR显著性水平α，也即尾部比例（如0.1/0.5/0.9，代表关注最坏α比例场景，α越小风险厌恶程度越高）
    返回：
        cvar: 每个样本的CVaR值，形状 (batch,)，与输入维度完全一致
    """
    # 转换为TensorFlow张量并指定浮点类型，保证计算图兼容性
    cost_mean = tf.convert_to_tensor(cost_mean, dtype=tf.float32)
    cost_var = tf.convert_to_tensor(cost_var, dtype=tf.float32)
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float32)  # 显著性水平转为张量，支持批量/标量

    # 数值稳定性处理：方差非负（避免数值误差导致负方差），标准差开方
    cost_var = tf.maximum(cost_var, 1e-8)  # 加极小值避免开方为0/负数
    cost_std = tf.sqrt(cost_var)  # 抢占率标准差 σ_ρ

    # 步骤1：计算标准正态分布的(1-α)分位数逆函数 Φ⁻¹(1-α)
    # tf.math.erfinv是逆误差函数，与标准正态分位数的转换关系：Φ⁻¹(x) = √2 · erfinv(2x-1)
    cl = 1 - alpha  # 对应1-α分位，置信水平
    inv_phi = tf.math.sqrt(2.0) * tf.math.erfinv(2.0 * cl - 1.0)

    # 步骤2：计算标准正态分布在inv_phi处的概率密度函数 φ(Φ⁻¹(1-α))
    # 标准正态PDF公式：φ(x) = (1/√(2π)) · exp(-x²/2)
    phi = tf.math.exp(-0.5 * tf.square(inv_phi)) / tf.math.sqrt(2.0 * np.pi)

    # 步骤3：闭式计算CVaR（核心公式）
    # 上尾风险溢价项：(σ_ρ · φ) / α ，叠加均值得到最终CVaR
    risk_premium = (cost_std * phi) / alpha
    cvar = cost_mean + risk_premium

    return cvar

def get_target_update(main_name, target_name, polyak):
    ''' Get a tensorflow op to update target variables based on main variables '''
    main_vars = {x.name: x for x in get_vars(main_name)}
    targ_vars = {x.name: x for x in get_vars(target_name)}
    assign_ops = []
    for v_targ in targ_vars:
        assert v_targ.startswith(target_name), f'bad var name {v_targ} for {target_name}'
        v_main = v_targ.replace(target_name, main_name, 1)
        assert v_main in main_vars, f'missing var name {v_main}'
        assign_op = tf.assign(targ_vars[v_targ], polyak*targ_vars[v_targ] + (1-polyak)*main_vars[v_main])
        assign_ops.append(assign_op)
    return tf.group(assign_ops)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=None)
    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    # Adjustment to log prob
    '''
    '''
    logp_pi -= tf.reduce_sum(2*(np.log(2) - pi - tf.nn.softplus(-2*pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


"""
Actors and Critics
"""
def mlp_actor(x, a, name='pi', hidden_sizes=(64,64), activation=tf.nn.relu,
              output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope(name):
        mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
    action_scale = action_space.high[0]
    mu *= action_scale
    pi *= action_scale

    return mu, pi, logp_pi

def mlp_var(x, a, pi, name, hidden_sizes=(64,64), activation=tf.nn.relu,
              output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    
    fn_mlp = lambda x : tf.squeeze(mlp(x=x,
                                       hidden_sizes=list(hidden_sizes)+[1],
                                       activation=activation,
                                       output_activation=None),
                                   axis=1)
    
    with tf.variable_scope(name):
        var = fn_mlp(tf.concat([x,a], axis=-1))
        var = tf.nn.softplus(var)

    with tf.variable_scope(name, reuse=True):
        var_pi = fn_mlp(tf.concat([x,pi], axis=-1))
        var_pi = tf.nn.softplus(var_pi)

    return var, var_pi


def mlp_critic(x, a, pi, name, hidden_sizes=(64,64), activation=tf.nn.relu,
               output_activation=None, policy=mlp_gaussian_policy, action_space=None):

    fn_mlp = lambda x : tf.squeeze(mlp(x=x,
                                       hidden_sizes=list(hidden_sizes)+[1],
                                       activation=activation,
                                       output_activation=None),
                                   axis=1)
    with tf.variable_scope(name):
        critic = fn_mlp(tf.concat([x,a], axis=-1))

    with tf.variable_scope(name, reuse=True):
        critic_pi = fn_mlp(tf.concat([x,pi], axis=-1))

    return critic, critic_pi


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.costs_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, cost):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.costs_buf[self.ptr] = cost
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    costs=self.costs_buf[idxs],
                    done=self.done_buf[idxs])


"""
Soft Actor-Critic
"""
def sac(env_fn, actor_fn=mlp_actor, critic_fn=mlp_critic, var_fn=mlp_var, ac_kwargs=dict(), seed=0,
        steps_per_epoch=1000, epochs=100, replay_size=int(1e6), gamma=0.99, alpha_sig_level=0.5,
        polyak=0.995, lr=1e-4, batch_size=1024, local_start_steps=int(1e3),
        max_ep_len=1000, logger_kwargs=dict(), save_freq=10, local_update_after=int(1e3),
        update_freq=1, render=False, 
        fixed_entropy_bonus=None, entropy_constraint=-1.0,
        fixed_cost_penalty=None, cost_lim=None,
        reward_scale=1, lr_scale = 1, damp_scale = 0,
        ):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_fn: A function which takes in placeholder symbols
            for state, ``x_ph``, and action, ``a_ph``, and returns the actor
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ===========  ================  ======================================

        critic_fn: A function which takes in placeholder symbols
            for state, ``x_ph``, action, ``a_ph``, and policy ``pi``,
            and returns the critic outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``critic``    (batch,)         | Gives one estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``critic_pi`` (batch,)         | Gives another estimate of Q* for
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_fn / critic_fn
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        batch_size (int): Minibatch size for SGD.

        local_start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        alpha_sig_level (float): CVaR significance level (tail proportion).
            Represents the worst-case α proportion of scenarios to focus on.
            Examples: 0.1 (focus on worst 10%), 0.5 (worst 50%), 0.9 (worst 90%).
            Smaller values indicate higher risk aversion.

        fixed_entropy_bonus (float or None): Fixed bonus to reward for entropy.
            Units are (points of discounted sum of future reward) / (nats of policy entropy).
            If None, use ``entropy_constraint`` to set bonus value instead.

        entropy_constraint (float): If ``fixed_entropy_bonus`` is None,
            Adjust entropy bonus to maintain at least this much entropy.
            Actual constraint value is multiplied by the dimensions of the action space.
            Units are (nats of policy entropy) / (action dimenson).

        fixed_cost_penalty (float or None): Fixed penalty to reward for cost.
            Units are (points of discounted sum of future reward) / (points of discounted sum of future costs).
            If None, use ``cost_lim`` to set penalty value instead (beta will be learned).

        cost_lim (float or None): CVaR constraint threshold.
            Units are (expectation of undiscounted sum of costs in a single episode).
            If None and fixed_cost_penalty is None, no cost constraints are used (naive optimization).
    """
    # 成本/风险约束开关：
    # - fixed_cost_penalty: 固定惩罚系数 beta
    # - cost_lim: CVaR 约束阈值（由你在命令行传入）
    use_costs = (fixed_cost_penalty is not None) or (cost_lim is not None)

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Env instantiation    
    env, test_env = env_fn(), env_fn()
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Setting seeds
    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    test_env.seed(seed)

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph, c_ph = placeholders(obs_dim, act_dim, obs_dim, None, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi = actor_fn(x_ph, a_ph, **ac_kwargs)
        qr1, qr1_pi = critic_fn(x_ph, a_ph, pi, name='qr1', **ac_kwargs)
        qr2, qr2_pi = critic_fn(x_ph, a_ph, pi, name='qr2', **ac_kwargs)
        qc,  qc_pi  = critic_fn(x_ph, a_ph, pi, name='qc', **ac_kwargs)
        qc_var,  qc_pi_var  = var_fn(x_ph, a_ph, pi, name='qc_var', **ac_kwargs)

    with tf.variable_scope('main', reuse=True):
        # Additional policy output from a different observation placeholder
        # This lets us do separate optimization updates (actor, critics, etc)
        # in a single tensorflow op.
        _, pi2, logp_pi2 = actor_fn(x2_ph, a_ph, **ac_kwargs)

    # Target value network
    with tf.variable_scope('target'):
        _, qr1_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr1', **ac_kwargs)
        _, qr2_pi_targ = critic_fn(x2_ph, a_ph, pi2, name='qr2', **ac_kwargs)
        _, qc_pi_targ  = critic_fn(x2_ph, a_ph, pi2, name='qc', **ac_kwargs)
        _, qc_pi_var_targ = var_fn(x2_ph, a_ph, pi2, name='qc_var', **ac_kwargs)

    # Entropy bonus
    if fixed_entropy_bonus is None:
        with tf.variable_scope('entreg'):
            soft_alpha = tf.get_variable('soft_alpha',
                                         initializer=0.0,
                                         trainable=True,
                                         dtype=tf.float32)
        alpha = tf.nn.softplus(soft_alpha)
    else:
        alpha = tf.constant(fixed_entropy_bonus)
    log_alpha = tf.log(tf.clip_by_value(alpha,1e-8,1e8))

    # Cost penalty
    if use_costs:
        if fixed_cost_penalty is None:
            with tf.variable_scope('costpen'):
                soft_beta = tf.get_variable('soft_beta',
                                             initializer=0.0,
                                             trainable=True,
                                             dtype=tf.float32)
            beta = tf.nn.softplus(soft_beta)
            log_beta = tf.log(tf.clip_by_value(beta,1e-8,1e8))
        else:
            beta = tf.constant(fixed_cost_penalty)
            log_beta = tf.log(tf.clip_by_value(beta,1e-8,1e8))
    else:
        beta = 0.0  # costs do not contribute to policy optimization
        print('Not using costs')

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    if proc_id()==0:
        var_counts = tuple(count_vars(scope) for scope in 
                           ['main/pi', 'main/qr1', 'main/qr2', 'main/qc', 'main/qc_var', 'main'])
        print(('\nNumber of parameters: \t pi: %d, \t qr1: %d, \t qr2: %d, \t qc: %d, \t qc_var: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(qr1_pi, qr2_pi)
    min_q_pi_targ = tf.minimum(qr1_pi_targ, qr2_pi_targ)
    
    
    qc_var = tf.clip_by_value(qc_var, 1e-8, 1e8)
    qc_pi_var = tf.clip_by_value(qc_pi_var, 1e-8, 1e8)
    qc_pi_var_targ = tf.clip_by_value(qc_pi_var_targ, 1e-8, 1e8)

    # Targets for Q and (cost mean/var) regression
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*(min_q_pi_targ - alpha * logp_pi2))
    # 目标长期成本均值（Bellman）：E[G_c,t] = c_t + gamma * E_{a'~pi}[ E[G_c,t+1 | s', a'] ]
    # 这里用单样本 a'=pi2 近似期望（与 SAC 常见实现一致）。
    qc_backup = tf.stop_gradient(c_ph + gamma*(1-d_ph)*qc_pi_targ)
    # 目标长期成本方差：Var[G_c,t] = gamma^2 * E_{a'~pi}[ Var[G_c,t+1 | s', a'] ]
    qc_var_backup = tf.stop_gradient((gamma**2) * (1-d_ph) * qc_pi_var_targ)
    qc_var_backup = tf.clip_by_value(qc_var_backup, 0.0, 1e8)

    # CVaR（使用显著性水平 alpha_sig_level）
    qc_cvar = cvar_from_mean_var(qc, qc_var, alpha_sig_level)
    qc_pi_cvar = cvar_from_mean_var(qc_pi, qc_pi_var, alpha_sig_level)

    # Soft actor-critic losses

    # 策略更新：用 CVaR 作为成本约束信号
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q_pi + beta * qc_pi_cvar)

    qr1_loss = 0.5 * tf.reduce_mean((q_backup - qr1)**2)
    qr2_loss = 0.5 * tf.reduce_mean((q_backup - qr2)**2)
    qc_loss =  0.5 * tf.reduce_mean((qc_backup - qc)**2)
    qc_var_loss = 0.5 * tf.reduce_mean(qc_var + qc_var_backup - 2 * ((qc_var * qc_var_backup) ** 0.5))
    q_loss = qr1_loss + qr2_loss + qc_loss + qc_var_loss

    # Loss for alpha
    entropy_constraint *= act_dim
    pi_entropy = -tf.reduce_mean(logp_pi)
    # alpha_loss = - soft_alpha * (entropy_constraint - pi_entropy)
    alpha_loss = - alpha * (entropy_constraint - pi_entropy)
    print('using entropy constraint', entropy_constraint)

    # Loss for beta（对偶变量）：推动 CVaR <= cost_lim
    if use_costs:
        if fixed_cost_penalty is None:
            if cost_lim is None:
                raise ValueError("use_costs=True 且 fixed_cost_penalty=None 时，必须提供 cost_lim（CVaR 阈值）。")
            if proc_id() == 0:
                print('using CVaR cost_lim', cost_lim)
            beta_loss = beta * (float(cost_lim) - qc_cvar)
        else:
            # 固定惩罚系数时，不训练 beta；这里给一个可记录的占位 loss
            beta_loss = tf.constant(0.0, dtype=tf.float32)

    # Policy train op
    # (has to be separate from value train op, because qr1_pi appears in pi_loss)
    train_pi_op = MpiAdamOptimizer(learning_rate=lr).minimize(pi_loss, var_list=get_vars('main/pi'), name='train_pi')

    # Value train op
    with tf.control_dependencies([train_pi_op]):
        train_q_op = MpiAdamOptimizer(learning_rate=lr).minimize(q_loss, var_list=get_vars('main/q'), name='train_q')

    if fixed_entropy_bonus is None:
        entreg_optimizer = MpiAdamOptimizer(learning_rate=lr)
        with tf.control_dependencies([train_q_op]):
            train_entreg_op = entreg_optimizer.minimize(alpha_loss, var_list=get_vars('entreg'))

    if use_costs and fixed_cost_penalty is None:
        costpen_optimizer = MpiAdamOptimizer(learning_rate=lr*lr_scale)
        if fixed_entropy_bonus is None:
            with tf.control_dependencies([train_entreg_op]):
                train_costpen_op = costpen_optimizer.minimize(beta_loss, var_list=get_vars('costpen'))
        else:
            with tf.control_dependencies([train_q_op]):
                train_costpen_op = costpen_optimizer.minimize(beta_loss, var_list=get_vars('costpen'))
            

    # Polyak averaging for target variables
    target_update = get_target_update('main', 'target', polyak)

    # Single monolithic update with explicit control dependencies
    with tf.control_dependencies([train_pi_op]):
        with tf.control_dependencies([train_q_op]):
            grouped_update = tf.group([target_update])

    if fixed_entropy_bonus is None:
        grouped_update = tf.group([grouped_update, train_entreg_op])
    if use_costs and fixed_cost_penalty is None:
        grouped_update = tf.group([grouped_update, train_costpen_op])

    # Initializing targets to match main variables
    # As a shortcut, use our exponential moving average update w/ coefficient zero
    target_init = get_target_update('main', 'target', 0.0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph},
                                outputs={'mu': mu, 'pi': pi, 'qr1': qr1, 'qr2': qr2, 'qc': qc})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]
        
    def test_agent(n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_cost, ep_len, ep_goals, = test_env.reset(), 0, False, 0, 0, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, info = test_env.step(get_action(o, True))
                if render and proc_id() == 0 and j == 0:
                    test_env.render()
                ep_ret += r
                ep_cost += info.get('cost', 0)
                ep_len += 1
                ep_goals += 1 if info.get('goal_met', False) else 0
            logger.store(TestEpRet=ep_ret, TestEpCost=ep_cost, TestEpLen=ep_len, TestEpGoals=ep_goals)

    start_time = time.time()
    o, r, d, ep_ret, ep_cost, ep_len, ep_goals = env.reset(), 0, False, 0, 0, 0, 0
    total_steps = steps_per_epoch * epochs

    # variables to measure in an update
    vars_to_get = dict(LossPi=pi_loss, LossQR1=qr1_loss, LossQR2=qr2_loss, LossQC=qc_loss, LossQCVar=qc_var_loss,
                       QR1Vals=qr1, QR2Vals=qr2, QCVals = qc, QCVar = qc_var, LogPi=logp_pi, PiEntropy=pi_entropy,
                       Alpha=alpha, LogAlpha=log_alpha, LossAlpha=alpha_loss)
    if use_costs:
        vars_to_get.update(dict(Beta=beta, LogBeta=log_beta, LossBeta=beta_loss))

    print('starting training', proc_id())

    # Main loop: collect experience in env and update/log each epoch
    number_model = 0
    cum_cost = 0
    local_steps = 0
    local_steps_per_epoch = steps_per_epoch // num_procs()
    local_batch_size = batch_size // num_procs()
    epoch_start_time = time.time()
    for t in range(total_steps // num_procs()):
        """
        Until local_start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards,
        use the learned policy.
        """
        if t > local_start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, info = env.step(a)
        r *= reward_scale  # yee-haw
        c = info.get('cost', 0)
        ep_ret += r
        ep_cost += c
        ep_len += 1
        ep_goals += 1 if info.get('goal_met', False) else 0
        local_steps += 1
        
        # Track cumulative cost over training
        cum_cost += c

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, c)
        
        # 打印经验值（每隔一定步数或episode结束时打印，避免输出过多）
        if proc_id() == 0 and (local_steps % 100 == 0 or d or ep_len == max_ep_len):
            print(f"[exp] step={local_steps:6d} | obs={o} | action={a} | reward={r:8.4f} | cost={c:8.4f} | "
                  f"next_obs={o2} | done={d} | ep_len={ep_len:3d} | ep_ret={ep_ret:8.2f} | ep_cost={ep_cost:8.4f}")

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len, EpGoals=ep_goals)
            o, r, d, ep_ret, ep_cost, ep_len, ep_goals = env.reset(), 0, False, 0, 0, 0, 0

        if t > 0 and t % update_freq == 0:
            #if index_risk < 0:
            #    index_risk = 0
                
            for j in range(update_freq):
                batch = replay_buffer.sample_batch(local_batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             c_ph: batch['costs'],
                             d_ph: batch['done'],
                            }
                #aaa = sess.run(logp_pi, feed_dict)
                #bbb = sess.run(pi_entropy, feed_dict)
                #print(aaa)
                #print(bbb)
                if t < local_update_after:
                    logger.store(**sess.run(vars_to_get, feed_dict))
                else:
                    values, _ = sess.run([vars_to_get, grouped_update], feed_dict)
                    logger.store(**values)

        # End of epoch wrap-up
        if t > 0 and t % local_steps_per_epoch == 0:
            epoch = t // local_steps_per_epoch
            
            #=====================================================================#
            #  Cumulative cost calculations                                       #
            #=====================================================================#
            cumulative_cost = mpi_sum(cum_cost)
            cost_rate = cumulative_cost / ((epoch+1)*steps_per_epoch)
            
            #if index_risk > 0:
            #    index_risk = index_risk - 1/300

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, number_model)
                number_model += 1

            # Test the performance of the deterministic version of the agent.
            test_start_time = time.time()
            test_agent()
            logger.store(TestTime=time.time() - test_start_time)

            logger.store(EpochTime=time.time() - epoch_start_time)
            epoch_start_time = time.time()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpCost', with_min_and_max=True)
            logger.log_tabular('TestEpCost', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('EpGoals', average_only=True)
            logger.log_tabular('TestEpGoals', average_only=True)
            logger.log_tabular('CumulativeCost', cumulative_cost)
            logger.log_tabular('CostRate', cost_rate)
            logger.log_tabular('TotalEnvInteracts', mpi_sum(local_steps))
            logger.log_tabular('QR1Vals', with_min_and_max=True)
            logger.log_tabular('QR2Vals', with_min_and_max=True)
            logger.log_tabular('QCVals', with_min_and_max=True)
            logger.log_tabular('QCVar', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQR1', average_only=True)
            logger.log_tabular('LossQR2', average_only=True)
            logger.log_tabular('LossQC', average_only=True)
            logger.log_tabular('LossQCVar', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('LogAlpha', average_only=True)
            logger.log_tabular('Alpha', average_only=True)
            if use_costs:
                logger.log_tabular('LossBeta', average_only=True)
                logger.log_tabular('LogBeta', average_only=True)
                logger.log_tabular('Beta', average_only=True)
            logger.log_tabular('PiEntropy', average_only=True)
            logger.log_tabular('TestTime', average_only=True)
            logger.log_tabular('EpochTime', average_only=True)
            logger.log_tabular('TotalTime', time.time()-start_time)
            logger.dump_tabular()
