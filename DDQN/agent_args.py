AGENT_ARGS = {
    'hidden_layers_size': 128,
    'buffer_size': 3_000,
    'batch_size': 64,
    'start_learning_steps': 1_500,
    'update_target_steps': 400,
    'agent_type': 'ddqn',
    'loss_func': 'mse',
    'learning_rate': 0.001,
    'grads_clip_lim': 1.0,
    'gamma': 0.95,
    'eps_start': 0.99,
    'eps_end': 0.05,
    'eps_decay_steps': 4_000
}
