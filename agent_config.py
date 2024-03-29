
# DDPG        
DDPG_Vanilla_agent_config = {'agent_name': 'DDPG', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 128, 'warm_up': 1024, 
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
DDPG_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

DDPG_TQC_agent_config = {'agent_name': 'DDPG', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
DDPG_TQC_agent_config['extension'] = {'name': 'TQC', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

DDPG_gSDE_agent_config = {'agent_name': 'DDPG', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
DDPG_gSDE_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}

# TD3
TD3_Vanilla_agent_config = {'agent_name': 'TD3', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'actor_update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
TD3_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

TD3_TQC_agent_config = {'agent_name': 'TD3', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'actor_update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
TD3_TQC_agent_config['extension'] = {'name': 'TQC', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

TD3_gSDE_agent_config = {'agent_name': 'TD3', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'actor_update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
TD3_gSDE_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}

# PPO
PPO_Vanilla_agent_config = {'agent_name': 'PPO', 'gamma' : 0.99, 'total_batch_size': 1024, 'batch_size': 256, 'epoch_num': 20, \
                            'entropy_coeff': 0.001, 'entropy_coeff_reduction_rate': 0.9997, 'entropy_coeff_min': 0.00005, 'epsilon': 0.2, \
                            'std_bound': [0.01, 0.3], 'std_reduction_rate': 1, 'std_min': 0.25, 'lr_actor': 0.000075, 'lr_critic': 0.00015, \
                            'reward_normalize' : True, 'reward_min': -5, 'reward_max': 5, 'log_prob_min': -3.5, 'log_prob_max': 3.5}
PPO_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'use_GAE': True, 'use_SIL': False}
PPO_Vanilla_agent_config['extension']['GAE_config'] = {'use_gae_norm': False, 'lambda': 0.98} # std_reduction is not good....

ME_PPO_agent_config = {'agent_name': 'PPO', 'gamma' : 0.99, 'total_batch_size': 512, 'batch_size': 128, 'epoch_num': 4, \
                            'entropy_coeff': 0.01, 'entropy_coeff_reduction_rate': 0.9997, 'entropy_coeff_min': 0.0005, \
                            'epsilon': 0.2, 'std_bound': [0.02, 0.2], 'lr_actor': 0.0005, 'lr_critic': 0.001, 'reward_normalize' : True, \
                            'reward_min': -50, 'reward_max': 150, 'log_prob_min': -1.0, 'log_prob_max': 2.0}
ME_PPO_agent_config['extension'] = {'name': 'ME', 'use_GAE': True, 'use_ME': True}
ME_PPO_agent_config['extension']['GAE_config'] = {'use_gae_norm': True, 'lambda': 0.95}
ME_PPO_agent_config['extension']['ME_config'] = {}

PPO_SIL_agent_config = {'agent_name': 'PPO', 'gamma' : 0.99, 'total_batch_size': 1024, 'batch_size': 256, 'epoch_num': 20, \
                            'entropy_coeff': 0.1, 'entropy_coeff_reduction_rate': 0.9997, 'entropy_coeff_min': 0.00005, 'epsilon': 0.2, \
                            'std_bound': [0.01, 0.3], 'std_reduction_rate': 1, 'std_min': 0.25, 'lr_actor': 0.000075, 'lr_critic': 0.00015, \
                            'reward_normalize' : True, 'reward_min': -5, 'reward_max': 5, 'log_prob_min': -3.5, 'log_prob_max': 3.5}
PPO_SIL_agent_config['extension'] = {'name': 'SIL', 'use_GAE': True, 'use_SIL': True}
PPO_SIL_agent_config['extension']['GAE_config'] = {'use_gae_norm': False, 'lambda': 0.97}
PPO_SIL_agent_config['extension']['SIL_config'] = {'buffer_size': 200000, 'min_batch_size': 64, 'batch_size': 512, 'lr_sil_actor': 0.0005, 'lr_sil_critic': 0.001,
                                                   'epoch_num': 2, 'return_criteria': 0, 'naive_criteria': False, 'recent_return_coeff' : 0.001,
                                                   'log_prob_min': -3, 'log_prob_max': 3, 'adv_min': -1, 'adv_max': 1}

PPO_gSDE_agent_config = {'agent_name': 'PPO', 'gamma' : 0.99, 'total_batch_size': 2048, 'batch_size': 512, 'epoch_num': 10, \
                            'entropy_coeff': 0.01, 'entropy_coeff_reduction_rate': 0.9997, 'entropy_coeff_min': 0.0005, \
                            'epsilon': 0.2, 'std_bound': [0.02, 0.2], 'lr_actor': 0.0005, 'lr_critic': 0.001, 'reward_normalize' : True, \
                            'reward_min': -20, 'reward_max': 20, 'log_prob_min': -1.0, 'log_prob_max': 2.0}
PPO_gSDE_agent_config['extension'] = {'name': 'gSDE', 'use_GAE': True, 'use_gSDE': True}
PPO_gSDE_agent_config['extension']['GAE_config'] = {'use_gae_norm': True, 'lambda': 0.95}
PPO_gSDE_agent_config['extension']['gSDE_config'] = {'latent_space': 64, 'n_step_reset': 16}

# PPG
PPG_Vanilla_agent_config = {'agent_name': 'PPG', 'gamma' : 0.99, 'batch_size': 128, 'epoch_num': 4, 'entropy_coeff': 0.005, 'entropy_reduction_rate': 0.999999, 
		                    'epsilon': 0.2, 'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 512, 'use_GAE': True, 'lambda': 0.95, 'reward_normalize' : False}
PPG_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'std_bound': [0.02, 0.3]}

PPG_SIL_agent_config = {'agent_name': 'PPG', 'gamma' : 0.99, 'batch_size': 128, 'epoch_num': 4, 'entropy_coeff': 0.005, 'entropy_reduction_rate': 0.999999, 
		                    'epsilon': 0.2, 'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 512, 'use_GAE': True, 'lambda': 0.95, 'reward_normalize' : False}
PPG_SIL_agent_config['extension'] = {'name': 'Vanilla', 'std_bound': [0.02, 0.3]}

PPG_gSDE_agent_config = {'agent_name': 'PPG', 'gamma' : 0.99, 'update_freq': 2, 'batch_size': 128, 'epoch_num': 20, 'eps_clip': 0.2, 'eps_reduction_rate': 0.999999, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_GAE': True, 'lambda': 0.995, 'reward_normalize' : False}
PPG_gSDE_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}

# SAC
SAC_Vanilla_agent_config = {'agent_name': 'SAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'actor_update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
SAC_Vanilla_agent_config['extension'] = {'name': 'Vanilla', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

SAC_TQC_agent_config = {'agent_name': 'SAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'actor_update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
SAC_TQC_agent_config['extension'] = {'name': 'TQC', 'gaussian_std': 0.1, 'noise_clip': 0.5, 'noise_reduction_rate': 0.999999}

SAC_gSDE_agent_config = {'agent_name': 'SAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'actor_update_freq': 2, 'batch_size': 128, 'warm_up': 1024, \
                        'lr_actor': 0.001, 'lr_critic': 0.002, 'buffer_size': 2000000, 'use_PER': True, 'use_ERE': False, 'reward_normalize' : False}
SAC_gSDE_agent_config['extension'] = {'name': 'gSDE', 'latent_space': 64, 'n_step_reset': 16}


IDAC_Gaussian_No_Alpha_No_Reparam_agent_config = {'agent_name': 'IDAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 256, 'warm_up': 2048, \
                        'lr_actor': 0.0003, 'lr_critic': 0.0003, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False, \
                        'alpha': 0.2, 'quantile_num': 32, 'noise_dim': 5}
IDAC_Gaussian_No_Alpha_No_Reparam_agent_config['extension'] = {'name': 'Gaussian_No_Alpha_No_Reparam', 'use_implicit_actor': False, 'use_automatic_entropy_tuning': False}
IDAC_Gaussian_No_Alpha_No_Reparam_agent_config['extension']['gaussian_actor_config'] = {'noise_dim': 5, 'log_sig_min': -20, 'log_sig_max': 2, \
                                                                            'log_prob_min': -5, 'log_prob_max': 5, 'use_reparam_trick': False}


IDAC_Gaussian_No_Alpha_Reparam_agent_config = {'agent_name': 'IDAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 256, 'warm_up': 2048, \
                        'lr_actor': 0.0003, 'lr_critic': 0.0003, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False, \
                        'alpha': 0.2, 'quantile_num': 32, 'noise_dim': 5}
IDAC_Gaussian_No_Alpha_Reparam_agent_config['extension'] = {'name': 'Gaussian_No_Alpha_Reparam', 'use_implicit_actor': False, 'use_automatic_entropy_tuning': False}
IDAC_Gaussian_No_Alpha_Reparam_agent_config['extension']['gaussian_actor_config'] = {'noise_dim': 5, 'log_sig_min': -20, 'log_sig_max': 2, \
                                                                            'log_prob_min': -5, 'log_prob_max': 5, 'use_reparam_trick': True}


IDAC_Gaussian_Alpha_No_Reparam_agent_config = {'agent_name': 'IDAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 256, 'warm_up': 2048, \
                        'lr_actor': 0.0003, 'lr_critic': 0.0003, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False, \
                        'alpha': 0.2, 'quantile_num': 32, 'noise_dim': 5}
IDAC_Gaussian_Alpha_No_Reparam_agent_config['extension'] = {'name': 'Gaussian_Alpha_No_Reparam', 'use_implicit_actor': False, 'use_automatic_entropy_tuning': True}
IDAC_Gaussian_Alpha_No_Reparam_agent_config['extension']['gaussian_actor_config'] = {'noise_dim': 5, 'log_sig_min': -20, 'log_sig_max': 2, \
                                                                            'log_prob_min': -5, 'log_prob_max': 5, 'use_reparam_trick': False}
IDAC_Gaussian_Alpha_No_Reparam_agent_config['extension']['automatic_alpha_config'] = {'use_target_entropy' : False}


IDAC_Gaussian_Alpha_Reparam_agent_config = {'agent_name': 'IDAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 256, 'warm_up': 2048, \
                        'lr_actor': 0.0003, 'lr_critic': 0.0003, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False, \
                        'alpha': 0.2, 'quantile_num': 32, 'noise_dim': 5}
IDAC_Gaussian_Alpha_Reparam_agent_config['extension'] = {'name': 'Gaussian_Alpha_Reparam', 'use_implicit_actor': False, 'use_automatic_entropy_tuning': True}
IDAC_Gaussian_Alpha_Reparam_agent_config['extension']['gaussian_actor_config'] = {'noise_dim': 5, 'log_sig_min': -20, 'log_sig_max': 2, \
                                                                            'log_prob_min': -5, 'log_prob_max': 5, 'use_reparam_trick': True}
IDAC_Gaussian_Alpha_Reparam_agent_config['extension']['automatic_alpha_config'] = {'use_target_entropy' : False}


IDAC_Implicit_No_Alpha_agent_config = {'agent_name': 'IDAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 256, 'warm_up': 2048, \
                        'lr_actor': 0.0003, 'lr_critic': 0.0003, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False, \
                        'alpha': 0.2, 'quantile_num': 16, 'noise_dim': 5}
IDAC_Implicit_No_Alpha_agent_config['extension'] = {'name': 'Implicit_No_Alpha', 'use_implicit_actor': True, 'use_automatic_entropy_tuning': False}
IDAC_Implicit_No_Alpha_agent_config['extension']['implicit_actor_config'] = {'action_num': 10, 'noise_dim': 5, 'log_sig_min': -20, 'log_sig_max': 2, \
                                                                            'log_prob_min': -5, 'log_prob_max': 5}


IDAC_Implicit_Alpha_agent_config = {'agent_name': 'IDAC', 'gamma' : 0.99, 'tau': 0.005, 'update_freq': 2, 'batch_size': 256, 'warm_up': 2048, \
                        'lr_actor': 0.0003, 'lr_critic': 0.0003, 'buffer_size': 1000000, 'use_PER': False, 'use_ERE': False, 'reward_normalize' : False, \
                        'alpha': 0.2, 'quantile_num': 16, 'noise_dim': 5}
IDAC_Implicit_Alpha_agent_config['extension'] = {'name': 'Implicit_Alpha', 'use_implicit_actor': True, 'use_automatic_entropy_tuning': True}
IDAC_Implicit_Alpha_agent_config['extension']['implicit_actor_config'] = {'action_num': 10, 'noise_dim': 5, 'log_sig_min': -20, 'log_sig_max': 2, \
                                                                            'log_prob_min': -5, 'log_prob_max': 5}
IDAC_Implicit_Alpha_agent_config['extension']['automatic_alpha_config'] = {'use_target_entropy' : False}
