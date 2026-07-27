DEFAULT_NET_CONFIG = {
    "summary_network_settings": {
        "dropout": 0.01070354852467715,
        "num_seeds": 7,
        "summary_dim": 32,
        "embed_dim": (128, 128),
        
    },
    "inference_network_settings": {
        "network_type": 'FlowMatching',
        "dropout": 0.01070354852467715
    },
    "workflow_settings": {
        "learning_rate": 0.0005721790353631461,
        "save_best_only":True
    },
}

MODEL_CONFIG = {
    "DDM": {
        "prior_range": {
            'a': [0.1, 2],
            'ndt': [0.1, 1],
            'v_c': [0, 5],
            'v_i': [0, 5]
        },
        "param_keys": ['a', 'ndt', 'v_c', 'v_i'],
        "param_names": [r'$a$', r'$t$', r'$v_{cong}$', r'$v_{incong}$']
    },
    "DMC": {
        "prior_range": {
            'a': [0.7, 1.26],     # transform according formula a_{new} = a_{old} * \frac{dc_{new}}{dc_{old}} * sqrt(\frac{dt_{old}}{dt_{new}})
            'ndt': [0.27, 0.40],      # 0.001 (dt_{new}}) times
            'v_c': [1.58, 6.32], # transform according formula v_{new} = v_{old} * \frac{dc_{new}}{dc_{old}} * sqrt(\frac{dt_{new}}{dt_{old}})
            'alpha': [1.5, 4.5],
            'eta': [118, 316],    # transform following as v_c
            'tau': [20, 120],
        },
        "param_keys": ['a', 'ndt', 'v_c', 'alpha', 'eta', 'tau'],
        "param_names": [r'$a$', r'$t$', r'$v_{c}$', r'$\alpha$', r'$\eta$', r'$\tau$']
    },
    # "DMC": { # see (White et al., 2018)
    #     "prior_range": {
    #         'a': [90, 160],
    #         'ndt': [270, 400],
    #         'v_cong': [0.2, 0.8],
    #         'alpha': [1.5, 4.5],
    #         'eta': [15, 40],
    #         'tau': [20, 120],
    #     },
    #    "param_keys": ['a', 'ndt', 'v_c', 'alpha', 'eta', 'tau'],
    #    "param_names": [r'$a$', r'$t$', r'$v_{c}$', r'$\alpha$', r'$\eta$', r'$\tau$']
    # },
    "SSP": {
        "prior_range": {
            'a': [1.4, 3.8],  # 20 times
            'ndt': [0.15, 0.45],
            'p': [2, 5.5],  # 10 times
            'sd_a': [1, 2.6],
            'r_d': [10, 26],  # 1000 times according to 1s to 0.001s
        },
        "param_keys": ['a', 'ndt', 'p', 'sd_a', 'r_d'],
        "param_names": [r'$a$', r'$t$', r'$p$', r'$sd_a$', r'$r_d$']
    },
    # "SSP": { # see (White et al., 2018)
    #     "prior_range": {
    #         'a': [0.14, 0.38],  # 2 times  
    #         'ndt': [0.15, 0.45],
    #         'p': [0.2, 0.55],
    #         'sd_a': [1, 2.6],
    #         'r_d': [0.01, 0.026], 
    #     },
    #     "param_keys": ['a', 'ndt', 'p', 'sd_a', 'r_d'],
    #     "param_names": [r'$a$', r'$t$', r'$p$', r'$sd_a$', r'$r_d$']
    # },
    "DSTP": {
        "prior_range": {
            'a': [1.4, 3.8],  # 10 times
            'ndt': [0.15, 0.45],
            'vta': [0.5, 1.5],  # 10 times
            'vfl': [0.5, 2.5],  # 10 times
            'vss': [2.5, 5.5],  # 10 times
            'vp2': [4.0, 12.0],  # 10 times
            'ass': [1.4, 3.8],  # 10 times
        },
        "param_keys": ['a', 'ndt', 'vta', 'vfl', 'vss', 'vp2', 'ass'],
        "param_names": [r'$a$', r'$t$', r'$v_{ta}$', r'$v_{fl}$', r'$v_{ss}$', r'$v_{p2}$', r'$a_{ss}$']
    },
    # "DSTP": { # see (White et al., 2018)
    #     "prior_range": {
    #         'a': [0.14, 0.38],
    #         'ndt': [0.15, 0.45],
    #         'vta': [0.05, 0.15],
    #         'vfl': [0.05, 0.25],
    #         'vss': [0.25, 0.55],
    #         'vp2': [0.40, 1.20],
    #         'ass': [0.14, 0.38],
    #     },
    #     "param_keys": ['a', 'ndt', 'vta', 'vfl', 'vss', 'vp2', 'ass'],
    #     "param_names": [r'$a$', r'$t$', r'$v_{ta}$', r'$v_{fl}$', r'$v_{ss}$', r'$v_{p2}$', r'$ass$']
    # },
    "SSP_fixed_ratio": {
        "prior_range": {
            'a': [1.4, 3.8],       # 20 times
            'ndt': [0.15, 0.45],
            'p': [2, 5.5],         # 10 times
            'rd_sda_ratio': [3, 30],   # origin ratio of sda on rd is like 38~260; but we use the ratio of r_d on sd_a with r_d times 1000
        },
        "param_keys": ['a', 'ndt', 'p', 'rd_sda_ratio'],
        "param_names": [r'$a$', r'$t$', r'$p$', r'$r_d/sd_a$']
    },
    "DSTP_fixed_ratio": {
        "prior_range": {
            'a': [1.4, 3.8],              # 10 times
            'ndt': [0.15, 0.45],
            'vta': [0.5, 1.5],         # 10 times
            'vfl_vss_ratio': [0.01, 1],   # same as flexDDM; with the full range. 
            'vp2': [4.0, 12.0],        # 10 times
            'ass': [1.4, 3.8],         # 10 times
        },
        "param_keys": ['a', 'ndt', 'vta', 'vfl_vss_ratio', 'vp2', 'ass'],
        "param_names": [r'$a$', r'$t$', r'$v_{ta}$', r'$v_{fl}/v_{ss}$', r'$v_{p2}$', r'$a_{ss}$']
    },
    "DMC_fixed_alpha": {
        "prior_range": {
            'a': [0.7, 1.26],
            'ndt': [0.27, 0.40],
            'v_c': [1.58, 6.32],
            'eta': [118, 316],
            'tau': [20, 120],
        },
        "param_keys": ['a', 'ndt', 'v_c', 'eta', 'tau'],
        "param_names": [r'$a$', r'$t$', r'$v_{c}$', r'$\eta$', r'$\tau$']
    }
}
def get_param_mappings(model_config):
    """
    Generate param_keys to param_names mapping for each model.
    
    Args:
        model_config: Dictionary containing model configurations
        
    Returns:
        Dictionary with model names as keys and {param_key: param_name} as values
    """
    return {
        model_name: dict(zip(config["param_keys"], config["param_names"]))
        for model_name, config in model_config.items()
    }

PARAMS_KEY_NAME_MAPPING = get_param_mappings(MODEL_CONFIG)

CONTEXT_CONFIG = {
    # experiment design
    "factor_levels":{
        "congruency":["incong", "cong"], # 1=cong
    },
    # n_trials_range for each cell in factor_levels. minimal is 50 from white 2018
    "n_trials_range":[50, 500], 
}
