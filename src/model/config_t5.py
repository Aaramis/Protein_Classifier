ds_config = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    # https://github.com/huggingface/transformers/issues/24640
    # "optimizer": {
    #     "type": "AdamW",
    #     "params": {
    #         "lr": "auto",
    #         "betas": "auto",
    #         "eps": "auto",
    #         "weight_decay": "auto"
    #     }
    # },

    # "scheduler": {
    #     "type": "WarmupLR",
    #     "params": {
    #         "warmup_min_lr": "auto",
    #         "warmup_max_lr": "auto",
    #         "warmup_num_steps": "auto"
    #     }
    # },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False
}
