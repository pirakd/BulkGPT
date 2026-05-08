"""Training hyperparameters and run configuration for train.py."""

from dataclasses import dataclass, field


def _default_dtype() -> str:
    import torch

    # Mixed precision defaults are CUDA-oriented. On MPS/CPU, prefer float32 for
    # numerical stability unless explicitly overridden from the CLI/config.
    if torch.cuda.is_available():
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


@dataclass
class TrainConfig:
    """Default config values designed to train a gpt2 (124M) on OpenWebText."""

    # I/O
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = True  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    # wandb logging
    wandb_log: bool = False  # disabled by default
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"  # 'run' + str(time.time())
    # data
    dataset: str = "openwebtext"
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    batch_size: int = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 1024
    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend: str = "nccl"  # 'nccl', 'gloo', etc.
    # system
    device: str = "mps"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = field(default_factory=_default_dtype)  # 'float32', 'bfloat16', or 'float16'
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster


@dataclass
class ShakespeareConfig(TrainConfig):
    dataset: str = "shakespeare"
    gradient_accumulation_steps: int = 4
    batch_size: int = 16
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False
    learning_rate: float = 1e-3
    max_iters: int = 100000
    warmup_iters: int = 500
    lr_decay_iters: int = 100000
    min_lr: float = 2e-5
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    eval_interval: int = 100
    compile: bool = False


@dataclass
class FastShakespeareConfig(ShakespeareConfig):
    """Speed-oriented preset for quick iteration on Shakespeare."""

    gradient_accumulation_steps: int = 2
    batch_size: int = 32
    eval_interval: int = 200
    eval_iters: int = 50
    log_interval: int = 10
    learning_rate: float = 5e-4
    warmup_iters: int = 200
    dtype: str = "float16"
    compile: bool = False