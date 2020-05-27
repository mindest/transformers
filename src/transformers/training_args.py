import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

from .file_utils import cached_property, is_torch_available, torch_required


if is_torch_available():
    import torch


logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False, metadata={"help": "Overwrite the content of the output directory"}
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluate_during_training: bool = field(
        default=False, metadata={"help": "Run evaluation during training at each logging step."}
    )

    per_gpu_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for training."})
    per_gpu_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU/CPU for evaluation."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    logging_dir: Optional[str] = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": "Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default"
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Avoid using CUDA even if it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})

    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html"
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    world_rank: int = field(default=-1, metadata={"help": "For distributed training: world_rank"})
    world_size: int = field(default=1, metadata={"help": "For distributed training: world_size"})
    master_port: int = field(default=12345, metadata={"help": "For distributed training: free port on rank 0 node"})
    master_node: str = field(default="localhost", metadata={"help": "For distributed training: address of rank 0 node"})
    ort_trainer: bool = field(default=False, metadata={"help": "Use ORT to train instead of PyTorch"})

    @property
    def train_batch_size(self) -> int:
        return self.per_gpu_train_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        return self.per_gpu_eval_batch_size * max(1, self.n_gpu)

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif self.ort_trainer:
            self.update_args()
            os.environ['RANK'] = str(self.world_rank)
            os.environ['WORLD_SIZE'] = str(self.world_size)
            os.environ['MASTER_ADDR'] = self.master_node
            os.environ['MASTER_PORT'] = str(self.master_port)

            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            n_gpu = self.world_size
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1
        return device, n_gpu

    @property
    @torch_required
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        return self._setup_devices[1]
    
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def update_args_from_env(self):
        self.local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        self.world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        self.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

        #NCCL environment. Still works without it.
        #os.environ['NCCL_SOCKET_IFNAME'] = '^docker0,lo'
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
        os.environ["NCCL_IB_DISABLE"] = '0' #for IB

        # Only for multi node. This env is not present for single node AML jobs
        if 'AZ_BATCH_MASTER_NODE' in os.environ:
            master_node_params = os.environ['AZ_BATCH_MASTER_NODE'].split(':')        
            self.master_node = master_node_params[0]
            self.master_port = master_node_params[1]
        elif 'AZ_BATCHAI_MPI_MASTER_NODE' in os.environ:
            self.master_node = os.environ['AZ_BATCHAI_MPI_MASTER_NODE']
    
    def update_args(self):
        if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
            self.update_args_from_env()
        else:
            self.local_rank = 0
            self.world_rank = 0
            self.world_size = 1