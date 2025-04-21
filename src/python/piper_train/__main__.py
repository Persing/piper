import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

from piper_tts_trainer.piper.src.python.piper_train.safe_checkpoint_io import SafeCheckpointIO
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


def load_checkpoint_fix(checkpoint_path):
    """
       Loads a checkpoint with weights_only=False to avoid UnpicklingError.
       """
    # Explicitly allow PosixPath type
    from pathlib import PosixPath
    torch.serialization.add_safe_globals([PosixPath])  # <--- CHANGE TO PosixPath

    # Use weights_only=False with caution
    return torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False  # <--- CHANGE TO FALSE BUT ADD WARNING
    )

def main():
    logging.basicConfig(level=logging.DEBUG)

    # Increase Python recursion limit to handle complex data structures
    sys.setrecursionlimit(10000)

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--default-root-dir", type=str, default=None,
                        help="Root directory for saving logs and checkpoints")

    parser.add_argument(
        "--resume-from-single-speaker-checkpoint",
        type=str,
        default=None,
        help="Path to a single-speaker checkpoint to resume from (for multi-speaker models only)"
    )

    # Training configuration
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-epochs", type=int, default=1000)


    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--devices", type=int, default=1)

    parser.add_argument("--quality", type=str, default="medium", choices=["low", "medium", "high"])

    parser.add_argument("--checkpoint-epochs", type=int, default=1)
    #
    # # Add model-specific args (including batch-size)
    VitsModel.add_model_specific_args(parser)

    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    # Configure trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=[ModelCheckpoint(every_n_epochs=args.checkpoint_epochs)] if args.checkpoint_epochs else [],
        logger=True,
        enable_progress_bar=True,
        num_sanity_val_steps=2,
        precision=32
    )

    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = Path(args.dataset_dir) / "dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} not found")

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    if args.checkpoint_epochs is not None:
        trainer.callbacks = [ModelCheckpoint(every_n_epochs=args.checkpoint_epochs)]
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        # dataset=str(dataset_path),
        **dict_args,
    )

    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
            map_location=load_checkpoint_fix,  # Use the custom loader
            weights_only=False
        )
        g_dict = model_single.model_g.state_dict()
        for key in list(g_dict.keys()):
            # Remove keys that can't be copied over due to missing speaker embedding
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)

        # Copy over the multi-speaker model, excluding keys related to the
        # speaker embedding (which is missing from the single-speaker model).
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, model_single.model_d.state_dict())
        _LOGGER.info(
            "Successfully converted single-speaker checkpoint to multi-speaker"
        )

    trainer.fit(model, args.resume_from_checkpoint)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()