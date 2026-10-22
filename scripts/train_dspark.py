#!/usr/bin/env python3
# coding=utf-8
"""DSpark Training Script."""

import argparse
import functools
import logging
import math
import os
import shutil
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dspark import OnlineDSparkModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import get_last_checkpoint, print_on_rank0, print_with_rank


def parse_args():
    parser = argparse.ArgumentParser(description="Train DSpark Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument("--target-model-backend", type=str, default="hf", choices=["sglang", "hf"])
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument("--mask-token-id", type=int, default=None)
    model_group.add_argument("--attention-backend", type=str, default="flex_attention", choices=["eager", "sdpa", "flex_attention"])
    model_group.add_argument("--trust-remote-code", action="store_true")
    model_group.add_argument("--num-anchors", type=int, default=512)
    model_group.add_argument("--loss-decay-gamma", type=float, default=4.0)
    model_group.add_argument("--ce-loss-alpha", type=float, default=0.1)
    model_group.add_argument("--l1-loss-alpha", type=float, default=0.9)
    model_group.add_argument("--kl-loss-alpha", type=float, default=0.0)
    model_group.add_argument("--position-weighting", type=str, default="decay", choices=["decay", "dpace"],
                             help="Position weights: static exponential decay (via --loss-decay-gamma) or the detached D-PACE continuation value (with --kl-loss-alpha > 0 the acceptance proxy becomes D-PAKL's soft-label exp(-C_i)).")
    model_group.add_argument("--dpace-alpha", type=float, default=0.5,
                             help="Smoothing alpha for --position-weighting dpace.")
    model_group.add_argument("--confidence-head-alpha", type=float, default=1.0)
    model_group.add_argument("--markov-rank", type=int, default=256)
    model_group.add_argument("--markov-head-type", type=str, default="vanilla", choices=["vanilla", "gated", "rnn"])
    model_group.add_argument("--embedding-key", type=str, default=None)
    model_group.add_argument("--lm-head-key", type=str, default=None)

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument("--build-dataset-num-proc", type=int, default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)))

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)
    output_group.add_argument("--save-schedule", type=str, default=None)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument("--tp-size", type=int, default=1)

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def needs_target_last_hidden(args, draft_model) -> bool:
    """Whether the run has to fetch the target's final hidden states.

    L1/KL distillation need them for the target distribution, and the
    confidence head needs the resulting per-token L1 as its BCE target
    (1 - 0.5 * L1), so a pure-CE run with the head enabled needs them too.
    """
    if args.l1_loss_alpha > 0 or args.kl_loss_alpha > 0:
        return True
    return args.confidence_head_alpha > 0 and draft_model.confidence_head is not None


def build_models(args) -> Tuple[DFlashTargetModel, DSparkDraftModel]:
    print_on_rank0(f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend")

    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
        # DSparkDraftModel reads every head field out of dflash_config, so the
        # CLI overrides have to land there (a dspark_config key is not read at
        # build time; it is only written back out for sglang at save time).
        if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
            draft_config.dflash_config = {}
        draft_config.dflash_config["markov_head_type"] = args.markov_head_type
        draft_config.dflash_config["markov_rank"] = args.markov_rank
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        draft_config.dflash_config = {
            "markov_rank": args.markov_rank,
            "markov_head_type": args.markov_head_type,
            "enable_confidence_head": args.confidence_head_alpha > 0,
            "confidence_head_with_markov": True,
        }
        draft_config.architectures = ["DSparkDraftModel"]

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    draft_config._attn_implementation = args.attention_backend
    draft_model = DSparkDraftModel(draft_config).cuda().to(torch.bfloat16)
    target_model.set_capture_layers(draft_model.target_layer_ids)

    # L1/KL distillation build the target distribution from the target's final
    # hidden states; the sglang backend only returns them when explicitly
    # enabled (return_last_hidden_state defaults to False), so without this the
    # L1 loss silently trains as 0. The confidence head needs them too: its BCE
    # target is 1 - 0.5 * L1, so a pure-CE run still has to fetch them.
    if needs_target_last_hidden(args, draft_model):
        target_model.enable_last_hidden_state()

    print_on_rank0(f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}")
    return target_model, draft_model


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    import hashlib
    cache_params_string = f"{args.train_data_path}-{args.max_length}-{args.chat_template}-{args.target_model_path}"
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    train_eagle3_dataset = build_eagle3_dataset(
        dataset=train_dataset, tokenizer=tokenizer, chat_template=args.chat_template,
        max_length=args.max_length, is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key, num_proc=args.build_dataset_num_proc,
    )

    min_loss_tokens = 2 * args.block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(lambda x: x["loss_mask"].sum() >= min_loss_tokens)
    print_on_rank0(f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples")

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset, args.batch_size, num_workers=args.dataloader_num_workers,
        shuffle=True, process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset, tokenizer=tokenizer, chat_template=args.chat_template,
            max_length=args.max_length, is_preformatted=args.is_preformatted,
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset, args.batch_size, num_workers=args.dataloader_num_workers,
            shuffle=False, process_group=get_dp_group(),
        )
    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, dspark_model, draft_model, optimizer):
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(dspark_model, StateDictType.FULL_STATE_DICT):
        state_dict = dspark_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v for k, v in state_dict.items() if "draft_model." in k
        }
        if dist.get_rank() == 0:
            torch.save({"epoch": epoch, "global_step": step, "args": args, **optimizer.state_dict()},
                       os.path.join(save_dir, "training_state.pt"))
            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)
            for src_name in ["dspark.py", "markov_head.py", "dflash.py"]:
                src = os.path.join(os.path.dirname(__file__), "..", "specforge", "modeling", "draft", src_name)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(save_dir, src_name))
            print_on_rank0(f"Saved checkpoint to {save_dir}")
    dist.barrier()


def record_metrics(args, loss, accuracy, global_step, tracker, optimizer, train_dataloader=None,
                   mode="train", pos_metrics=None, accept_len=None, extra_losses=None):
    logdict = {}
    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()
    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    if accept_len is not None:
        logdict[f"{mode}/accept_len"] = accept_len
    if extra_losses:
        for k, v in extra_losses.items():
            logdict[f"{mode}/{k}"] = v
    if pos_metrics is not None:
        for k in range(1, len(pos_metrics["pos_acc"])):
            logdict[f"pos_train/acc_draft{k}"] = pos_metrics["pos_acc"][k].item()
            logdict[f"pos_train/loss_draft{k}"] = pos_metrics["pos_loss"][k].item()

    print_msg = f"{mode.capitalize()} - Step {global_step}, Loss: {loss:.4f}, Acc: {accuracy:.4f}"
    if accept_len is not None:
        print_msg += f", AcceptLen: {accept_len:.2f}"
    print_on_rank0(print_msg)
    tracker.log(logdict, step=global_step)


def main():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    warnings.filterwarnings("ignore", "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed")

    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)

    draft_model_last_checkpoint = None
    ckpt_info = (0, 0)
    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint, ckpt_info = get_last_checkpoint(args.output_dir)

    if draft_model_last_checkpoint:
        checkpoint_config_path = os.path.join(draft_model_last_checkpoint, "config.json")
        if os.path.exists(checkpoint_config_path):
            args.draft_config_path = checkpoint_config_path

    target_model, draft_model = build_models(args)

    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = DSparkDraftModel.from_pretrained(draft_model_last_checkpoint, torch_dtype=torch.bfloat16)
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        training_state_path = os.path.join(draft_model_last_checkpoint, "training_state.pt")
        if os.path.exists(training_state_path):
            resume_state = torch.load(training_state_path, map_location="cpu", weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    draft_model.mask_token_id = mask_token_id
    # A saved checkpoint has two independent readers:
    #   * DSparkDraftModel (reload / resume here) reads dflash_config;
    #   * sglang's parse_dspark_draft_config (serving) reads dspark_config and has
    #     no dflash_config fallback for markov_rank / markov_head_type.
    # So mirror the serving-relevant fields into both.
    dflash_config = getattr(draft_model.config, "dflash_config", None) or {}
    dflash_config["mask_token_id"] = mask_token_id
    dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    draft_model.config.dflash_config = dflash_config

    dspark_config = getattr(draft_model.config, "dspark_config", None) or {}
    dspark_config["mask_token_id"] = mask_token_id
    dspark_config["target_layer_ids"] = draft_model.target_layer_ids
    dspark_config["markov_rank"] = dflash_config.get("markov_rank", args.markov_rank)
    dspark_config["markov_head_type"] = dflash_config.get("markov_head_type", args.markov_head_type)
    draft_model.config.dspark_config = dspark_config

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)
    steps_per_epoch = math.ceil(len(train_dataloader) / args.accumulation_steps)
    total_steps = args.num_epochs * steps_per_epoch

    if args.save_schedule:
        batches_per_epoch = len(train_dataloader)
        save_steps = {int(float(f) * batches_per_epoch) for f in args.save_schedule.split(",")}
        save_steps.discard(0)

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path, embed_key=args.embedding_key, lm_head_key=args.lm_head_key,
        device="cuda", trust_remote_code=args.trust_remote_code,
    )

    dspark_model = OnlineDSparkModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        ce_loss_alpha=args.ce_loss_alpha,
        l1_loss_alpha=args.l1_loss_alpha,
        kl_loss_alpha=args.kl_loss_alpha,
        confidence_head_alpha=args.confidence_head_alpha,
        position_weighting=args.position_weighting,
        dpace_alpha=args.dpace_alpha,
    )

    # FSDP
    fsdp_kwargs = dict(
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    block_names = set(getattr(draft_model, "_no_split_modules", None) or [])
    block_classes = {type(m) for m in dspark_model.modules() if type(m).__name__ in block_names}
    if block_classes:
        fsdp_kwargs["auto_wrap_policy"] = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=block_classes)
    dspark_model = FSDP(dspark_model, **fsdp_kwargs)

    start_epoch = ckpt_info[0]
    global_step = ckpt_info[1]
    optimizer = BF16Optimizer(draft_model, lr=args.learning_rate, max_grad_norm=args.max_grad_norm,
                              warmup_ratio=args.warmup_ratio, total_steps=total_steps)

    if resume_state is not None:
        optimizer.scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        del resume_state

    skip_steps = global_step - start_epoch * len(train_dataloader)
    capture_last_hidden = needs_target_last_hidden(args, draft_model)
    tracker = create_tracker(args, args.output_dir)
    last_time = time.time()
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        draft_model.train()

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=True) if dist.get_rank() == 0 else train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            global_step += 1

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            target_output = target_model.generate_dflash_data(input_ids, attention_mask, loss_mask)
            hidden_states = target_output.hidden_states.cuda()

            # Get last hidden states for the L1/KL losses and for the confidence
            # BCE target (detach to avoid backprop through the target)
            target_last_hidden = None
            if capture_last_hidden and target_output.last_hidden_state is not None:
                target_last_hidden = target_output.last_hidden_state.detach()
            del target_output  # free target model outputs

            loss, accuracy, pos_metrics = dspark_model(
                input_ids=input_ids, hidden_states=hidden_states,
                loss_mask=loss_mask, target_last_hidden_states=target_last_hidden,
            )

            (loss / args.accumulation_steps).backward()

            if global_step % args.accumulation_steps == 0:
                optimizer.step()

            if global_step % args.log_interval == 0:
                loss_log = loss.clone()
                acc_log = accuracy.clone()
                dist.all_reduce(loss_log)
                dist.all_reduce(acc_log)
                loss_log /= dist.get_world_size()
                acc_log /= dist.get_world_size()

                accept_len_log = pos_metrics["accept_len"].clone()
                dist.all_reduce(accept_len_log)
                accept_len_log /= dist.get_world_size()

                record_metrics(args, loss_log.item(), acc_log.item(), global_step // args.accumulation_steps,
                               tracker, optimizer, train_dataloader, mode="train",
                               pos_metrics=pos_metrics, accept_len=accept_len_log.item(),
                               extra_losses={"ce_loss": pos_metrics["ce_loss"].item(),
                                             "l1_loss": pos_metrics["l1_loss"].item(),
                                             "kl_loss": pos_metrics["kl_loss"].item(),
                                             # wandb key kept as conf_loss so the
                                             # already-running cells stay comparable
                                             "conf_loss": pos_metrics["confidence_loss"].item()})

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{accuracy.item():.4f}", "iter_time": f"{elapsed:.2f}s"})

            if args.save_schedule:
                should_save = global_step in save_steps
            else:
                should_save = global_step % args.save_interval == 0
            if should_save:
                save_checkpoint(args, epoch, global_step, dspark_model, draft_model, optimizer)

    save_checkpoint(args, args.num_epochs, global_step, dspark_model, draft_model, optimizer)
    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
