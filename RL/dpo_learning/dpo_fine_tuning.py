## ref:https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb
import os
import gc
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Union, Optional

import json
import numpy as np
from sympy import EX
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import get_scheduler
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch.distributed as dist
import transformers

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers import GPT2Tokenizer, GPT2Model,  GPT2LMHeadModel
import math
import torch.nn.functional as F
import copy


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text

def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    ## 随机打印一个数据
    model_input = format_input(data[50])
    print(model_input)


    desired_response = f"### Response:\n{data[50]['chosen']}"
    print(desired_response)


    possible_response = f"### Response:\n{data[50]['rejected']}"
    print(possible_response)

    return data

class ModelModule(pl.LightningModule):

    def __init__(self, training_args, tokenizer, model):
        super().__init__()
        self.save_hyperparameters()
        self.training_args = training_args
        self.tokenizer = tokenizer
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        refer_model = copy.deepcopy(model)
        refer_model.eval()
        for name, param in refer_model.named_parameters():
            param.requires_grad = False
        self.model = model
        self.refer_model = refer_model
    
    def training_step(self, batch, batch_idx):
        policy_chosen_logits = self.model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
        policy_chosen_log_probas = compute_logprobs(
            logits=policy_chosen_logits,
            labels=batch["chosen_input_ids"],
            selection_mask=batch["chosen_attention_mask"]
        )
        policy_rejected_logits = self.model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits
        policy_rejected_log_probas = compute_logprobs(
            logits=policy_rejected_logits,
            labels=batch["rejected_input_ids"],
            selection_mask=batch["rejected_attention_mask"]
        )
        with torch.no_grad():
            ref_chosen_logits = self.refer_model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
            ref_chosen_log_probas = compute_logprobs(
                logits=ref_chosen_logits,
                labels=batch["chosen_input_ids"],
                selection_mask=batch["chosen_attention_mask"]
            )
            ref_rejected_logits = self.refer_model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits
            ref_rejected_log_probas = compute_logprobs(
                logits=ref_rejected_logits,
                labels=batch["rejected_input_ids"],
                selection_mask=batch["rejected_attention_mask"])
        
        loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
            model_chosen_logprobs=policy_chosen_log_probas,
            model_rejected_logprobs=policy_rejected_log_probas,
            reference_chosen_logprobs=ref_chosen_log_probas,
            reference_rejected_logprobs=ref_rejected_log_probas,
            beta=self.training_args.beta
        )
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/chosen_rewards', chosen_rewards, prog_bar=True)
        self.log('train/rejected_rewards', rejected_rewards, prog_bar=True)
        self.log('train/train_reward_margin', chosen_rewards-rejected_rewards, prog_bar=True)
        self.log('lr', self.lr_schedulers().get_lr()[0], prog_bar=True, logger=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        policy_chosen_logits = self.model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
        policy_chosen_log_probas = compute_logprobs(
            logits=policy_chosen_logits,
            labels=batch["chosen_input_ids"],
            selection_mask=batch["chosen_attention_mask"]
        )
        policy_rejected_logits = self.model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits
        policy_rejected_log_probas = compute_logprobs(
            logits=policy_rejected_logits,
            labels=batch["rejected_input_ids"],
            selection_mask=batch["rejected_attention_mask"]
        )
        with torch.no_grad():
            ref_chosen_logits = self.refer_model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
            ref_chosen_log_probas = compute_logprobs(
                logits=ref_chosen_logits,
                labels=batch["chosen_input_ids"],
                selection_mask=batch["chosen_attention_mask"]
            )
            ref_rejected_logits = self.refer_model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits
            ref_rejected_log_probas = compute_logprobs(
                logits=ref_rejected_logits,
                labels=batch["rejected_input_ids"],
                selection_mask=batch["rejected_attention_mask"])
        
        loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
            model_chosen_logprobs=policy_chosen_log_probas,
            model_rejected_logprobs=policy_rejected_log_probas,
            reference_chosen_logprobs=ref_chosen_log_probas,
            reference_rejected_logprobs=ref_rejected_log_probas,
            beta=self.training_args.beta
        )

        self.validation_step_outputs.append(loss.item())
        
        return loss

    def on_validation_epoch_end(self):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, self.validation_step_outputs)
            # gathered_outputs = sum(gathered_outputs, [])
            gathered_outputs = np.array(gathered_outputs).reshape(-1)
        else:
            gathered_outputs = self.validation_step_outputs
            gathered_outputs = np.array(gathered_outputs).reshape(-1)
        self.validation_step_outputs.clear()
        
        valid_loss = [np.inf]
        if self.trainer.is_global_zero:
            valid_loss = [gathered_outputs.mean()]
        
        dist.broadcast_object_list(valid_loss)

        self.log('val/loss', valid_loss[0], prog_bar=True, rank_zero_only=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        policy_chosen_logits = self.model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
        policy_chosen_log_probas = compute_logprobs(
            logits=policy_chosen_logits,
            labels=batch["chosen_input_ids"],
            selection_mask=batch["chosen_attention_mask"]
        )
        policy_rejected_logits = self.model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits
        policy_rejected_log_probas = compute_logprobs(
            logits=policy_rejected_logits,
            labels=batch["rejected_input_ids"],
            selection_mask=batch["rejected_attention_mask"]
        )
        with torch.no_grad():
            ref_chosen_logits = self.refer_model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"]).logits
            ref_chosen_log_probas = compute_logprobs(
                logits=ref_chosen_logits,
                labels=batch["chosen_input_ids"],
                selection_mask=batch["chosen_attention_mask"]
            )
            ref_rejected_logits = self.refer_model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"]).logits
            ref_rejected_log_probas = compute_logprobs(
                logits=ref_rejected_logits,
                labels=batch["rejected_input_ids"],
                selection_mask=batch["rejected_attention_mask"])
        
        loss, chosen_rewards, rejected_rewards = compute_dpo_loss(
            model_chosen_logprobs=policy_chosen_log_probas,
            model_rejected_logprobs=policy_rejected_log_probas,
            reference_chosen_logprobs=ref_chosen_log_probas,
            reference_rejected_logprobs=ref_rejected_log_probas,
            beta=self.training_args.beta
        )

        self.test_step_outputs.append(loss.item())
        return loss

    def on_test_epoch_end(self):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, self.test_step_outputs)
            # gathered_outputs = sum(gathered_outputs, [])
            gathered_outputs = np.array(gathered_outputs).reshape(-1)
        else:
            gathered_outputs = self.test_step_outputs
            gathered_outputs = np.array(gathered_outputs).reshape(-1)
            
        self.test_step_outputs.clear()
        
        test_loss = [np.inf]
        if self.trainer.is_global_zero:
            test_loss = [gathered_outputs.mean()]
        
        dist.broadcast_object_list(test_loss)

        self.log('test/loss', test_loss[0], prog_bar=True, rank_zero_only=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        ## TODO
        pass    
    
    def configure_optimizers(self):
        num_training_steps = self.trainer.num_training_steps
        self.print(f'Num training steps: {num_training_steps}')
        num_warmup_steps = int(num_training_steps * self.training_args.warmup_ratio)
        # parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.training_args.learning_rate, weight_decay=self.training_args.weight_decay)
        scheduler = get_scheduler(self.training_args.scheduler, optimizer, num_warmup_steps, num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath=None) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)
        return filepath


class DataModule(pl.LightningDataModule):
    def __init__(self, training_args, tokenizer=None):
        super().__init__()
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_len = self.training_args.max_len 
    
    def custom_train_collate_fn(self, batch):
        self.tokenizer.padding_side = "right"
        prompt = [_["prompt"] for _ in batch]
        chosen = [_["chosen"] for _ in batch]
        rejected = [_["rejected"] for _ in batch]

        prompt_result = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=self.max_len)
        prompt_input_ids = prompt_result["input_ids"]
        prompt_attention_mask = prompt_result["attention_mask"]

        chosen_result = self.tokenizer(chosen, return_tensors="pt", padding=True, max_length=self.max_len)
        chosen_input_ids = chosen_result["input_ids"]
        chosen_attention_mask = chosen_result["attention_mask"]
        ## mask prompt
        chosen_attention_mask[:,:prompt_attention_mask.shape[-1]][prompt_attention_mask==1]=0

        rejected_result = self.tokenizer(rejected, return_tensors="pt", padding=True, max_length=self.max_len)
        rejected_input_ids = rejected_result["input_ids"]
        rejected_attention_mask = rejected_result["attention_mask"]
        ## mask prompt
        rejected_attention_mask[:,:prompt_attention_mask.shape[-1]][prompt_attention_mask==1]=0
        return {
            "prompt_input_ids":prompt_input_ids,
            "prompt_attention_mask":prompt_attention_mask,
            "chosen_input_ids":chosen_input_ids,
            "chosen_attention_mask":chosen_attention_mask,
            "rejected_input_ids":rejected_input_ids,
            "rejected_attention_mask":rejected_attention_mask,
        }
    
    def custom_test_collate_fn(self, batch):
        
        ## 如果是generate请选择left padding
        # self.tokenizer.padding_side = "left"
        ## 如果算loss用请选择right padding
        return self.custom_train_collate_fn(batch)


    def prepare_data(self):
        ## 导入数据
        data = load_data(self.training_args.data_path)
        
        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]

            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            
            self.encoded_texts.append({
                "prompt": prompt,
                "chosen": chosen_full_text,
                "rejected": rejected_full_text,
            })
        
        train_portion = int(len(data) * 0.85)
        test_portion = int(len(data) * 0.1)
        val_portion = len(data) - train_portion - test_portion

        self.train_dataset = self.encoded_texts[:train_portion]
        self.test_dataset = self.encoded_texts[train_portion + val_portion:]
        self.val_dataset = self.encoded_texts[train_portion:train_portion + val_portion]
    
    @property
    def pad_id(self):
        return self.tokenizer[self.training_args.format].PAD_ID

    def print_stats(self):
        ## 打印数据集
        print(f'Train dataset: {len(self.train_dataset)}')
        print(f'Valid dataset: {len(self.val_dataset)}')
        print(f'Test dataset: {len(self.test_dataset)}')

    def setup(self, stage: str = None):  # 必须包含 stage 参数
        pass
        # if stage == "fit" or stage is None:
        #     # 加载训练/验证数据
        #     self.train_dataset = ...
        #     self.val_dataset = ...
        # if stage == "test" or stage is None:
        #     # 加载测试数据
        #     self.test_dataset = ...

    def train_dataloader(self):
        if (self.training_args.debug is False):
            return torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.training_args.batch_size, num_workers=self.training_args.num_workers,
                collate_fn=self.custom_train_collate_fn,
                shuffle=True)
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.training_args.batch_size, num_workers=0,
                collate_fn=self.custom_train_collate_fn, shuffle=True)

    def val_dataloader(self):
        if (self.training_args.debug is False):
            return torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.training_args.batch_size*2, num_workers=self.training_args.num_workers,
                collate_fn=self.custom_test_collate_fn)
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.training_args.batch_size*2, num_workers=0,
                collate_fn=self.custom_test_collate_fn)

    def test_dataloader(self):
        if (self.training_args.debug is False):
            return torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.training_args.batch_size*2, num_workers=self.training_args.num_workers,
                collate_fn=self.custom_test_collate_fn)
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                    self.test_dataset, batch_size=self.training_args.batch_size*2, num_workers=0,
                    collate_fn=self.custom_test_collate_fn)

def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load a pre-trained tokenizer.
    Args:
        model_name (str): The name or path of the pre-trained model.
    Returns:
        PreTrainedTokenizer: The tokenizer associated with the specified model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def load_model(model_name: str) -> PreTrainedModel:
    """
    Loads a pre-trained encoder-decoder model.
    Args:
        model_name (str): The name or path of the pre-trained model to load.
    Returns:
        PreTrainedModel: The loaded pre-trained model
    """

    model = GPT2LMHeadModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    return model

def compute_logprobs(logits, labels, selection_mask=None):
    """
    Compute log probabilities.

    Args:
      logits: Tensor of shape (batch_size, num_tokens, vocab_size)
      labels: Tensor of shape (batch_size, num_tokens)
      selection_mask: Tensor for shape (batch_size, num_tokens)

    Returns:
      mean_log_prob: Mean log probability excluding padding tokens.
    """

    # Labels are the inputs shifted by one
    labels = labels[:, 1:].clone()

    # Truncate logits to match the labels num_tokens
    logits = logits[:, :-1, :]

    log_probs = F.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the actual labels
    selected_log_probs = torch.gather(
        input=log_probs,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()

        # Apply the mask to filter out padding tokens
        selected_log_probs = selected_log_probs * mask

        # Calculate the average log probability excluding padding tokens
        # This averages over the tokens, so the shape is (batch_size,)
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)

        return avg_log_prob

    else:
        return selected_log_probs.mean(-1)

def compute_dpo_loss(
      model_chosen_logprobs,
      model_rejected_logprobs,
      reference_chosen_logprobs,
      reference_rejected_logprobs,
      beta=0.1,
    ):
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logprobs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logprobs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logprobs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logprobs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss; typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.

    Returns:
        A tuple of three tensors: (loss, chosen_rewards, rejected_rewards).
    """

    model_logratios = model_chosen_logprobs - model_rejected_logprobs
    reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_logratios - reference_logratios

    # DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(beta * logits)

    # Optional values to track progress during training
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    # .mean() to average over the samples in the batch
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


@dataclass
class TrainingArguments:
    data_path : str = "instruction-data-with-preference.json"
    model_name : str = "gpt2"
    
    debug: bool = False
    seed : int = 42
    precision: Union[int, str] ='bf16'
    load_path: Optional[str] = None  # 加载预训练模型的路径（可选）
    ori_model_path : Optional[str] = "./outputs/gpt2_sft/checkpoints/best.ckpt"
    do_train: bool = True  # 是否进行训练
    do_valid: bool = True  # 是否进行验证
    do_test: bool = True  # 是否进行测试
    resume: bool = False # 是否继续进行训练
    warmup_ratio: float = 0.00 # warmup的比例
    weight_decay: float = 0.0001 #L2 正则化
    max_grad_norm: float = 1  # 最大梯度范数（用于梯度裁剪）
    debug: bool = False  # 是否启用调试模式

    save_dir: str = "./outputs/gpt2_dpo"  # 模型保存路径
    gpus: int = 1  # 使用的GPU数量
    epochs: int = 10  # 训练轮数
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    eval_per_epoch: int = 5  # 每多少轮评估一次
    batch_size: int = 16  # 批量大小
    scheduler: str = "cosine"  # 学习率调度器（如 "linear", "cosine"）
    learning_rate: float = 1e-5  # 学习率
    warmup_ratio: float = 0.00 # warmup的比例
    num_workers: int = 4
    max_len: int = 512
    beta: float = 0.1


if __name__ == "__main__":
    training_args = TrainingArguments(epochs=50, debug=True, learning_rate=1e-4)

    if training_args.debug is True:
        training_args.gpus = 1

    pl.seed_everything(training_args.seed, workers=True)

    tokenizer = load_tokenizer(training_args.model_name)
    # https://github.com/huggingface/transformers/issues/12594
    tokenizer.pad_token = tokenizer.eos_token
    # Load and format the dataset
    dm = DataModule(training_args, tokenizer=tokenizer)
    dm.prepare_data()
    dm.print_stats()

    try:
        import tokenizers
        ## torch2.6 引入的新机制
        torch.serialization.add_safe_globals(
            [
                transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast,
                TrainingArguments,
                tokenizers.Tokenizer,
                tokenizers.models.Model,
                transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel,
                transformers.models.gpt2.modeling_gpt2.GPT2Model,
                torch.nn.modules.sparse.Embedding,
                torch.nn.modules.dropout.Dropout,
                torch.nn.modules.container.ModuleList,
                transformers.models.gpt2.modeling_gpt2.GPT2Block,
                torch.nn.modules.normalization.LayerNorm,
                transformers.models.gpt2.modeling_gpt2.GPT2Attention,
                transformers.pytorch_utils.Conv1D,
                transformers.models.gpt2.configuration_gpt2.GPT2Config,
                transformers.models.gpt2.modeling_gpt2.GPT2MLP,
                transformers.activations.NewGELUActivation,
                torch.nn.modules.linear.Linear,
                transformers.generation.configuration_utils.GenerationConfig,
                transformers.generation.configuration_utils.CompileConfig,
            ]
        )
    except:
        pass

    ## 原始模型
    ori_model = load_model(training_args.model_name)
    if training_args.ori_model_path is not None and os.path.exists(training_args.ori_model_path):
        checkpoint = torch.load(training_args.ori_model_path)
        ori_model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("load form `ori_model_path` successfully")

    ## 模型
    model = ModelModule(training_args, tokenizer, ori_model)
    if training_args.load_path is not None and os.path.exists(training_args.load_path):
        model = ModelModule.load_from_checkpoint(training_args.load_path, strict=False, training_args=training_args, tokenizer=tokenizer)
        print("load form `load_path` successfully")

    checkpoint = ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=1, filename='best', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    os.makedirs(training_args.save_dir, exist_ok=True)

    logger = pl.loggers.TensorBoardLogger(training_args.save_dir, name='', version='')

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=False),
        accelerator='gpu',
        precision = training_args.precision,
        devices=training_args.gpus,
        logger=logger,
        default_root_dir=training_args.save_dir,
        callbacks=[checkpoint, lr_monitor],
        max_epochs=training_args.epochs,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        check_val_every_n_epoch=training_args.eval_per_epoch,
        log_every_n_steps=50,
        deterministic='warn')

    if training_args.do_train:
        trainer.num_training_steps = math.ceil(
            len(dm.train_dataset) / (training_args.batch_size * training_args.gpus * training_args.gradient_accumulation_steps)) * training_args.epochs
        model.eval_dataset = dm.val_dataset
        ckpt_path = os.path.join(training_args.save_path, 'checkpoints/last.ckpt') if training_args.resume else None
        ## 一定要先验证，不然指标不对等于白做
        trainer.validate(model, datamodule=dm)
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        ## 导入最优的模型
        model = ModelModule.load_from_checkpoint(checkpoint.best_model_path, training_args=training_args, tokenizer=tokenizer, model=ori_model)

    if training_args.do_valid:
        model.eval_dataset = dm.val_dataset
        trainer.validate(model, datamodule=dm)

    if training_args.do_test:
        model.eval_dataset = dm.test_dataset
        trainer.test(model, datamodule=dm)
