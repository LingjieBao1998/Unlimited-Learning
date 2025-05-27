import gc
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import evaluate
import nltk
import torch
print("torch.cuda.is_bf16_supported():",torch.cuda.is_bf16_supported())  # 如果返回 False，则不支持
from datasets import Dataset, DatasetDict, load_dataset
from torch import FloatTensor, LongTensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
)

import os
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import get_scheduler
from pytorch_lightning.strategies.ddp import DDPStrategy
import math
from typing import Optional
import torch.distributed as dist
import json
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, PeftModel
rouge_eval = evaluate.load("rouge")


@dataclass
class TrainingArguments:
    model_name: str = "google/pegasus-cnn_dailymail"  # 模型名称
    seed: int = 42  # 随机种子
    epochs: int = 1  # 训练轮数
    num_workers: int = 4  # 数据加载线程数
    batch_size: int = 4  # 批量大小
    learning_rate: float = 1e-5  # 学习率
    update_old_after: int = 1000  # 更新旧策略的步数
    group_size: int = 5  # 组大小（GRPO相关）
    logging_steps: int = 10  # 日志记录步数
    max_new_tokens: int = 128  # 生成的最大token数
    max_document_length: int = 512  # 文档最大长度
    max_summary_length: int = 128  # 摘要最大长度
    grpo_epsilon: float = 0.1  # GRPO的epsilon参数
    grpo_beta: float = 0.04  # GRPO的beta参数
    gradient_max_norm: float = 0.2  # 梯度裁剪的最大范数
    save_steps: int = 100  # 保存模型的步数
    scheduler: str = "linear"  # 学习率调度器（如 "linear", "cosine"）
    save_dir: str = "./grpo_pegasus-cnn-dailymail"  # 模型保存路径
    debug: bool = False  # 是否启用调试模式
    eval_per_epoch: int = 5  # 每多少轮评估一次
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    max_grad_norm: float = 0.1  # 最大梯度范数（用于梯度裁剪）
    precision: str = 16  # 精度（如 16 或 "bf16" 或 32）
    gpus: int = 1  # 使用的GPU数量
    load_path: Optional[str] = None  # 加载预训练模型的路径（可选）
    do_train: bool = True  # 是否进行训练
    do_valid: bool = True  # 是否进行验证
    do_test: bool = True  # 是否进行测试
    resume: bool = False # 是否继续进行训练
    warmup_ratio: float = 0.00 # warmup的比例
    weight_decay: float = 0.0001 #L2 正则化

    # refer：https://github.com/YanSte/NLP-LLM-Fine-tuning-QA-LoRA-T5/blob/main/nlp-llm-fine-tuning-lora-t5-l.ipynb
    use_lora : bool = False
    lora_task_type = TaskType.SEQ_2_SEQ_LM
    lora_inference_mode = False
    lora_r = 8
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules=["q_proj", "v_proj", "k_proj"]


@dataclass
class BatchRewards:
    rewards: FloatTensor


@dataclass
class GRPOOutput:
    loss: FloatTensor
    reward: FloatTensor
    kl: FloatTensor


def load_model(model_name: str) -> PreTrainedModel:
    """
    Loads a pre-trained encoder-decoder model.
    Args:
        model_name (str): The name or path of the pre-trained model to load.
    Returns:
        PreTrainedModel: The loaded pre-trained model
    """

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    )
    model = model.to("cuda")
    return model


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


def load_cnndm() -> DatasetDict:
    """
    Load and preprocess the CNN/DailyMail dataset.
    Returns:
        DatasetDict: A dictionary containing the preprocessed dataset splits.
    """

    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    for split in dataset.column_names:
        dataset[split] = dataset[split].remove_columns(["id"])
        dataset[split] = dataset[split].rename_columns(
            {"article": "document", "highlights": "summary"}
        )
    return dataset


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_document_length: int,
    max_summary_length: int,
) -> DatasetDict:
    """
    Tokenizes the documents and summaries in the dataset.
    Args:
        dataset (DatasetDict): The dataset containing documents and summaries to be tokenized.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for tokenizing the text.
        max_document_length (int): The maximum length for tokenized documents.
        max_summary_length (int): The maximum length for tokenized summaries.
    Returns:
        DatasetDict: The tokenized dataset with documents and summaries replaced by their tokenized versions.
    """

    def tokenize_function(example):
        model_inputs = tokenizer(
            example["document"],
            max_length=max_document_length,
            truncation=True,
        )
        labels = tokenizer(
            example["summary"],
            max_length=max_summary_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        tokenize_function, batched=True, remove_columns=["document", "summary"]
    )


def rouge_reward(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the average ROUGE (1, 2, Lsum) scores for a set of predictions and references.
    Args:
        predictions (list[str]): A list of predicted text strings.
        references (list[str]): A list of reference text strings.
    Returns:
        float: The average ROUGE score (ROUGE-1, ROUGE-2, and ROUGE-Lsum).
    """

    scores = rouge_eval.compute(predictions=predictions, references=references)
    return (scores["rouge1"] + scores["rouge2"] + scores["rougeLsum"]) / 3.0


def postprocess_text(
    preds: list[str], labels: list[str]
) -> Tuple[list[str], list[str]]:
    """
    Post-processes the predicted and label texts,
    formatting them for ROUGE-L summarization evaluation.
    Args:
        preds (list[str]): List of predicted text strings.
        labels (list[str]): List of label text strings.
    Returns:
        Tuple[list[str], list[str]]: A tuple containing two lists:
            - The first list contains the post-processed predicted texts.
            - The second list contains the post-processed label texts.
    """

    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = [
        "\n".join(nltk.sent_tokenize(pred.replace("<n>", " ")))
        for pred in preds
    ]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_rewards(
    token_ids: LongTensor, labels: LongTensor, tokenizer: PreTrainedTokenizer
) -> BatchRewards:
    """
    Compute rewards based on the ROUGE avg score between generated completions and reference summaries.
    Args:
        token_ids (LongTensor): Tensor containing token IDs of the generated completions.
        labels (LongTensor): Tensor containing token IDs of the reference summaries.
        tokenizer (PreTrainedTokenizer): Tokenizer used to decode the token IDs.
    Returns:
        BatchRewards: A tensor containing the computed rewards for each completion.
    """
    labels[labels == -100] = tokenizer.pad_token_id
    completions = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    summaries = tokenizer.batch_decode(labels, skip_special_tokens=True)
    completions, summaries = postprocess_text(completions, summaries)

    rewards = torch.zeros(token_ids.shape[0], device=token_ids.device)
    for idx, (completion, summary) in enumerate(zip(completions, summaries)):
        rouge_score = rouge_reward(
            predictions=[completion], references=[summary]
        )
        rewards[idx] = rouge_score

    return BatchRewards(rewards)


def selective_log_softmax(
    logits: FloatTensor, index: LongTensor
) -> FloatTensor:
    """
    Computes the log softmax of the input logits selectively based on the provided indices.
    This function performs the same operation as applying `log_softmax` on the logits tensor
    along the last dimension and then gathering the results based on the provided indices.
    However, it processes the logits row by row to save memory by leveraging PyTorch internals.
    Taken from https://www.tylerromero.com/posts/2025-02-selective-log-softmax/
    Args:
        logits (FloatTensor): A tensor of shape (batch_size, num_classes) containing the raw
                              logits for each class.
        index (LongTensor): A tensor of shape (batch_size, num_indices) containing the indices
                            of the classes for which to compute the log softmax.
    Returns:
        FloatTensor: A tensor of shape (batch_size, num_indices) containing the log softmax
                     values for the specified indices.
    """

    token_logprobs = []
    for logits_row, index_row in zip(logits, index):
        logprobs_row = logits_row.log_softmax(dim=-1)
        token_logprobs_row = torch.gather(
            logprobs_row, dim=-1, index=index_row.unsqueeze(-1)
        ).squeeze(-1)
        token_logprobs.append(token_logprobs_row)
    return torch.stack(token_logprobs)


def gather_token_scores(
    logits: FloatTensor, generated_ids: LongTensor
) -> FloatTensor:
    """
    Gathers token scores from logits based on generated token IDs.
    Args:
        logits (FloatTensor): The logits output from the model. It can be a tuple of tensors or a single tensor.
        generated_ids (LongTensor): The IDs of the generated tokens.
    Returns:
        FloatTensor: The token scores after applying a selective log softmax on the logits.
    """

    if isinstance(logits, tuple):
        # Stack the logits (batch_size*group_size, output_length, vocab)
        logits = torch.stack(logits, axis=0).permute((1, 0, 2))

    # Logsoftmax the logits
    token_scores = selective_log_softmax(logits, generated_ids)

    return token_scores


def compute_token_scores(
    model: PreTrainedModel,
    encoder_input_ids: LongTensor,
    encoder_attention_mask: LongTensor,
    decoder_input_ids: LongTensor,
    decoder_attention_mask: LongTensor,
    batch_size: int,
    group_size: int,
) -> FloatTensor:
    """
    Computes token scores for a given batch of input sequences using a pre-trained model.
    Args:
        model (PreTrainedModel): The pre-trained model to use for generating logits.
        encoder_input_ids (LongTensor): Tensor containing input IDs for the encoder.
        encoder_attention_mask (LongTensor): Tensor containing attention masks for the encoder inputs.
        decoder_input_ids (LongTensor): Tensor containing input IDs for the decoder.
        decoder_attention_mask (LongTensor): Tensor containing attention masks for the decoder inputs.
        batch_size (int): The size of the batch.
        group_size (int): The size of the group.
    Returns:
        FloatTensor: A tensor containing the computed token scores, reshaped to (batch_size, group_size, -1).
    """
    ## 常规的forward过程
    logits = model(
        input_ids=encoder_input_ids,
        attention_mask=encoder_attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
    ).logits # [batch_size * training_args.group_size,  decoder_input_ids.shape[0], num_vocab] 
    
    ## 这里可以理解为计算-loss(损失函数的负数)
    scores = gather_token_scores(logits[:, :-1], decoder_input_ids[:, 1:]) 
    scores = scores.view(batch_size, group_size, -1)
    del logits
    torch.cuda.empty_cache()
    return scores


def grpo(
    generated_ids: LongTensor,
    old_scores: FloatTensor,
    current_scores: FloatTensor,
    reference_scores: FloatTensor,
    labels: LongTensor,
    tokenizer: PreTrainedTokenizer,
    epsilon: float,
    beta: float,
) -> GRPOOutput:
    """
    Compute the loss of Group Relative Policy Optimization (GRPO) on the given inputs.
    Args:
        generated_ids (LongTensor): Tensor of generated token IDs.
        old_scores (FloatTensor): Tensor of old policy scores.
        current_scores (FloatTensor): Tensor of current policy scores.
        reference_scores (FloatTensor): Tensor of reference policy scores.
        truths (LongTensor): Tensor of ground truth token IDs.
        tokenizer (PreTrainedTokenizer): Tokenizer used for encoding/decoding.
        epsilon (float): Clipping parameter for policy ratios.
        beta (float): Weighting factor for the Kullback-Leibler divergence term.
    Returns:
        GRPOOutput: A dataclass containing the mean loss, rewards and KL divergences.
    """
    losses = torch.zeros(generated_ids.shape[0])
    rewards = torch.zeros(generated_ids.shape[0])
    kls = torch.zeros(generated_ids.shape[0])

    for idx, (
        group_ids,
        group_labels,
        group_old_scores,
        group_current_scores,
        group_reference_scores,
    ) in enumerate(
        zip(generated_ids, labels, old_scores, current_scores, reference_scores)
    ):
        # Compute advantages
        group_rewards = compute_rewards(group_ids, group_labels, tokenizer) #计算奖励
        mean = group_rewards.rewards.mean()
        centered = group_rewards.rewards - mean
        std = group_rewards.rewards.std()
        if std < 1e-8:
            advantages = torch.zeros_like(centered)
        else:
            advantages = centered / (std + 1e-8)

        # Store the mean of each rewards for the group
        rewards[idx] = group_rewards.rewards.mean()

        # Compute the ratios
        ratios = torch.exp(group_current_scores - group_old_scores)

        # Compute the clipped ratios
        clipped_ratios = torch.clamp(
            ratios, min=1.0 - epsilon, max=1.0 + epsilon
        )

        # Compute kullback-leibler divergence between reference and current policy
        kl = (
            torch.exp(group_reference_scores - group_current_scores)
            - (group_reference_scores - group_current_scores)
            - 1
        )
        kls[idx] = kl.mean()

        # Compute mean loss of the group
        completion_mask = group_ids[:, 1:] != tokenizer.pad_token_id
        loss = (
            torch.min(
                ratios * advantages.unsqueeze(-1),
                clipped_ratios * advantages.unsqueeze(-1),
            )
            - beta * kl
        )
        loss = -(loss * completion_mask).sum() / completion_mask.sum()
        losses[idx] = loss

    return GRPOOutput(
        loss=losses.mean(),
        reward=rewards.mean(),
        kl=kls.mean(),
    )

def compare_models(model1, model2, tol=1e-6):
    """
    比较两个模型的所有参数是否相同（允许一定的数值误差）
    
    Args:
        model1 (torch.nn.Module): 第一个模型
        model2 (torch.nn.Module): 第二个模型
        tol (float): 允许的数值误差范围
    
    Returns:
        bool: 是否所有参数都相同
        list: 不同的参数名称（如果有）
    """
    diff_params = []
    
    # 确保两个模型的参数顺序一致
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert name1 == name2, f"参数名称不匹配: {name1} != {name2}"
        
        # 检查形状是否相同
        if param1.shape != param2.shape:
            diff_params.append(name1)
            continue
        
        # 检查数值是否相同（允许一定的误差）
        if not torch.allclose(param1, param2, atol=tol):
            diff_params.append(name1)
    
    return len(diff_params) == 0, diff_params

## 新增DataModule
class DataModule(pl.LightningDataModule):
    def __init__(self, training_args, tokenizer=None):
        super().__init__()
        self.training_args = training_args
        self.tokenizer = tokenizer

    def prepare_data(self):
        ## 导入数据
        dataset = load_cnndm()
        self.dataset = tokenize_dataset(
            dataset,
            self.tokenizer,
            self.training_args.max_document_length,
            self.training_args.max_summary_length,
        )
        
        if self.training_args.debug is False:
            self.train_dataset=self.dataset["train"]
            self.val_dataset=self.dataset["validation"]
            self.test_dataset=self.dataset["test"]
        else:
            ## 为了测试方便，10000条
            self.train_dataset = self.dataset["train"].select(range(10000))
            self.val_dataset=self.dataset["validation"].select(range(100))       
            self.test_dataset=self.dataset["test"].select(range(100))
    
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
            ## 瓶颈不在数据
            # return torch.utils.data.DataLoader(
            #     self.dataset["train"], batch_size=self.training_args.batch_size, num_workers=self.training_args.num_workers,
            #     collate_fn=DataCollatorForSeq2Seq(self.tokenizer), prefetch_factor=2, persistent_workers=True, pin_memory=True,
            #     shuffle=True)
            return torch.utils.data.DataLoader(
                self.dataset["train"], batch_size=self.training_args.batch_size, num_workers=self.training_args.num_workers,
                collate_fn=DataCollatorForSeq2Seq(self.tokenizer),
                shuffle=True)
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                self.train_dataset, batch_size=self.training_args.batch_size, num_workers=0,
                collate_fn=DataCollatorForSeq2Seq(self.tokenizer), shuffle=True)

    def val_dataloader(self):
        if (self.training_args.debug is False):
            return torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.training_args.batch_size*2, num_workers=self.training_args.num_workers,
                collate_fn=DataCollatorForSeq2Seq(self.tokenizer))
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                self.val_dataset, batch_size=self.training_args.batch_size*2, num_workers=0,
                collate_fn=DataCollatorForSeq2Seq(self.tokenizer))

    def test_dataloader(self):
        if (self.training_args.debug is False):
            return torch.utils.data.DataLoader(
                self.test_dataset, batch_size=self.training_args.batch_size*2, num_workers=self.training_args.num_workers,
                collate_fn=DataCollatorForSeq2Seq(self.tokenizer))
        else:
            ## 单线程调试
            return torch.utils.data.DataLoader(
                    self.test_dataset, batch_size=self.training_args.batch_size*2, num_workers=0,
                    collate_fn=DataCollatorForSeq2Seq(self.tokenizer))

class ModelModule(pl.LightningModule):

    def __init__(self, training_args, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.training_args = training_args
        self.tokenizer = tokenizer
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.model, self.old_model, self.reference_model = self.get_model()
    
    def get_model(self):
        ## 导入模型
        model = load_model(self.training_args.model_name)
        reference_model = deepcopy(model)
        reference_model.eval()
        for name, param in reference_model.named_parameters():
            param.requires_grad = False
        if self.training_args.use_lora:
            ## ref:https://www.kaggle.com/code/aayushg1/lora-bart/notebook
            peft_config = LoraConfig(
                r=self.training_args.lora_r, 
                lora_alpha=self.training_args.lora_alpha,
                lora_dropout=self.training_args.lora_dropout,
                task_type=self.training_args.lora_task_type, 
                inference_mode=self.training_args.lora_inference_mode,
                target_modules=self.training_args.lora_target_modules, #["q_proj", "v_proj", "k_proj"],  # 编码器和解码器的注意力层模块名称,#
                bias="none",
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        old_model = deepcopy(model)
        old_model.eval()
        for name, param in old_model.named_parameters():
            param.requires_grad = False
        
        model.train()
        return model, old_model, reference_model
        

    def training_step(self, batch, batch_idx):
        ## 更新模型
        # Update old policy periodically
        if ((self.global_step + 1) * self.trainer.num_devices - 1 ) % self.training_args.update_old_after == 0:
            self.old_model.load_state_dict(self.model.state_dict(), strict=False)
            torch.cuda.empty_cache()


        # Prepare the batch data
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        effective_batch_size = input_ids.shape[0]

        # Generate ids with the old policy
        generated_ids = self.old_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.training_args.max_new_tokens,
            do_sample=True,
            num_beams=self.training_args.group_size, #default=5
            num_return_sequences=self.training_args.group_size,
        ) # [batch_size * training_args.group_size, max_len]

        # Prepare attention mask for computing current
        # and reference logits on the generated ids
        decoder_attention_mask = generated_ids != self.tokenizer.pad_token_id # [batch_size * training_args.group_size, max_len]

        # Interleave input_ids and attention_mask to have
        # the same shape than the generated completions
        repeated_input_ids = input_ids.repeat_interleave(
            repeats=self.training_args.group_size, dim=0
        ) # [batch_size, max_len] -> [batch_size * training_args.group_size, max_len], 将input_ids沿着某个维度进行复制

        repeated_attention_mask = attention_mask.repeat_interleave(
            repeats=self.training_args.group_size, dim=0
        ) # [batch_size, max_len] -> [batch_size * training_args.group_size, max_len], 将attention_mask沿着某个维度进行复制

        with torch.inference_mode():
            old_scores = compute_token_scores(
                self.old_model,
                encoder_input_ids=repeated_input_ids,
                encoder_attention_mask=repeated_attention_mask,
                decoder_input_ids=generated_ids,
                decoder_attention_mask=decoder_attention_mask,
                batch_size=effective_batch_size,
                group_size=self.training_args.group_size,
            ).detach() # [batch_size, training_args.group_size, max_len]
        
        
        self.model.eval()
        current_scores = compute_token_scores(
            self.model,
            encoder_input_ids=repeated_input_ids,
            encoder_attention_mask=repeated_attention_mask,
            decoder_input_ids=generated_ids,
            decoder_attention_mask=decoder_attention_mask,
            batch_size=effective_batch_size,
            group_size=self.training_args.group_size,
        )
        self.model.train()

        with torch.inference_mode():
            reference_scores = compute_token_scores(
                self.reference_model,
                encoder_input_ids=repeated_input_ids,
                encoder_attention_mask=repeated_attention_mask,
                decoder_input_ids=generated_ids,
                decoder_attention_mask=decoder_attention_mask,
                batch_size=effective_batch_size,
                group_size=self.training_args.group_size,
            ).detach() 
        
        # Group the generated ids (batch_size, group_size, output_length)
        generated_ids = generated_ids.view(
            effective_batch_size, self.training_args.group_size, -1
        )

        # Repeat the labels and group (batch_size, group_size)
        labels = labels.repeat_interleave(
            repeats=self.training_args.group_size, dim=0
        ).view(effective_batch_size, self.training_args.group_size, -1)
        
        ## 新旧模型的权重还是一致的
        # Compute GRPO objective
        # 核心
        grpo_output = grpo(
            generated_ids,
            old_scores,
            current_scores,
            reference_scores,
            labels,
            self.tokenizer,
            self.training_args.grpo_epsilon,
            self.training_args.grpo_beta,
        )
        self.log('train/loss', grpo_output.loss, prog_bar=True)
        self.log('train/reward', grpo_output.reward, prog_bar=True)
        self.log('train/kl', grpo_output.kl, prog_bar=True)
        self.log('train/completion_length', decoder_attention_mask.sum(-1).float().mean(), prog_bar=True)
        self.log('lr', self.lr_schedulers().get_lr()[0], prog_bar=True, logger=False)

        # Free GPU memory at the end
        del (
            generated_ids,
            old_scores,
            input_ids,
            attention_mask,
            repeated_input_ids,
            repeated_attention_mask,
            current_scores,
            reference_scores,
            labels,
        )
        torch.cuda.empty_cache()
        gc.collect()
 
        return grpo_output.loss


    def validation_step(self, batch, batch_idx):
        ## 只评价top1
        ## 其他参数，比如topk，topp，temperature可以加入到model.generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=self.training_args.max_new_tokens,
                do_sample=True,
                num_return_sequences=1,
            )
        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        batch["labels"][batch["labels"]==-100]=self.tokenizer.pad_token_id
        label = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        ## 打包保存
        zipped = list(zip(result, label))
        self.validation_step_outputs.append(zipped)
        return result

    def on_validation_epoch_end(self):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, self.validation_step_outputs)
            # gathered_outputs = sum(gathered_outputs, [])
            gathered_outputs = np.array(gathered_outputs).reshape(-1,2)
        else:
            gathered_outputs = self.validation_step_outputs
            gathered_outputs = np.array(gathered_outputs).reshape(-1,2)
        self.validation_step_outputs.clear()
        if self.current_epoch<=0:
            print("gathered_outputs", gathered_outputs.shape)
        scores = [0]

        if self.trainer.is_global_zero:
            score = rouge_reward(gathered_outputs[:,0].tolist(), gathered_outputs[:,1].tolist())
            scores = [score]
        
        dist.broadcast_object_list(scores)
        self.log('val/score', scores[0], prog_bar=True, rank_zero_only=True, sync_dist=True)
        

    def test_step(self, batch, batch_idx):
        ## 只评价top1
        ## 其他参数，比如topk，topp，temperature可以加入到model.generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=self.training_args.max_new_tokens,
                do_sample=True,
                num_return_sequences=1,
            )
        result = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        batch["labels"][batch["labels"]==-100]=self.tokenizer.pad_token_id
        label = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        ## 打包保存
        zipped = list(zip(result, label))
        self.test_step_outputs.append(zipped)

    def on_test_epoch_end(self):
        if self.trainer.num_devices > 1:
            gathered_outputs = [None for i in range(self.trainer.num_devices)]
            dist.all_gather_object(gathered_outputs, self.test_step_outputs)
            # gathered_outputs = sum(gathered_outputs, [])
            gathered_outputs = np.array(gathered_outputs).reshape(-1,2)
        else:
            gathered_outputs = self.test_step_outputs
            gathered_outputs = np.array(gathered_outputs).reshape(-1,2)
        self.test_step_outputs.clear()

        if self.current_epoch<=0:
            print("gathered_outputs", gathered_outputs.shape)

        scores = [0]

        if self.trainer.is_global_zero:
            score = rouge_reward(gathered_outputs[:,0].tolist(), gathered_outputs[:,1].tolist())
            scores = [score]
        
        dist.broadcast_object_list(scores)
        self.log('val/score', scores[0], prog_bar=True, rank_zero_only=True)

    def predict_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        num_training_steps = self.trainer.num_training_steps
        self.print(f'Num training steps: {num_training_steps}')
        num_warmup_steps = int(num_training_steps * self.training_args.warmup_ratio)
        # parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.training_args.learning_rate, weight_decay=self.training_args.weight_decay)
        scheduler = get_scheduler(self.training_args.scheduler, optimizer, num_warmup_steps, num_training_steps)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
    
    def on_train_epoch_end(self):
        ## 保存模型
        if (self.current_epoch%5) == 0:
            self.trainer.save_checkpoint(os.path.join(self.training_args.save_path, f'checkpoints/epoch_{self.current_epoch}.ckpt'))        
        
class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def _get_metric_interpolated_filepath_name(self, monitor_candidates, trainer, del_filepath=None) -> str:
        filepath = self.format_checkpoint_name(monitor_candidates)
        return filepath

def main():
    """
    Main function for training an encoder-decoder model with
    GRPO to optimize ROUGE on the CNN/DailyMail dataset.
    """

    # Define training arguments
    ## model_name = "google/pegasus-cnn_dailymail" old
    # https://www.kaggle.com/code/anasakhtar1919/fine-tune-facebook-bart-large-cnn
    # ref:https://www.kaggle.com/code/aayushg1/lora-bart/notebook
    training_args = TrainingArguments(
        model_name = "ainize/bart-base-cnn",
        seed=42,
        epochs=1,
        num_workers=4,
        batch_size=4,
        learning_rate=1e-5,
        update_old_after=1000,
        group_size=4,
        logging_steps=10,
        max_new_tokens=128,
        max_document_length=512,
        max_summary_length=128,
        grpo_epsilon=0.1,
        grpo_beta=0.04,
        gradient_max_norm=0.2,
        save_steps=100,
        scheduler="linear",
        save_dir="./grpo_ainize_bart-base-cnn",
        debug=False,
        eval_per_epoch=5,#每5轮测试一次
        gradient_accumulation_steps=1,
        max_grad_norm=0.1,
        precision='bf16',#"bf16训练"
        gpus=2,
        load_path=None,
        do_train=True,
        do_valid=True,
        do_test=True,
        resume=False,
        warmup_ratio=0.02,
        use_lora=True
    )

    ## 单卡单线程调试
    if training_args.debug is True:
        training_args.gpus = 1


    pl.seed_everything(training_args.seed, workers=True)

    # Instantiate current policy and reference model
    # model_name = "google/pegasus-cnn_dailymail"
    
    
    tokenizer = load_tokenizer(training_args.model_name)

    # Load and format the dataset
    dm = DataModule(training_args, tokenizer=tokenizer)
    dm.prepare_data()
    dm.print_stats()

    ## 模型
    model = ModelModule(training_args, tokenizer)
    if training_args.load_path is not None and os.path.exists(training_args.load_path):
        model = ModelModule.load_from_checkpoint(training_args.load_path, strict=False, training_args=training_args, tokenizer=tokenizer)
        print("load form `load_path` successfully")

    checkpoint = ModelCheckpoint(monitor='val/score', mode='max', save_top_k=1, filename='best', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    os.makedirs(training_args.save_dir, exist_ok=True)

    # logger = pl.loggers.TensorBoardLogger(training_args.save_dir, name='', version='')

    # # 如果要用wandb,
    logger = WandbLogger(
        project="GRPO-Summarization",
        save_dir=training_args.save_dir,  # 日志保存目录（可选）
        version="",                    # 可选：版本号（默认为空）
        config={
            "model": training_args.model_name,
            "dataset": "abisee/cnn_dailymail",
            "training_args": training_args.__dict__,
        },
    )

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
        log_every_n_steps=10,
        deterministic='warn')

    if training_args.do_train:
        trainer.num_training_steps = math.ceil(
            len(dm.train_dataset) / (training_args.batch_size * training_args.gpus * training_args.gradient_accumulation_steps)) * training_args.epochs
        model.eval_dataset = dm.val_dataset
        ckpt_path = os.path.join(training_args.save_path, 'checkpoints/last.ckpt') if training_args.resume else None
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
        ## 导入最优的模型
        model = ModelModule.load_from_checkpoint(checkpoint.best_model_path, training_args=training_args, tokenizer=tokenizer)
    
    if training_args.do_valid:
        model.eval_dataset = dm.val_dataset
        trainer.validate(model, datamodule=dm)

    if training_args.do_test:
        model.eval_dataset = dm.test_dataset
        trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()