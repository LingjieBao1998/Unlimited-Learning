import gc
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import evaluate
import nltk
import torch
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

import wandb
# ref:https://fabianofalcao.medium.com/metrics-for-evaluating-summarization-of-texts-performed-by-transformers-how-to-evaluate-the-b3ce68a309c3
rouge_eval = evaluate.load("rouge")


@dataclass
class TrainingArguments:
    epochs: int
    batch_size: int
    learning_rate: float
    update_old_after: int
    group_size: int
    logging_steps: int
    max_new_tokens: int
    max_document_length: int
    max_summary_length: int
    grpo_epsilon: float
    grpo_beta: float
    gradient_max_norm: float
    save_steps: int
    save_dir: str


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

def train(
    dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    training_args: TrainingArguments,
) -> None:
    """
    Train a language model using the GRPO (Group Relative Policy Optimization) objective.
    Args:
        dataset (Dataset): The dataset containing training data.
        model (PreTrainedModel): The model to be trained.
        tokenizer (PreTrainedTokenizer): The tokenizer used for encoding the data.
        training_args (TrainingArguments): The training arguments containing hyperparameters and configurations.
    """
    ## print the length of dataset
    print("train_data",len(dataset["train"]))
    # Prepare the dataloader
    train_dataloader = DataLoader(
        dataset["train"],
        collate_fn=DataCollatorForSeq2Seq(tokenizer),
        batch_size=training_args.batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

    # Prepare policies
    reference_model = deepcopy(model)
    old_model = deepcopy(model)
    reference_model.eval()
    old_model.eval()
    model.train()
    ## 目前新旧模型的权重是一致的

    # Prepare optimizer and lr scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
    )

    scheduler = LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.1,
        total_iters=training_args.epochs * len(train_dataloader),
    )

    # Prepare the metrics
    running_metrics = {
        "loss": 0.0,
        "reward": 0.0,
        "completion_length": 0.0,
        "kl": 0.0,
    }

    # Let's train
    training_step = 0
    best_reward = 0.0
    for _ in range(training_args.epochs):
        # Update the old policy
        ## old model是按照epoch进行更新
        old_model.load_state_dict(model.state_dict(), strict=False)
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            # Prepare the batch data
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            effective_batch_size = input_ids.shape[0]
            
            # Generate ids with the old policy
            generated_ids = old_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=training_args.max_new_tokens,
                do_sample=True,
                num_beams=training_args.group_size, #default=5
                num_return_sequences=training_args.group_size,
            ) # [batch_size * training_args.group_size, max_len]

            # Prepare attention mask for computing current
            # and reference logits on the generated ids
            decoder_attention_mask = generated_ids != tokenizer.pad_token_id # [batch_size * training_args.group_size, max_len]

            # Interleave input_ids and attention_mask to have
            # the same shape than the generated completions
            repeated_input_ids = input_ids.repeat_interleave(
                repeats=training_args.group_size, dim=0
            ) # [batch_size, max_len] -> [batch_size * training_args.group_size, max_len], 将input_ids沿着某个维度进行复制
            repeated_attention_mask = attention_mask.repeat_interleave(
                repeats=training_args.group_size, dim=0
            ) # [batch_size, max_len] -> [batch_size * training_args.group_size, max_len], 将attention_mask沿着某个维度进行复制

            # Compute the sequence scores of the old policy
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                old_scores = compute_token_scores(
                    old_model,
                    encoder_input_ids=repeated_input_ids,
                    encoder_attention_mask=repeated_attention_mask,
                    decoder_input_ids=generated_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    batch_size=effective_batch_size,
                    group_size=training_args.group_size,
                ) # [batch_size, training_args.group_size, max_len]

            # Compute the sequence scores of the current policy
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.eval()
                current_scores = compute_token_scores(
                    model,
                    encoder_input_ids=repeated_input_ids,
                    encoder_attention_mask=repeated_attention_mask,
                    decoder_input_ids=generated_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    batch_size=effective_batch_size,
                    group_size=training_args.group_size,
                )
                model.train()


            # Compute the sequence scores of the reference model
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                reference_scores = compute_token_scores(
                    reference_model,
                    encoder_input_ids=repeated_input_ids,
                    encoder_attention_mask=repeated_attention_mask,
                    decoder_input_ids=generated_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    batch_size=effective_batch_size,
                    group_size=training_args.group_size,
                )

            # Group the generated ids (batch_size, group_size, output_length)
            generated_ids = generated_ids.view(
                effective_batch_size, training_args.group_size, -1
            )

            # Repeat the labels and group (batch_size, group_size)
            labels = labels.repeat_interleave(
                repeats=training_args.group_size, dim=0
            ).view(effective_batch_size, training_args.group_size, -1)

            ## 新旧模型的权重还是一致的
            # Compute GRPO objective
            # 核心
            with torch.autocast("cuda", dtype=torch.bfloat16):
                grpo_output = grpo(
                    generated_ids,
                    old_scores,
                    current_scores,
                    reference_scores,
                    labels,
                    tokenizer,
                    training_args.grpo_epsilon,
                    training_args.grpo_beta,
                )

            # Update the current policy
            grpo_output.loss.backward()
            clip_grad_norm_(
                model.parameters(),
                training_args.gradient_max_norm,
            )
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Update old policy periodically
            if (training_step + 1) % training_args.update_old_after == 0:
                old_model.load_state_dict(model.state_dict(), strict=False)
                torch.cuda.empty_cache()

            # Update log metrics
            batch_metrics = {
                "loss": grpo_output.loss.item(),
                "reward": grpo_output.reward.item(),
                "kl": grpo_output.kl.item(),
                "completion_length": decoder_attention_mask.sum(-1)
                .float()
                .mean()
                .item(),
            }
            running_metrics = {
                key: running_metrics[key] + batch_metrics.get(key, 0)
                for key in running_metrics
            }

            # And report them periodically
            if (training_step + 1) % training_args.logging_steps == 0:
                wandb.log(
                    {
                        **{
                            key: val / (training_step + 1)
                            for key, val in running_metrics.items()
                        },
                        **{"lr": scheduler.get_last_lr()[0]},
                    }
                )

            # Save the model each periodically
            if (training_step + 1) % training_args.save_steps == 0:
                last_reward = running_metrics["loss"] / (training_step + 1)
                if last_reward > best_reward:
                    model.save_pretrained(f"{training_args.save_dir}")
                    best_reward = last_reward
                    print(
                        "Saving model with reward:",
                        best_reward,
                        f"step: {training_step+1}",
                    )
                else:
                    print(
                        f"Model not saved because didn't improve the reward at step {training_step+1}"
                    )

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
                grpo_output,
                labels,
            )
            torch.cuda.empty_cache()
            gc.collect()
            training_step += 1


def main():
    """
    Main function for training an encoder-decoder model with
    GRPO to optimize ROUGE on the CNN/DailyMail dataset.
    """
    # Instantiate current policy and reference model
    # model_name = "google/pegasus-cnn_dailymail"
    model_name = "google/pegasus-cnn_dailymail"
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)

    # Define training arguments
    training_args = TrainingArguments(
        epochs=1,
        batch_size=4,
        learning_rate=1e-5,
        update_old_after=1000,
        group_size=5,
        logging_steps=10,
        max_new_tokens=128,
        max_document_length=512,
        max_summary_length=128,
        grpo_epsilon=0.1,
        grpo_beta=0.04,
        gradient_max_norm=0.2,
        save_steps=100,
        save_dir="./grpo_pegasus-cnn-dailymail",
    )

    # Load and format the dataset
    dataset = load_cnndm()

    dataset = tokenize_dataset(
        dataset,
        tokenizer,
        training_args.max_document_length,
        training_args.max_summary_length,
    )

    # Initialize wandb
    wandb.login()
    wandb.init(
        project="GRPO-Summarization",
        config={
            "model": model_name,
            "dataset": "abisee/cnn_dailymail",
            "training_args": training_args.__dict__,
        },
    )

    # Let's train!
    train(dataset, model, tokenizer, training_args)

    # Save the model and finish logging
    model.save_pretrained(f"grpo_{model_name.replace('/', '_')}_cnn_dailymail")
    wandb.finish()


if __name__ == "__main__":
    main()