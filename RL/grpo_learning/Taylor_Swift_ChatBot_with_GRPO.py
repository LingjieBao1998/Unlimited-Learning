## ref:https://blog.gopenai.com/llm-fine-tuning-with-grpo-example-6eebe903907b
## 1. Importing required libraries
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from rouge_score import rouge_scorer
from peft import LoraConfig
import time

# export HF_ENDPOINT=https://hf-mirror.com
## 2. Setting up database
# Define a system prompt to maintain response consistency
SYSTEM_PROMPT = """You are a Taylor Swift expert. Answer CORRECTLY and CONCISELY questions about Taylor Swift's life, achievements, songs, and more."""

dataset_name = "lamini/taylor_swift"


def get_data(dataset_name, split="train") -> Dataset:
    """Loads and formats the dataset into a structured prompt format."""
    data = load_dataset(dataset_name)[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": x["answer"],
        }
    )
    return data.select_columns(["prompt", "answer"])

dataset = get_data(dataset_name=dataset_name)

## 3.Setting up Rewards
# * ROUGE-L Score: Measures n-gram overlap with reference answers.
# * Length Similarity: Ensures the response structure aligns with ground truth.
# * LLM-Judging (LLM-J): Uses an LLM as a judge to assess response correctness.
# > One of the most exciting aspects of this experiment is the integration of LLM-Judging (LLM-J), where a separate LLM evaluates generated responses for correctness.

# updated reward function: R=0.3×ROUGE-L+0.2×Length Similarity+0.5×LLM-J

### Dynamic Context Length Scaling
# ROUGE-L based reward function to evaluate response similarity to reference answers
def rouge_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    responses = [completion[0]["content"] for completion in completions]
    rewards = [
        scorer.score(ref_answer, response)["rougeL"].fmeasure
        for response, ref_answer in zip(responses, answer)
    ]
    return rewards


# Reward function based on length similarity
def length_similarity_reward_func(
    prompts, completions, answer, scale_factor=0.5, **kwargs
) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    rewards = [
        min(len(response.split()), len(ref_answer.split()))
        / max(len(response.split()), len(ref_answer.split()))
        * scale_factor
        if len(ref_answer.split()) > 0
        else 0.0
        for response, ref_answer in zip(responses, answer)
    ]
    return rewards


def llm_judge_reward(
    prompt, generated_response, reference_answer, model, tokenizer
):  # llm_judge_score
    """Uses the fine-tuning Qwen-0.5B model to score responses locally."""
    eval_prompt = f"""
    Evaluate the correctness of the following response compared to the reference.

    Prompt: {prompt}
    Reference Answer: {reference_answer}
    Generated Response: {generated_response}

    Score the response between 0 and 1, where:
    - 1.0 is a perfect answer.
    - 0.0 is completely incorrect.

    Provide only a numerical score with no explanation.
    """

    # Tokenize input using the already-loaded tokenizer
    inputs = tokenizer(
        eval_prompt, return_tensors="pt", truncation=True, max_length=512
    )
    inputs = {
        k: v.to(model.device) for k, v in inputs.items()
    }  # Ensure tensors are on the correct device

    # Generate response using the already-loaded model
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    # Decode and extract the score
    score_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    try:
        score = float(score_text)
        return max(0.0, min(1.0, score))  # Ensure the score is within the [0,1] range
    except:
        return 0.5  # Default fallback score if parsing fails


def llm_judge_reward(prompts, generated_responses, answer, model, tokenizer):
    """Uses Qwen-0.5B locally to evaluate response correctness."""
    eval_prompts = [
        f"""Evaluate the correctness of the following response compared to the reference.

    Prompt: {p}
    Reference Answer: {r}
    Generated Response: {g}

    Score the response between 0 and 1. Only return a number."""
        for p, g, r in zip(prompts, generated_responses, answer)
    ]

    inputs = tokenizer(
        eval_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    scores = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]

    try:
        return [max(0.0, min(1.0, float(s))) for s in scores]
    except:
        return [0.5] * len(prompts)  # Default fallback


def llm_judge_reward_batch(prompts, generated_responses, answer, model, tokenizer):
    """Uses the fine-tuned Qwen-0.5B model to score multiple responses at once."""

    eval_prompts = [
        f"""Evaluate the correctness of the following response compared to the reference.

        Prompt: {p}
        Reference Answer: {r}
        Generated Response: {g}

        Score the response between 0 and 1. Only return a number."""
        for p, g, r in zip(prompts, generated_responses, answer)
    ]

    # Tokenize all prompts in a batch
    inputs = tokenizer(
        eval_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate responses in a batch
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    # Decode scores
    scores = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]

    # Convert scores to float values
    try:
        return [max(0.0, min(1.0, float(s))) for s in scores]
    except:
        return [0.5] * len(prompts)  # Default fallback


def combined_reward(prompts, completions, answer, model, tokenizer, completion_ids):
    """Combines ROUGE, Length Similarity, and Qwen-0.5B as LLM-J."""

    rouge_scores = rouge_reward_func(prompts, completions, answer)
    length_scores = length_similarity_reward_func(prompts, completions, answer)

    generated_responses = [c[0]["content"] for c in completions]
    llm_scores = llm_judge_reward_batch(
        prompts, generated_responses, answer, model, tokenizer
    )

    # Weighted combination of scores
    final_rewards = [
        (0.3 * rouge) + (0.2 * length) + (0.5 * llm)
        for rouge, length, llm in zip(rouge_scores, length_scores, llm_scores)
    ]

    return final_rewards


## 4. training
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
output_dir = "outputs/Qwen-0.5B-GRPO"

training_args = GRPOConfig(
    output_dir=output_dir,
    optim="adamw_torch_fused",  ######
    learning_rate=0.0001,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.2,
    # warmup_steps=100,
    # lr_scheduler_type='cosine_with_restarts',
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    bf16=True,
    # fp16=True,  ####
    per_device_train_batch_size=8,  # 8, #4,
    gradient_accumulation_steps=2,  # 1,
    num_generations=8,  # 4,
    max_prompt_length=192,
    max_completion_length=160,
    num_train_epochs=5,
    save_steps=100,
    # max_grad_norm=0,  # 0.1, # disables gradient clipping
    log_on_each_node=False,
    use_vllm=False,  # True,
    vllm_gpu_memory_utilization=0.3,  # 0.6,
    report_to="none",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype=torch.bfloat16,
    torch_dtype=torch.float16,  #####
    device_map=None,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

import os

os.environ["VLLM_DTYPE"] = "float16"  #####

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    lora_dropout=0.1,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        # TypeError: <lambda>() got an unexpected keyword argument 'completion_ids'
        lambda prompts, completions, completion_ids, answer: combined_reward(
            prompts, completions, answer, model, tokenizer, completion_ids
        )
    ],  # Pass model & tokenizer
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

from transformers import TrainerCallback


class AdjustContextLengthCallback(TrainerCallback):
    """Dynamically increases max_completion_length during training."""

    def on_step_begin(self, args, state, control, **kwargs):
        """Adjusts max_completion_length based on training progress."""
        step = state.global_step

        if step >= 1000:
            args.max_prompt_length = 384  # Allow longer completions
        elif step >= 500:
            args.max_completion_length = 256  # Gradually increase

        # Log changes
        if step in [500, 1000]:
            print(
                f"Adjusted max_completion_length to {args.max_completion_length} at step {step}"
            )


# Add dynamic context adjustment
trainer.add_callback(AdjustContextLengthCallback())


## F1-Score Before-tuning
eval_dataset = get_data(dataset_name=dataset_name, split="test")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
separator = ";Assistant:"

total_rouge_l = 0
total_inference_time = 0
for example in eval_dataset:
    start_time = time.time()  # Start timer
    prompt_text = "".join([d["content"] for d in example["prompt"]]) + " " + separator
    generated_text = model.generate(
        **trainer.processing_class(prompt_text, return_tensors="pt", padding=True).to(
            "cuda"
        )
    )
    # generated_text = model(generated_text, skip_special_tokens=True)[0]
    generated_text = tokenizer.batch_decode(generated_text, skip_special_tokens=True)[0]

    if separator in generated_text:
        generated_text = generated_text.split(separator, 1)[-1].strip()
    inference_time = time.time() - start_time  # Measure inference time
    total_inference_time += inference_time

    # Calculate ROUGE-L F1 score
    rouge_scores = scorer.score(example["answer"], generated_text)
    rouge_l_f1 = rouge_scores["rougeL"].fmeasure
    total_rouge_l += rouge_l_f1

average_rouge_l = total_rouge_l / len(eval_dataset)
average_inference_time = total_inference_time / len(eval_dataset)
print(f"(Previous) Average ROUGE-L F1 score on test set: {average_rouge_l}")
print(f"Inference Time: {average_inference_time:.4f} sec\n")

trainer.train()


eval_dataset = get_data(dataset_name=dataset_name, split="test")
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
separator = ";Assistant:"

total_rouge_l = 0
total_inference_time = 0
for example in eval_dataset:
    start_time = time.time()  # Start timer
    prompt_text = "".join([d["content"] for d in example["prompt"]]) + " " + separator
    generated_text = trainer.model.generate(
        **trainer.processing_class(prompt_text, return_tensors="pt", padding=True).to(
            "cuda"
        )
    )
    generated_text = trainer.processing_class.batch_decode(
        generated_text, skip_special_tokens=True
    )[0]

    if separator in generated_text:
        generated_text = generated_text.split(separator, 1)[-1].strip()
    inference_time = time.time() - start_time  # Measure inference time
    total_inference_time += inference_time

    # Calculate ROUGE-L F1 score
    rouge_scores = scorer.score(example["answer"], generated_text)
    rouge_l_f1 = rouge_scores["rougeL"].fmeasure
    total_rouge_l += rouge_l_f1

average_rouge_l = total_rouge_l / len(eval_dataset)
average_inference_time = total_inference_time / len(eval_dataset)
print(f"(Post) Average ROUGE-L F1 score on test set: {average_rouge_l}")
print(f"Inference Time: {average_inference_time:.4f} sec\n")
