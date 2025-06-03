<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [dataset](#dataset)
- [参考](#%E5%8F%82%E8%80%83)
- [TODO](#todo)
- [loss function of `DPO` (core)](#loss-function-of-dpo-core)
- [优缺点](#%E4%BC%98%E7%BC%BA%E7%82%B9)
  - [优点](#%E4%BC%98%E7%82%B9)
  - [缺点](#%E7%BC%BA%E7%82%B9)
- [应用](#%E5%BA%94%E7%94%A8)
- [其他](#%E5%85%B6%E4%BB%96)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## dataset
* `instruction-data.json`:"https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
* `instruction-data.json`:"https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/04_preference-tuning-with-dpo/instruction-data-with-preference.json"


## 参考

- `gpt-2教程`:https://machinelearningmastery.com/text-generation-with-gpt-2-model/
- `gpt-2教程`:https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh#scrollTo=vCPohrZ-CTWu
- `gpt-2指令微调`:https://www.kaggle.com/code/umerhaddii/gpt-instruction-fine-tuning
- `gpt-2dpo微调`：ref:https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb

## TODO
add some metric to evaluate the performance of model

## loss function of `DPO` (core)
`DPO`(Direct Preference Optimization)的损失函数是其核心，通过对比学习直接优化模型的输出与人类偏好进行对齐

- After we took care of the model loading and dataset preparation in the previous sections, we can now get to the fun part and code the DPO loss
- Note that the DPO loss code below is based on the method proposed in the [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) paper
- For reference, the core DPO equation is shown again below:

<img src="assets/loss_fn_dpo.png"></img>


- In the equation above,
  - "expected value" $\mathbb{E}$ is statistics jargon and stands for the average or mean value of the random variable (the expression inside the brackets); optimizing $-\mathbb{E}$ aligns the model better with user preferences
  - The $\pi_{\theta}$ variable is the so-called policy (a term borrowed from reinforcement learning) and represents the LLM we want to optimize; $\pi_{ref}$ is a reference LLM, which is typically the original LLM before optimization (at the beginning of the training, $\pi_{\theta}$ and $\pi_{ref}$ are typically the same)
  - $\beta$ is a hyperparameter to control the divergence between the $\pi_{\theta}$ and the reference model; increasing $\beta$ increases the impact of the difference between
$\pi_{\theta}$ and $\pi_{ref}$ in terms of their log probabilities on the overall loss function, thereby increasing the divergence between the two models
  - the logistic sigmoid function, $\sigma(\centerdot)$ transforms the log-odds of the preferred and rejected responses (the terms inside the logistic sigmoid function) into a probability score 
  - $y_w$ ( $y_{win}$ ) relatively preferred by humans; $y_l$ ( $y_{loss}$ ) relatively dispreferred by humans 

- In code, we can implement the DPO loss as follows:
```python
import torch.nn.functional as F

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
```
> DPO 并没有显式采用KL散度； 隐式奖励模型

## 优缺点
TODO
### 优点
- **训练流程简单**：DPO直接从偏好进行数据优化，无需专门训练奖励模型或者使用复杂的RL算法，实现简单。（PPO的简化版本）
- **收敛快**：不涉及与环境交互（off-line）
- **稳定性好**：采用监督式对比学习，训练过程稳定

### 缺点
- **利用偏好数据**：偏好数据不易收集，多样性，奖励稀疏的场景受限；受偏好数据质量影响（比如两者评价相近，但是由于DPO的损失函数的影响，无法给出其评价相近的结果）; 不易探索新策略（off-line）

> ref:https://www.bilibili.com/video/BV1G5jgzrEWA/?spm_id_from=333.1007.tianma.9-3-28.click&vd_source=6fb7b8caa636000580be20b6d9641d90

## 应用
- 适用场景：微调模型
- 不适用场景（借鉴缺点）：受偏好数据质量影响；离线优化模式，无法在在线场景使用
> ref:https://www.bilibili.com/video/BV1G5jgzrEWA/?spm_id_from=333.1007.tianma.9-3-28.click&vd_source=6fb7b8caa636000580be20b6d9641d90


## 其他
刚开始复现的时候，学习率设置过小（`learning_rate=1e-5`），导致训练loss很大很难收敛，修改学习率后（`learning_rate=1e-4`），loss变小比较容易收敛；感觉`dpo`有点像监督学习啊啊啊啊啊啊啊