import pandas as pd
import re
import torch
import torch.optim as optim
import random
from torch.distributions import Categorical


## process_data
# 1. Reading the data
data = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1")
print(data.head())
# 2. Adding column names
data.columns = ['target', "id", "date", "type", "user", "text"]
# 3. Picking highly negative comments
data = data[["text", "target"]]
negative_comments = data[data.target == 0]
# 4. Sampling a fraction of the dataset
negative_comments = negative_comments.sample(frac=0.05)
# 5. Removing “@” tags
def remove_tags(string_with_tags):
    string_without_tags = re.sub(r'@\w+', '', string_with_tags)
    return string_without_tags

# 6. Adding everything to a text file
txt_file = ". ".join([remove_tags(txt) for txt in negative_comments.text.values.tolist()])
with open("./negative_reviews_small.txt", "w") as fp:
  fp.write(txt_file)


# Training the GPT-2 Language Model
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, TextDataset, DataCollatorForLanguageModeling, AutoTokenizer
from transformers import Trainer, TrainingArguments
from tqdm import tqdm_notebook, tnrange

# 3. Model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.cuda()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

# 4. Dataset and tokenization
train_dataset = TextDataset(tokenizer=tokenizer, file_path="negative_reviews_small.txt", block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# training_args = TrainingArguments(
#     output_dir="output/gpt2_finetune",
#     overwrite_output_dir=True,
#     num_train_epochs=20,
#     per_device_train_batch_size=16,
#     save_steps=800,
#     warmup_steps=500,
#     local_rank=-1,  # 禁用分布式
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=train_dataset,
# )

# trainer.train()

# 6. Generate some text
tokenizer.decode(model.generate(tokenizer.encode("I m going to", return_tensors="pt").cuda())[0], skip_special_tokens = True)

# The Positive Reinforcement
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer

# nltk.download("vader_lexicon")
# def reward_function(text):
#     sia = SentimentIntensityAnalyzer()
#     sentiment = sia.polarity_scores(text)
#     return sentiment["pos"] - sentiment["neg"]

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
def reward_function(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment["pos"] - sentiment["neg"]

# Reward function using a pre-trained DistilBert

from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def normalize_scores(score, new_min, new_max, old_min=0, old_max=1):
    return new_min + ((score - old_min) * (new_max - new_min)) / (old_max - old_min)

def get_normalized_sentiment_scores(text, new_min=-1, new_max=1):
    result = sentiment_pipeline(text)[0]
    positive_score = result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]
    negative_score = 1 - positive_score
    
    positive_score = normalize_scores(positive_score, new_min, new_max)
    negative_score = normalize_scores(negative_score, new_min, new_max)
    
    return positive_score - negative_score


class PPO:

  def __init__(self, model, tokenizer, reward_function, corpus, device="cuda"):

    self.model = model.to(device)
    self.tokenizer = tokenizer
    self.reward_function = reward_function
    self.device = device
    self.corpus = corpus.split(".")
  
  def random_chunk_choice(self):
    txt = random.choice(self.corpus)
    rtrn_txt = txt[random.choice([0, 2, 5]):random.choice([7, 8, 10])]
    while not len(rtrn_txt) >= 3:
      rtrn_txt = txt[random.choice([0, 2, 5]):random.choice([7, 8, 10])]
    return rtrn_txt

  def generate(self, input_text, max_length = 50):
    input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
    with torch.no_grad():
      output = self.model.generate(input_ids, max_length = max_length, do_sample = True)
    return self.tokenizer.decode(output[0], skip_special_tokens = True)
  
  def get_action_probs(self, input_text):
    input_ids = self.tokenizer.encode(input_text, return_tensors = "pt").to(self.device)
    with torch.no_grad():
      logits = self.model(input_ids).logits[:, -1, :]
      action_probs = torch.softmax(logits, dim = -1)
    return action_probs
  
  def get_reward(self, input_text):
    return self.reward_function(input_text)
  
  def train(self, num_epochs, num_rollouts, num_steps, lr, clip_epsilon, discount_factor):
    optimizer = optim.Adam(self.model.parameters(), lr=lr)

    for epoch in tnrange(num_epochs):
        self.model.train()
        old_log_probs = []
    
        for rollout in range(num_rollouts):
    
            input_text = self.random_chunk_choice()
            log_probs = []
            rewards = []

            for t in range(num_steps):
                action_probs = self.get_action_probs(input_text)
                m = Categorical(action_probs)
                action = m.sample()
                log_prob = m.log_prob(action)
                generated_text = self.tokenizer.decode(action.cpu().numpy(), skip_special_tokens=True)
                input_text += generated_text
                reward = self.get_reward(input_text)
                log_probs.append(log_prob)
                rewards.append(reward)
            
            old_log_probs.extend(log_probs)
            
            print(f'EPOCH: {epoch} | ROLLOUT: {rollout} | MEAN REWARDS: {torch.tensor(rewards).mean()}')

            discounted_rewards = []
            Gt = 0
            for reward in reversed(rewards):
                Gt = reward + discount_factor * Gt
                discounted_rewards.insert(0, Gt)

            discounted_rewards = torch.tensor(discounted_rewards).to(self.device)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            policy_loss = []
            for log_prob, old_log_prob, Gt in zip(log_probs, old_log_probs, discounted_rewards):
                ratio = torch.exp(log_prob - old_log_prob.detach())
                advantage = Gt
                policy_loss_1 = ratio * advantage
                policy_loss_2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
                policy_loss.append(-torch.min(policy_loss_1, policy_loss_2))

            policy_loss = torch.tensor(torch.stack(policy_loss).sum(), requires_grad = True)
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        old_log_probs = log_probs
        print(f"Epoch {epoch + 1}/{num_epochs} completed ")
    

ppo_agent = PPO(model, tokenizer, get_normalized_sentiment_scores, "cuda")
ppo_agent.train(num_epochs = 20, num_rollouts = 4, num_steps = 128, lr = 2e-5, clip_epsilon=0.2, discount_factor=0.99)
print(tokenizer.decode(model.generate(tokenizer.encode("I m going to be", return_tensors="pt").cuda())[0], skip_special_tokens = True))