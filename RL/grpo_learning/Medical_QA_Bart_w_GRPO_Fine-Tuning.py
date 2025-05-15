import torch
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BartTokenizer, BartForConditionalGeneration
from trl import GRPOConfig
import os
import torch.nn as nn
import torch.nn.functional as F
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK resources
try:
    nltk.download('punkt')
except:
    pass

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration - with fallback to CPU if CUDA is not available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"  # You can use larger models like t5-base if you have enough GPU memory
tokenizer = BartTokenizer.from_pretrained(model_name)
base_model = BartForConditionalGeneration.from_pretrained(model_name)

# Function to load and preprocess data
def load_and_preprocess_data(file_path, sample_size=None):
    # Load data from CSV file
    df = pd.read_csv(file_path)
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Extract questions and answers
    questions = df['Question'].tolist()
    answers = df['Answer'].tolist()
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        questions, answers, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

# Create a GRPO environment wrapper for Seq2Seq tasks
class QAEnvironment:
    def __init__(self, model, tokenizer, questions, answers):
        self.model = model
        self.tokenizer = tokenizer
        self.questions = questions
        self.answers = answers
        self.current_idx = 0
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def step(self, generated_tokens):
        # Get reference answer
        reference = self.answers[self.current_idx]
        
        # Decode the generated tokens
        generated_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        # Calculate reward based on ROUGE scores
        scores = self.calculate_reward(generated_text, reference)
        reward = scores['combined']
        
        # Move to next example
        self.current_idx = (self.current_idx + 1) % len(self.questions)
        
        # Return reward as tensor on the same device as the model
        return torch.tensor([reward], device=device)
    
    def calculate_reward(self, prediction, reference):
        # Calculate ROUGE scores
        rouge_scores = self.scorer.score(prediction, reference)
        
        # Calculate BLEU score
        try:
            smooth = SmoothingFunction().method1
            reference_tokens = nltk.word_tokenize(reference.lower())
            prediction_tokens = nltk.word_tokenize(prediction.lower())
            bleu_score = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smooth)
        except:
            bleu_score = 0.0
        
        # Create a combined score
        scores = {
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'bleu': bleu_score,
            'combined': (
                rouge_scores['rouge1'].fmeasure + 
                rouge_scores['rouge2'].fmeasure + 
                rouge_scores['rougeL'].fmeasure + 
                bleu_score
            ) / 4.0
        }
        
        return scores
    
    def reset(self):
        # Reset index to beginning or randomly
        self.current_idx = random.randint(0, len(self.questions) - 1)
        return self.get_current_input()
    
    def get_current_input(self):
        question = self.questions[self.current_idx]
        # For T5, prefix the input with a task-specific prefix
        encoding = self.tokenizer(f"answer: {question}", return_tensors='pt', truncation=True, max_length=128)
        return {k: v.to(device) for k, v in encoding.items()}

    def get_current_target(self):
        answer = self.answers[self.current_idx]
        target_encoding = self.tokenizer(text_target=answer, return_tensors='pt', truncation=True, max_length=128)
        return target_encoding['input_ids'].to(device)

# Custom GRPO Trainer for Seq2Seq tasks
class QAGRPOTrainer:
    def __init__(self, model, tokenizer, train_questions, train_answers, test_questions, test_answers, 
                 max_length=128, max_new_tokens=64):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_env = QAEnvironment(model, tokenizer, train_questions, train_answers)
        self.test_env = QAEnvironment(model, tokenizer, test_questions, test_answers)
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        
        # Configure GRPO
        self.grpo_config = GRPOConfig(
            learning_rate=5e-5,
            gradient_accumulation_steps=1,
            seed=42,
            output_dir='./'
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.grpo_config.learning_rate)
        
        # Initialize losses and metrics tracking
        self.train_losses = []
        self.metrics = []
        
    def train(self, epochs=10, eval_freq=1):
        best_score = 0.0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_losses = []
            
            # Training loop
            progress_bar = tqdm(range(len(self.train_env.questions)))
            for i in progress_bar:
                # Get current example
                inputs = self.train_env.get_current_input()
                target_ids = self.train_env.get_current_target()
                
                # Forward pass with teacher forcing for supervised learning
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=target_ids
                )
                
                # Calculate standard seq2seq loss (teacher forcing loss)
                supervised_loss = outputs.loss
                
                # Sample from the model for GRPO using generate()
                with torch.no_grad():
                    generated_tokens = self.model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                
                # Get reward from environment
                reward = self.train_env.step(generated_tokens)
                
                # Use log probabilities and reward for GRPO loss
                # For simplicity, we're just scaling the supervised loss by the reward
                # This is a simple approximation - in a real implementation, you'd compute 
                # the policy gradient loss more precisely
                grpo_loss = -reward * supervised_loss
                
                # Combine losses - balance between supervised and RL loss
                loss = supervised_loss + 0.5 * grpo_loss
                
                # Backpropagate
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Record loss
                epoch_losses.append(loss.item())
                progress_bar.set_description(f"Loss: {loss.item():.4f}")
                
                # Move to next example
                self.train_env.current_idx = (self.train_env.current_idx + 1) % len(self.train_env.questions)
            
            # Record average loss for epoch
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            self.train_losses.append(avg_loss)
            print(f"Average training loss: {avg_loss:.4f}")
            
            # Evaluate on test set
            if (epoch + 1) % eval_freq == 0:
                metrics = self.evaluate()
                self.metrics.append(metrics)
                
                # Save best model based on combined score
                if metrics['combined'] > best_score:
                    best_score = metrics['combined']
                    torch.save(self.model.state_dict(), "best_qa_model.pt")
                    print(f"Saved new best model with combined score: {best_score:.4f}")
        
        # Load best model for final evaluation
        self.model.load_state_dict(torch.load("best_qa_model.pt"))
        return self.model
    
    def evaluate(self):
        self.model.eval()
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        combined_scores = []
        
        generated_examples = []
        
        with torch.no_grad():
            for i in tqdm(range(min(len(self.test_env.questions), 100))):  # Evaluate on at most 100 examples
                # Get test example
                question = self.test_env.questions[i]
                reference = self.test_env.answers[i]
                
                # Generate answer
                inputs = self.tokenizer(f"answer: {question}", return_tensors='pt', truncation=True, max_length=self.max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                generated_ids = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=self.max_new_tokens,
                    num_beams=4,
                    early_stopping=True
                )
                
                prediction = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Calculate scores
                scores = self.test_env.calculate_reward(prediction, reference)
                
                # Record scores
                rouge1_scores.append(scores['rouge1'])
                rouge2_scores.append(scores['rouge2'])
                rougeL_scores.append(scores['rougeL'])
                bleu_scores.append(scores['bleu'])
                combined_scores.append(scores['combined'])
                
                # Save a few examples
                if len(generated_examples) < 5:
                    generated_examples.append({
                        'question': question,
                        'reference': reference,
                        'prediction': prediction
                    })
        
        # Calculate average scores
        avg_metrics = {
            'rouge1': sum(rouge1_scores) / len(rouge1_scores),
            'rouge2': sum(rouge2_scores) / len(rouge2_scores),
            'rougeL': sum(rougeL_scores) / len(rougeL_scores),
            'bleu': sum(bleu_scores) / len(bleu_scores),
            'combined': sum(combined_scores) / len(combined_scores)
        }
        
        # Print metrics
        print("\n=== Evaluation Metrics ===")
        print(f"ROUGE-1: {avg_metrics['rouge1']:.4f}")
        print(f"ROUGE-2: {avg_metrics['rouge2']:.4f}")
        print(f"ROUGE-L: {avg_metrics['rougeL']:.4f}")
        print(f"BLEU: {avg_metrics['bleu']:.4f}")
        print(f"Combined Score: {avg_metrics['combined']:.4f}")
        
        # Print examples
        print("\n=== Generated Examples ===")
        for i, example in enumerate(generated_examples):
            print(f"Example {i+1}:")
            print(f"Question: {example['question']}")
            print(f"Reference: {example['reference']}")
            print(f"Prediction: {example['prediction']}")
            print("----------------------------")
        
        self.model.train()
        return avg_metrics
    
    def plot_learning_curves(self):
        plt.figure(figsize=(15, 10))
        
        # Plot training loss
        plt.subplot(2, 3, 1)
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot ROUGE-1
        plt.subplot(2, 3, 2)
        plt.plot([m['rouge1'] for m in self.metrics])
        plt.title('ROUGE-1 Score')
        plt.xlabel('Evaluation')
        plt.ylabel('Score')
        
        # Plot ROUGE-2
        plt.subplot(2, 3, 3)
        plt.plot([m['rouge2'] for m in self.metrics])
        plt.title('ROUGE-2 Score')
        plt.xlabel('Evaluation')
        plt.ylabel('Score')
        
        # Plot ROUGE-L
        plt.subplot(2, 3, 4)
        plt.plot([m['rougeL'] for m in self.metrics])
        plt.title('ROUGE-L Score')
        plt.xlabel('Evaluation')
        plt.ylabel('Score')
        
        # Plot BLEU
        plt.subplot(2, 3, 5)
        plt.plot([m['bleu'] for m in self.metrics])
        plt.title('BLEU Score')
        plt.xlabel('Evaluation')
        plt.ylabel('Score')
        
        # Plot Combined Score
        plt.subplot(2, 3, 6)
        plt.plot([m['combined'] for m in self.metrics])
        plt.title('Combined Score')
        plt.xlabel('Evaluation')
        plt.ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('learning_curves_seq2seq_grpo.png')
        plt.close()
    
    def generate_answer(self, question):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(f"answer: {question}", return_tensors='pt', truncation=True, max_length=self.max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            generated_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                early_stopping=True
            )
            
            answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        self.model.train()
        return answer

def main():
    # Define the data file path (update this with your actual file path)
    file_path = "train.csv"  # Replace with your actual CSV path
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path, sample_size=300)
    
    print(f"Training on {len(X_train)} examples, testing on {len(X_test)} examples")
    
    # Initialize the model
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Create trainer
    trainer = QAGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_questions=X_train,
        train_answers=y_train,
        test_questions=X_test,
        test_answers=y_test,
        max_length=128,
        max_new_tokens=64
    )
    
    # Train model
    print("Training model with GRPO...")
    trainer.train(epochs=30, eval_freq=1)
    
    # Plot learning curves
    trainer.plot_learning_curves()
    
    # Final evaluation
    print("\n=== Final Model Evaluation ===")
    metrics = trainer.evaluate()

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('learning_curves_seq2seq_grpo.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')  
    plt.show()  
    
    # Test with example questions
    test_questions = [
        "What is machine learning?",
        "How does a neural network work?",
        "Explain the concept of reinforcement learning."
    ]
    
    print("\n=== Test Questions ===")
    for question in test_questions:
        answer = trainer.generate_answer(question)
        print(f"Question: {question}")
        print(f"Generated Answer: {answer}")
        print("----------------------------")

if __name__ == "__main__":
    main()
    