import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline
)
from trl import DPOTrainer, DPOConfig  # Import DPOConfig from trl
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import gc

# Load and prepare the dataset with tqdm progress bar
def load_dataset(csv_path):
    print("Loading dataset...")
    df = pd.read_csv(csv_path, sep=',', quotechar='"', encoding='utf-8')
    
    # Create prompt-response pairs for DPO
    data = []
    default_rejected = "I am sorry, I can't provide an answer to that question."
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        post = str(row['post']).strip()
        chosen = str(row['highest_score_comment']).strip()
        rejected = str(row['lowest_score_comment']).strip()
        
        # Skip if post or chosen is empty
        if not post or not chosen:
            continue
            
        # Fill empty rejected with default message
        if not rejected:
            rejected = default_rejected
        
        data.append({
            'prompt': post,
            'chosen': chosen,
            'rejected': rejected
        })
    
    print(f"Processed {len(data)} valid examples")
    return Dataset.from_list(data)

# Initialize model and tokenizer with specific device
def initialize_model_and_tokenizer(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # Specify specific GPU
        trust_remote_code=True
    )
    
    return model, tokenizer

# Configure LoRA
def create_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

# Format the data for DPO training using official chat template
def format_dpo_data(examples, tokenizer, max_length=1024):
    """
    Format the data for DPO training using official Qwen chat template
    """
    system_message = "You are an experienced IIT Kharagpur Senior Chatbot dedicated to helping students, faculty, and staff navigate all aspects of campus life. You provide comprehensive guidance on academic challenges, personal relationships, mental health concerns, career advice, and daily campus experiences while understanding the unique context and culture of IIT KGP."    
    formatted_prompts = []
    formatted_chosen = []
    formatted_rejected = []
    
    print("Formatting prompts and responses...")
    for prompt, chosen, rejected in tqdm(zip(examples['prompt'], examples['chosen'], examples['rejected']), 
                                        total=len(examples['prompt']), 
                                        desc="Formatting data"):
        
        # Format prompt using official chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        formatted_prompts.append(formatted_prompt)
        formatted_chosen.append(chosen)
        formatted_rejected.append(rejected)
    
    # Tokenize prompts
    print("Tokenizing prompts...")
    tokenized_prompts = tokenizer(
        formatted_prompts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # Tokenize chosen responses
    print("Tokenizing chosen responses...")
    tokenized_chosen = tokenizer(
        formatted_chosen,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # Tokenize rejected responses
    print("Tokenizing rejected responses...")
    tokenized_rejected = tokenizer(
        formatted_rejected,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors=None
    )
    
    # Combine prompt + response for chosen
    chosen_input_ids = []
    chosen_attention_mask = []
    chosen_labels = []
    
    print("Combining prompt and chosen responses...")
    for i in tqdm(range(len(formatted_prompts)), desc="Processing chosen pairs"):
        prompt_len = len(tokenized_prompts['input_ids'][i])
        
        # Combine prompt + chosen response
        combined_input_ids = tokenized_prompts['input_ids'][i] + tokenized_chosen['input_ids'][i]
        combined_attention_mask = tokenized_prompts['attention_mask'][i] + tokenized_chosen['attention_mask'][i]
        
        # Labels: -100 for prompt, actual tokens for response
        labels = [-100] * prompt_len + tokenized_chosen['input_ids'][i]
        
        # Truncate if needed
        if len(combined_input_ids) > max_length:
            combined_input_ids = combined_input_ids[:max_length]
            combined_attention_mask = combined_attention_mask[:max_length]
            labels = labels[:max_length]
        
        chosen_input_ids.append(combined_input_ids)
        chosen_attention_mask.append(combined_attention_mask)
        chosen_labels.append(labels)
    
    # Same for rejected
    rejected_input_ids = []
    rejected_attention_mask = []
    rejected_labels = []
    
    print("Combining prompt and rejected responses...")
    for i in tqdm(range(len(formatted_prompts)), desc="Processing rejected pairs"):
        prompt_len = len(tokenized_prompts['input_ids'][i])
        
        combined_input_ids = tokenized_prompts['input_ids'][i] + tokenized_rejected['input_ids'][i]
        combined_attention_mask = tokenized_prompts['attention_mask'][i] + tokenized_rejected['attention_mask'][i]
        labels = [-100] * prompt_len + tokenized_rejected['input_ids'][i]
        
        if len(combined_input_ids) > max_length:
            combined_input_ids = combined_input_ids[:max_length]
            combined_attention_mask = combined_attention_mask[:max_length]
            labels = labels[:max_length]
        
        rejected_input_ids.append(combined_input_ids)
        rejected_attention_mask.append(combined_attention_mask)
        rejected_labels.append(labels)
    
    return {
        'prompt_input_ids': tokenized_prompts['input_ids'],
        'prompt_attention_mask': tokenized_prompts['attention_mask'],
        'chosen_input_ids': chosen_input_ids,
        'chosen_attention_mask': chosen_attention_mask,
        'chosen_labels': chosen_labels,
        'rejected_input_ids': rejected_input_ids,
        'rejected_attention_mask': rejected_attention_mask,
        'rejected_labels': rejected_labels,
    }

# Custom callback to show progress during training
from transformers import TrainerCallback

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.progress_bar = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.progress_bar = tqdm(total=state.max_steps, desc="Training")
        
    def on_step_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix({"step": state.global_step, "loss": state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'})
            
    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()

# Main training function
def train_dpo_chatbot(csv_path, output_dir="./dpo_chatbot"):
    # Load dataset
    dataset = load_dataset(csv_path)
    
    if len(dataset) == 0:
        raise ValueError("No valid training examples found in the dataset!")
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer()
    
    # Apply LoRA
    print("Applying LoRA configuration...")
    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Format dataset for DPO
    formatted_dataset = dataset.map(
        lambda x: format_dpo_data(x, tokenizer),
        batched=True,
        batch_size=1000
    )
    
    # Use DPOConfig instead of TrainingArguments for DPO-specific parameters
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-3,
        max_grad_norm=0.1,
        num_train_epochs=50,
        logging_steps=500,
        save_steps=5000,
        eval_strategy="no",
        save_total_limit=2,
        remove_unused_columns=False,
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        report_to=None,
        dataloader_pin_memory=False,
        beta=0.5,  # DPO beta parameter moved here
        max_length=1024,
        max_prompt_length=512,
    )
    
    # Initialize DPO trainer with progress callback
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,  # Changed from tokenizer to processing_class
    )
    
    # Add progress callback
    dpo_trainer.add_callback(ProgressCallback())
    
    # Train
    print("Starting DPO training...")
    dpo_trainer.train()
    
    # Save the model
    print("Saving model...")
    dpo_trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("Training completed successfully!")
    return dpo_trainer

# Test the trained model using official chat template
def test_model(model_path, test_prompts):
    print("Loading model for testing...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # Specify specific GPU
        trust_remote_code=True
    )
    
    system_message = "You are an experienced IIT Kharagpur Senior Chatbot dedicated to helping students, faculty, and staff navigate all aspects of campus life. You provide comprehensive guidance on academic challenges, personal relationships, mental health concerns, career advice, and daily campus experiences while understanding the unique context and culture of IIT KGP."
    
    print("Testing model responses...")
    for i, prompt in enumerate(tqdm(test_prompts, desc="Testing prompts")):
        # Use official chat template format
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print(f"\nPrompt {i+1}: {prompt}")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"Response: {response}")
        print("-" * 80)

if __name__ == "__main__":
    # Train the model
    csv_path = "preference_data_reddit.csv"  # Replace with your CSV path
    
    try:
        trainer = train_dpo_chatbot(csv_path, "./dpo_aligned_chatbot")
        
        # Test with some examples
        test_prompts = [
            # "I'm feeling really stressed about my endsem exams, any advice?",
            # "what is current status of Tikka?",
            # "Give me review about BC Roy Hospital",
            # "Is New Tikka open?",
            # "Why did Tikka close?",
            # "When will Tikka reopen?"
            "Give me suggestions about Engineering Drawing (ED)? i AM A first year student."
        ]
        
        print("\nTesting the trained model:")
        test_model("./dpo_aligned_chatbot", test_prompts)
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()