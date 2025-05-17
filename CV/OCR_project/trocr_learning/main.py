
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel
from torch.optim import AdamW
from tqdm import tqdm
from evaluate import load

cer_metric = load("cer")

## Dataset
class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer


MODEL_NAME = "microsoft/trocr-small-printed"

if __name__ == "__main__":

    df = pd.read_fwf('IAM/gt_test.txt', header=None)
    df.rename(columns={0: "file_name", 1: "text"}, inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    ### DATASET
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    train_dataset = IAMDataset(root_dir='IAM/image/',
                            df=train_df,
                            processor=processor)
    eval_dataset = IAMDataset(root_dir='IAM/image/',
                            df=test_df,
                            processor=processor)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    encoding = train_dataset[0]
    for k,v in encoding.items():
        print(k, v.shape)
    
    ## 增加一个可视化
    # import ipdb
    # ipdb.set_trace()

    ## dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=24)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.to(device)

    ##需要添加
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    

    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(300):  # loop over the dataset multiple times
        # train
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dataloader):
            # get the inputs
            for k,v in batch.items():
                batch[k] = v.to(device)

            # forward + backward + optimize
            # batch["pixel_values"].shape [24, 3, 384, 384]
            # batch["labels"].shape [24, 128]
            
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
            
        # evaluate
        model.eval()
        valid_cer = 0.0
        with torch.no_grad():
            for batch in tqdm(eval_dataloader):
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device))
                # compute metrics
                cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                valid_cer += cer 
        
        total_cer = valid_cer / len(eval_dataloader)
        print("Validation CER:", total_cer)
        if total_cer < 0.01:
            import datetime
            save_pretrained_dir = f'drive/MyDrive/{total_cer}_{epoch}_{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9),"JST")).strftime("%Y%m%dT%H%M%S")}'
            model.save_pretrained(save_pretrained_dir)

    model.save_pretrained(".")