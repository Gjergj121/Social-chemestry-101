import pandas as pd
import torch
from transformers import (AutoTokenizer, 
                          AutoModelForSeq2SeqLM, 
                          Seq2SeqTrainingArguments, 
                          DataCollatorForSeq2Seq,
                          Seq2SeqTrainer)
from datasets import load_dataset, DatasetDict
from argparse import ArgumentParser
import evaluate
import numpy as np
from nltk.tokenize import sent_tokenize
import numpy as np
from nltk.tokenize import sent_tokenize

from utils import get_situations_with_not_unique_rot_categories, prepare_new_column


rouge_score = evaluate.load("rouge")
bleu_score = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    bleu_result_1 = bleu_score.compute(
        predictions=decoded_preds, references=decoded_labels, max_order=1
    )
    bleu_result_2 = bleu_score.compute(
        predictions=decoded_preds, references=decoded_labels, max_order=2
    )
    bleu_result_4 = bleu_score.compute(
        predictions=decoded_preds, references=decoded_labels, max_order=4
    )

    # Extract the median scores
    #result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result["bleu_1gram"] = bleu_result_1["bleu"]
    result["bleu_2gram"] = bleu_result_2["bleu"]
    result["bleu_4gram"] = bleu_result_4["bleu"]

    return {k: round(v, 4) for k, v in result.items()}


parser = ArgumentParser()
parser.add_argument("--model_name", dest="model_name", required=True, type=str)
parser.add_argument("--include_rot_categorizations", dest="include_rot_categorizations", action='store_true', help="Enable or not the rot categorizations as part of input text.")

if __name__ == "__main__":
    args = parser.parse_args()
    INCLUDE_ROT_CATEGORIZATIONS = args.include_rot_categorizations
    raw_dataset = load_dataset("metaeval/social-chemestry-101")["train"]
    situations_with_not_unique_rot = set()

    if INCLUDE_ROT_CATEGORIZATIONS:
        situations_with_not_unique_rot = get_situations_with_not_unique_rot_categories(raw_dataset)
        values = prepare_new_column(raw_dataset)
        raw_dataset = raw_dataset.add_column("situation_rot_categories", values)

#    raw_dataset = raw_dataset.select_columns(["split", "rot", "situation"])

    dataset = DatasetDict({
        "train": raw_dataset.filter(lambda example: example['split'] == 'train' and example['situation-short-id'].split("/")[-1] not in situations_with_not_unique_rot),
        "val": raw_dataset.filter(lambda example: example['split'] == 'dev' and example['situation-short-id'].split("/")[-1] not in situations_with_not_unique_rot),
        "test": raw_dataset.filter(lambda example: example['split'] == 'test' and example['situation-short-id'].split("/")[-1] not in situations_with_not_unique_rot)
    })

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model size: {model_size/1000**2:.1f}M parameters")

    max_input_length = 128
    max_target_length = 64

    def preprocess_function(examples):
        if INCLUDE_ROT_CATEGORIZATIONS:
            model_inputs = tokenizer(
                        examples["situation_rot_categories"],
                        max_length=max_input_length,
                        truncation=True,
            )
        else:
            model_inputs = tokenizer(
                examples["situation"],
                max_length=max_input_length,
                truncation=True,
            )
        labels = tokenizer(
            examples["rot"], max_length=max_target_length, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True, desc="Running tokenizer on dataset", remove_columns=raw_dataset.column_names)
    print(tokenized_datasets)


    num_train_epochs = 10
    model_name = model_name.split("/")[-1]
    batch_size = 128
    flag = str(INCLUDE_ROT_CATEGORIZATIONS)
    
    args = Seq2SeqTrainingArguments(
        output_dir=f"../data/includeRotCategories_{flag}_{model_name}-finetuned-sit-rot",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=batch_size,
        learning_rate=5.6e-5,
        auto_find_batch_size=True,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        load_best_model_at_end=True,
        report_to="wandb",
    )   


    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    predictions_output = trainer.predict(tokenized_datasets["test"])
    print(predictions_output.metrics)
    trainer.save_metrics("test", predictions_output.metrics)




        