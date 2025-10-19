import os
import math
import time
import json
import torch
import random
import argparse
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from PIL import Image
from torch.nn import functional as F

from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


def load_llava(model_name_or_path, device_map="auto", use_quantization=True):
    quantization_config = None
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=use_quantization,
            bnb_4bit_compute_dtype=torch.float16
        )
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    tokenizer = processor.tokenizer
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path,
                                                          device_map=device_map,
                                                          attn_implementation="eager",
                                                          quantization_config=quantization_config)
    return model, tokenizer, processor


def eval_pope(model, tokenizer, processor, args):
    pope_ds = load_dataset(args.pope_path, "default")["test"]
    pope_dataset = pope_ds.filter(lambda x: x["category"] == args.pope_catagory)

    generate_config = {
        "return_dict_in_generate": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "max_new_tokens": args.pope_max_new_token,
    }

    correct_ans = 0.0
    
    start_time = time.time()
    for pope_data in tqdm(pope_dataset):
        # print(pope_data)
        question = pope_data["question"]
        answer = pope_data["answer"]
        image = pope_data["image"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question} Answer with only yes or no."},
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print(text_prompt)

        inputs = processor(images=image, text=text_prompt, return_tensors='pt').to(device)
        output = model.generate(**inputs, **generate_config)
        output_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        response_answer = output_text.split("ASSISTANT:")[-1].strip()

        response_answer_reformat = response_answer.lower()
        # print(type(response_answer_reformat), response_answer_reformat)
        if answer in response_answer_reformat:
            correct_ans += 1.0
        
    end_time = time.time()
    poope_acc = correct_ans / len(pope_dataset)
    print(f"POPE Evaluation result: {poope_acc:.4f}, use time: {(end_time - start_time):.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset settings
    parser.add_argument('--pope_path', type=str, default='lmms-lab/POPE')
    parser.add_argument('--pope_catagory', type=str, default='adversarial', choices=['adversarial', 'popular', 'random'])
    parser.add_argument('--pope_max_new_token', type=int, default=8)

    # model_info
    parser.add_argument('--model_path', type=str, default='llava-hf/llava-1.5-7b-hf')

    args = parser.parse_args()
    print(args)

    # Load model
    print("-"*30+"Load model"+"-"*30)
    model, tokenizer, processor = load_llava(args.model_path)
    device = model.device
    print(f"device: {device}")

    # print(model)

    # POPE Evaluation
    print("-"*30+"POPE Evaluation"+"-"*30)
    eval_pope(model, tokenizer, processor, args)
    

