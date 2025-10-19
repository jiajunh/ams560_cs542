import os
import math
import time
import json
import torch
import random
import argparse
from tqdm import tqdm
from types import MethodType

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

from PIL import Image
from torch.nn import functional as F

from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

# from transformers.models.llava.modeling_llava import (
#     LlavaModelOutputWithPast
# )

from transformers.feature_extraction_utils import BatchFeature

from transformers.image_utils import get_image_size, to_numpy_array

from transformers.models.llava.processing_llava import (
    LlavaProcessorKwargs
)


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


def get_first_and_last_img_token_idx(input_ids, img_token_id=32000):
    mask = input_ids[0] == img_token_id
    img_token_idx = torch.where(mask)[0]
    return img_token_idx[0], img_token_idx[-1]


def new_get_img_features(self,
    pixel_values,
    vision_feature_layer=None,
    vision_feature_select_strategy=None,
    **kwargs,):
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )

    if vision_feature_select_strategy not in ["default", "full"]:
        raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
    image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, output_attentions=True, **kwargs)
    # If we have one vision feature layer, return the corresponding hidden states,
    # otherwise, select the hidden states of each feature layer and concatenate them
    if isinstance(vision_feature_layer, int):
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        selected_attention = image_outputs.attentions[vision_feature_layer]
        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
            selected_attention = selected_attentions[:, 1:]
    else:
        hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
        att_pool = [image_outputs.attentions[layer_idx] for layer_idx in vision_feature_layer]
        # For default; crop CLS from each hidden state in the hidden state pool
        if vision_feature_select_strategy == "default":
            hs_pool = [hs[:, 1:] for hs in hs_pool]
            att_pool = [hs[:, 1:] for hs in att_pool]
        selected_image_feature = torch.cat(hs_pool, dim=-1)
        selected_attention = torch.cat(att_pool, dim=-1)

    image_features = self.multi_modal_projector(selected_image_feature)
    if "image_sizes" in kwargs:
        split_sizes = [
            (height // self.vision_tower.patch_size) * (width // self.vision_tower.patch_size)
            for height, width in kwargs["image_sizes"]
        ]
        image_features = torch.split(image_features.squeeze(0), split_sizes)
        attentions = torch.split(selected_attention.squeeze(0), split_sizes)
    else:
        image_features = list(image_features)
        attentions = list(selected_attention)
    return image_features, attentions


def bind_get_img_features(model):
    model.model.get_image_features = MethodType(
        new_get_img_features, model.model
    )


def apply_vispruner(inputs, model, args):
    # print(inputs["input_ids"].shape, inputs["input_ids"])
    first_img_idx, last_img_idx = get_first_and_last_img_token_idx(inputs["input_ids"], img_token_id=32000)
    inputs["input_ids"] = torch.cat([inputs["input_ids"][:,:first_img_idx+min(last_img_idx-first_img_idx+1, args.retain_img_tokens)], inputs["input_ids"][:,last_img_idx+1:]], dim=1)
    inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape).to(inputs["input_ids"].device)
        
    pixel_values = inputs.pop("pixel_values")
    inputs_embeds = model.model.get_input_embeddings()(inputs["input_ids"])

    # print(inputs_embeds.shape)
    # print(pixel_values.shape)
    vision_feature_layer = model.config.vision_feature_layer
    vision_feature_select_strategy = "full"

    image_features, image_attentions = model.model.get_image_features(
        pixel_values=pixel_values,
        vision_feature_layer=vision_feature_layer,
        vision_feature_select_strategy=vision_feature_select_strategy,
        image_sizes=None,
    )

    image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)[1:,:]
    image_attentions = torch.cat(image_attentions, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
    cls_attention = image_attentions.mean(dim=0)[0][1:]

    # print(cls_attention.shape)
    # print(image_features.shape, image_attentions.shape)

    important_token_idx = torch.argsort(cls_attention, descending=True)[:args.important_tokens]
    # print(important_token_idx)

    all_ids = torch.arange(576, device=important_token_idx.device)
    remaining_idx = all_ids[~torch.isin(all_ids, important_token_idx)]
    # print(remaining_idx.shape)

    ############################################################################
    # Similarity-based Duplication Removal
    r = args.removal_number
    while r > 0:
        remaining = image_features[remaining_idx,:]
        a, b = remaining[::2], remaining[1::2]
        score = a @ b.transpose(-1, -2)
        score = score.max(dim=-1).values

        # print(score.argsort(dim=-1, descending=True))
        diverse_idx = score.argsort(dim=-1, descending=True)[r:]
        remaining_idx = torch.cat([remaining_idx[::2][diverse_idx], remaining_idx[1::2]], dim=-1)
        # print(remaining_idx.shape)
        r = min(r, remaining_idx.shape[0] - args.diverse_tokens)
        # print(score.shape)
        # print(remaining.shape)
        # break
    selected_idx = torch.cat([important_token_idx, remaining_idx], dim=0)
    selected_idx, _ = torch.sort(selected_idx)
    # print(selected_idx)
    ############################################################################

    special_image_mask = model.model.get_placeholder_mask(
        inputs["input_ids"], inputs_embeds=inputs_embeds, image_features=image_features[selected_idx,:]
    )
    inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features[selected_idx,:])
    inputs["inputs_embeds"] = inputs_embeds
    # print(inputs.keys())
    # print(inputs["input_ids"].shape, inputs["input_ids"])


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
        inputs = processor(images=image, text=text_prompt, return_tensors='pt').to(device)
        
        apply_vispruner(inputs, model, args)
        # print(inputs.keys())

        output = model.generate(**inputs, **generate_config)
        output_text = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        response_answer = output_text.split("ASSISTANT:")[-1].strip()

        response_answer_reformat = response_answer.lower()
        # print(type(response_answer_reformat), response_answer_reformat)
        if answer in response_answer_reformat:
            correct_ans += 1.0
        
    end_time = time.time()
    pope_acc = correct_ans / len(pope_dataset)
    print(f"POPE Evaluation result: {pope_acc:.4f}, use time: {(end_time - start_time):.4f}")
    
    with open(args.output_file, "a", encoding="utf-8") as f:
        f.write("-"*30 + "Start llava vispruner POPE" + "-"*30 + "\n")
        f.write(f"retain_img_tokens: {args.retain_img_tokens}\n")
        f.write(f"Important_tokens: {args.important_tokens}\n")
        f.write(f"Diverse tokens: {args.diverse_tokens}\n")
        f.write(f"POPE Evaluation result: {pope_acc:.4f}\n")
        f.write(f"Use time: {(end_time - start_time):.4f}\n")
        f.write("-"*30 + "End llava vispruner POPE" + "-"*30 + "\n")
        f.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset settings
    parser.add_argument('--pope_path', type=str, default='lmms-lab/POPE')
    parser.add_argument('--pope_catagory', type=str, default='adversarial', choices=['adversarial', 'popular', 'random'])
    parser.add_argument('--pope_max_new_token', type=int, default=8)

    # model_info
    parser.add_argument('--model_path', type=str, default='llava-hf/llava-1.5-7b-hf')
    parser.add_argument('--retain_img_tokens', type=int, default=64)
    parser.add_argument('--vispruner_ratio', type=float, default=0.5)
    parser.add_argument('--removal_number', type=int, default=64)

    parser.add_argument('--output_file', type=str, default="./llava_vispruner.txt")

    args = parser.parse_args()
    print(args)

    args.important_tokens = int(args.vispruner_ratio * args.retain_img_tokens)
    args.diverse_tokens = args.retain_img_tokens - args.important_tokens

    # Load model
    print("-"*30+"Load model"+"-"*30)
    model, tokenizer, processor = load_llava(args.model_path)
    device = model.device
    print(f"device: {device}")
    # print(model)

    print("-"*30+"Binding model get_img_features"+"-"*30)
    bind_get_img_features(model)

    # POPE Evaluation
    print("-"*30+"POPE Evaluation"+"-"*30)
    eval_pope(model, tokenizer, processor, args)
    

