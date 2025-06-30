import os
import pickle
# Adapted from ARM
# Source: GitHub - stepjam/ARM: Q-attention (within the ARM system) and coarse-to-fine Q-attention (within C2F
# License: https://github.com/stepjam/ARM/LICENSE

import numpy as np

import torch
from helpers.clip.core.clip import build_model, load_clip, tokenize

LOW_DIM_SIZE = 8

from pdb import set_trace

# TODO: modify your path to PPI
path_to_PPI = "/PATH/TO/PPI"

# TODO: add insrtuctions
def create_intruction():
    
    instruction_list = [
        "move the fruit tray",
        "wipe the plate",
        "carry the tray",
        "handover and insert the plate",
        "scan the bottle",
        "press the bottle",
        "wear the scarf",
        "scan the bottle single arm"
    ]

    return instruction_list

def get_embedding(language_instruction):
    model, _ = load_clip('RN50', jit=False, device='cpu')
    clip_model = build_model(model.state_dict())
    clip_model.to('cpu')
    tokens = tokenize([language_instruction]).numpy()
    token_tensor = torch.from_numpy(tokens).to('cpu')
    lang_feats, lang_embs = clip_model.encode_text_with_embeddings(token_tensor)
    return lang_feats[0].float().detach().cpu().numpy()

def main():
    instruction_embedding_dict = {}
    instruction_list = create_intruction()
    for instruction in instruction_list:
        instruction_embedding_dict[instruction]=get_embedding(instruction)
        #set_trace()
        print(instruction)
    with open(f'{path_to_PPI}/pretrained_models/instruction_embeddings_real.pkl','wb') as f:
        pickle.dump(instruction_embedding_dict,f)
        
if __name__ == "__main__":
    main()