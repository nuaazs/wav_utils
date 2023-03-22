import os
import numpy as np
import paddle
import metrics

import torch
from tqdm import tqdm

# set seed
import random
random.seed(0)
paddle.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# save log file
fh = logging.FileHandler('log.txt')
fh.setLevel(logging.INFO)
logger.addHandler(fh)
import cfg

if "paddle" in cfg.NAME:
    similarity = paddle.nn.CosineSimilarity(axis=-1, eps=1e-6)
elif "speechbrain" in cfg.NAME:
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
else:
    similarity = None

def get_embedding(file_path,embeddings):
    phone = file_path.split('/')[-2]
    filename = file_path.split('/')[-1].split('.')[0]
    return phone,filename,embeddings[phone][filename]

def get_score(features1, features2):
    score = float(paddle.dot(features1.squeeze(), features2.squeeze()))
    return score

if __name__ == "__main__":
    embeddings = {}
    all_wavs = []

    # if npy exist, just load
    if os.path.exists(f"./npys/{cfg.NAME}/{cfg.NAME}_embeddings.npy") and cfg.LOAD_NPY:
        embeddings = np.load(f"./npys/{cfg.NAME}/{cfg.NAME}_embeddings.npy",allow_pickle=True).item()
        for phone in embeddings:
            phone_file_nums = 0
            for filename in embeddings[phone]:
                all_wavs.append(os.path.join(cfg.DATA_FOLDER_8k_vad,phone,filename))
                phone_file_nums += 1
                # if phone_file_nums>=2:
                #     break
        print("Load npy file successfully.")
    else:
        from sb_encoder import generate_embedding as generate_embedding_sb
        from encoder import generate_embedding
        for phone in tqdm(os.listdir(cfg.DATA_FOLDER_8k_vad)):
            embeddings[phone] = {}
            phone_path = os.path.join(cfg.DATA_FOLDER_8k_vad, phone)
            for file in os.listdir(phone_path):
                try:
                    file_path = os.path.join(phone_path, file)
                    filename = os.path.splitext(file)[0]
                    file_size = os.path.getsize(file_path)
                    # if > 5 MB skip
                    if file_size > 20 * 1024 * 1024:
                        continue
                    if "paddle" in cfg.NAME:
                        embeddings[phone][filename] = generate_embedding(file_path).detach().cpu().numpy()
                    elif "speechbrain" in cfg.NAME:
                        embeddings[phone][filename]=generate_embedding_sb(file_path).detach().cpu().numpy()
                    else:
                        print("Please check your config file.")
                    all_wavs.append(file_path)
                except Exception as e:
                    logger.error(f"Error in {file_path}: {e}")
                    continue
        np.save(f"./npys/{cfg.NAME}/{cfg.NAME}_embeddings.npy",embeddings)
    
    all_wavs = sorted(all_wavs)

    # if os.path.exists(f"./npys/{cfg.NAME}/{cfg.NAME}_wav_pairs.npy") and cfg.LOAD_NPY:
    #     wav_pairs = np.load(f"./npys/{cfg.NAME}/{cfg.NAME}_wav_pairs.npy",allow_pickle=True).tolist()
    #     print("Load npy file (wav pairs) successfully.")
    # else:
    #     wav_pairs = []
        
    #     for wav1 in tqdm(all_wavs):
    #         for wav2 in all_wavs:
    #             if wav1 == wav2:
    #                 continue
    #             if (wav1, wav2) in wav_pairs or (wav2, wav1) in wav_pairs:
    #                 continue
    #             wav_pairs.append((wav1, wav2))
    #     wav_pairs = np.array(wav_pairs)
    #     np.save(f"./npys/{cfg.NAME}/{cfg.NAME}_wav_pairs.npy",wav_pairs)

    # print(f"Total wav pairs: {len(wav_pairs)}")