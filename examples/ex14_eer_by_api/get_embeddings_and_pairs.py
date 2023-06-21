import requests
from tqdm import tqdm
import numpy as np

def get_embedding(file_path):
    url = "http://106.14.148.126:8191/get_embedding/file"
    payload={"spkid":"zhaosheng"}
    files=[
    ('wav_file',(file_path.split('/')[-1],open(file_path,'rb'),'application/octet-stream'))
    ]
    headers = {
    'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)'
    }
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return np.array(response.json()["embeddings"][0])

    

if __name__ == "__main__":
    # embeddings = {}
    # all_wavs = []
    # # if npy exist, just load
    # if os.path.exists(f"../../cache/{cfg.NAME}/{cfg.NAME}_embeddings.npy") and cfg.LOAD_NPY:
    #     embeddings = np.load(f"../../cache/{cfg.NAME}/{cfg.NAME}_embeddings.npy",allow_pickle=True).item()
    #     for phone in embeddings:
    #         phone_file_nums = 0
    #         for filename in embeddings[phone]:
    #             all_wavs.append(os.path.join(cfg.DATA_FOLDER,phone,filename))
    #             phone_file_nums += 1
    #     print("Load npy file successfully.")
    # else:
    #     from sb_encoder import generate_embedding as generate_embedding_sb
    #     for phone in tqdm(os.listdir(cfg.DATA_FOLDER)):
    #         embeddings[phone] = {}
    #         phone_path = os.path.join(cfg.DATA_FOLDER, phone)
    #         for file in os.listdir(phone_path):
    #             try:
    #                 file_path = os.path.join(phone_path, file)
    #                 filename = os.path.splitext(file)[0]
    #                 file_size = os.path.getsize(file_path)
    #                 # if > 20 MB skip
    #                 if file_size > 20 * 1024 * 1024:
    #                     continue
    #                 embeddings[phone][filename]=generate_embedding_sb(file_path,cfg.SR).detach().cpu().numpy()
    #                 all_wavs.append(file_path)
    #             except Exception as e:
    #                 logger.error(f"Error in {file_path}: {e}")
    #                 continue
    #     os.makedirs(f"../../cache/{cfg.NAME}",exist_ok=True)
    #     np.save(f"../../cache/{cfg.NAME}/{cfg.NAME}_embeddings.npy",embeddings)
    # print("Done!")
    a = get_embedding("/home/zhaosheng/voiceprint-recognition-system/utils/dataset/cti_mini_test_8k/13055014477/cti_record_11003_1642640137635828_1-dcafce9b-5c47-47b3-8427-b35611fb464a.wav")
    print(a.shape)