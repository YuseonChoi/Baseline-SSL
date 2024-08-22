import os
import pickle
from tqdm import tqdm

print(os.getcwd())

load_folder = "./data/train-backup/"
save_folder = "./data/train/"
file_list = os.listdir(load_folder)

for file in tqdm(file_list):
    data = open(os.path.join(load_folder,file),'rb')
    dataPickle = data.read()
    data.close()
    dict = pickle.loads(dataPickle)
    keys = list(dict.keys())
    for k in keys:
        if k not in ['mic_vad_sources', 'DOA']:
            dict.pop(k)

    # save files
    data = open(os.path.join(save_folder, file), 'wb')
    data.write(pickle.dumps(dict))
    data.close()
