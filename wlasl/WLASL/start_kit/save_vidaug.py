import json
import pickle
import time
import os
from vid.aug_vid import load_data_json
from vid.gen_key_arr import KeyArrGen
from vid.vidaug import VidAug


def main(data="train"):
    aug_time = []
    s_time = time.time()
    try: 
        for i in load_data_json(data):
            j = 0
            while j < len(i['files']):
                count = 0
                print(i['files'][j])
                while count < 5:
                    st_time = time.time()
                    # print(i['gloss'], len(i['files']))
                    # print(i['files'][j])

                    key_arr = save_augvid2keyarr(i['files'][j], count, data)
                    if key_arr is None: continue
                    print(key_arr.shape)
                    aug_time.append({len(key_arr): time.time() - st_time})
                    count += 1
                j += 1
    except KeyboardInterrupt as e:
        print(aug_time)
    finally:
        with open(f"./processing_time_{data}.json", 'w') as f:
            json.dump(aug_time,f)
        print(time.time() - s_time)

def save_without_aug(dir = "test"):
    process_time = []
    s_time = time.time()
    for i in load_data_json(dir):
        print(i)
        for j in i['files']:
            print(j)
            st_time = time.time()
            key_arr = save_purevid2keyarr(j)
            if key_arr is None: continue
            print(key_arr.shape)
            process_time.append({len(key_arr): time.time() - st_time})
    
    with open(f"./{dir}_processing_time.json", 'w') as f:
            json.dump(process_time,f)
    
    print(time.time() - s_time)

def save_purevid2keyarr(file_path):
    vid = VidAug(file_path)
    vid_name = file_path.split(os.sep)[-1]
    key_arr = KeyArrGen(vid.vid_arr).get_keyarr()

    if len(key_arr) < 15: return None

    save_dir = aug_valid_n_test_mkdir(file_path.split(os.sep)[-2])
    save_path = os.path.join(save_dir,vid_name)+ ".pickle"
    pickle.dump(key_arr,open(save_path,'wb'))

    return key_arr

def aug_train_mkdir(gloss, dir="train"):
    path = f"./{dir}/{gloss}"
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def aug_valid_n_test_mkdir(gloss):
    path = f"./test/{gloss}"
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def save_augvid2keyarr(file_path, count, dir="train"):

    # vid aug
    vid = VidAug(file_path)
    vid_name = file_path.split(os.sep)[-1]
    aug_vid = vid.get_randomly_aug_vid()

    # gen keypoint
    key_arr = KeyArrGen(aug_vid).get_keyarr()
    
    #print(key_arr.shape)
    if len(key_arr) < 15: return None


    save_dir = aug_train_mkdir(file_path.split(os.sep)[-2], dir)
    save_path = os.path.join(save_dir,vid_name)+ f"-{count}.pickle"
    pickle.dump(key_arr,open(save_path,'wb'))

    return key_arr



if __name__ =="__main__":
    dir = ["own_vid",'more_vid','train']
    for i in dir:
        print(i)
        save_without_aug(i)
    