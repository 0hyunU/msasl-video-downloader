import os
import glob
import shutil
import json
from textwrap import indent

def test_set_mkdir():

    if os.path.exists("./train_test"):
        os.mkdir("./train_test")



def split_train_test_videos():
    class_list = os.listdir("./videos")
    test_set_mkdir()
    for d in class_list:
        files = glob.glob(os.path.join("./videos",d,"*"))
        # videos 데이터 중 10개 이상 데이터를 보유하고 있는 클래스를
        # train_test 디렉터리로 카피
        if len(files) > 10: 
            try:
                print(d)
                shutil.copytree(f"./videos/{d}",f"./train_test/{d}")
                print(f"{d} copy success")
            except Exception as e:
                print(e)


def make_data_json():
    data_dir = "./train_test"

    data_dict = []

    for i in os.listdir(data_dir):
        gloss = i
        videopath_list = []

        #is not dir, is file
        if os.path.isfile(os.path.join(data_dir,i)): continue

        for j in os.listdir(os.path.join(data_dir,i)):
            if ".mp4" in os.path.abspath(j)[-4:]:
                 data_dict.append({'gloss': gloss, "data_path" :os.path.abspath(os.path.join(data_dir,i,j))})
        
       

    path_json_save = os.path.join(data_dir,"data.json")
    with open(path_json_save, 'w') as f:
        json.dump(data_dict,f)


if __name__ == "__main__":
    make_data_json()