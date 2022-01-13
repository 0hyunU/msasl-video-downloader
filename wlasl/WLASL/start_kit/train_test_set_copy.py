import os
import glob
import shutil

def test_set_mkdir():

    if os.path.exists("./train_test"):
        os.mkdir("./train_test")


class_list = os.listdir("./videos")
test_set_mkdir()
for d in class_list:
    files = glob.glob(os.path.join("./videos",d,"*"))
    if len(files) > 10: 
        try:
            print(d)
            shutil.copytree(f"./videos/{d}",f"./train_test/{d}")
            print(f"{d} copy success")
        except Exception as e:
            print(e)