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
    # videos 데이터 중 10개 이상 데이터를 보유하고 있는 클래스를
    # train_test 디렉터리로 카피
    if len(files) > 10: 
        try:
            print(d)
            shutil.copytree(f"./videos/{d}",f"./train_test/{d}")
            print(f"{d} copy success")
        except Exception as e:
            print(e)