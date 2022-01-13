import json
import os

labeling_data_path = "C:/Users/user/temp/wlasl/WLASL/start_kit/WLASL_v0.3.json"
video_file_path = "C:/Users/user/temp/wlasl/WLASL/start_kit/videos"

def create_directory(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    except OSError as e:
        print(e)


def main():
        
    with open(labeling_data_path, 'r') as f:
        a = json.load(f) 
        exi = 0
        notexi=0
        for i in a:
            for j in i['instances']:

                try:
                    video_name = str(j['video_id']) + '.mp4'
                    src_path = os.path.join(video_file_path, video_name)
                    des_path = os.path.join(video_file_path, i['gloss'])
                    
                    #분류 디렉토리 생성
                    create_directory(des_path)

                    if os.path.exists(src_path):
                        print("video exist")
                        os.replace(src_path, 
                        dst = os.path.join(des_path, video_name))

                        exi +=1
                    else:
                        notexi +=1
                        # pass
                        # print("not exist")
                except Exception as e:
                    #print(e)
                    continue

        print("exist files:" ,exi)
        print("not exitst files:", notexi)



if __name__ == "__main__":
    main()
