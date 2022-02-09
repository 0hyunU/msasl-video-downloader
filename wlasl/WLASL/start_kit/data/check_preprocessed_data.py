import os
import glob
import pickle
import matplotlib.pyplot as plt

def remove_short_video_keypoint(abs_file_path):
    os.remove(abs_file_path)

def check_preprocessed_data():

    frame_list = []
    for i in glob.glob("./train_test/**/*.pickle",recursive=True):
        a = pickle.load(open(i, 'rb'))
        if a.avail_frame_len < 10: 
            print(i, a.avail_frame_len)
            short_video_path = os.path.abspath(i)
            remove_short_video_keypoint(short_video_path)
            continue

        frame_list.append(a.avail_frame_len)
    print(frame_list)
    print(len(frame_list))
    
    plt.hist(frame_list,bins=20)
    plt.xlabel("available frame")
    plt.savefig("available_frame.png")
    plt.show()    

if __name__ == "__main__":
    check_preprocessed_data()