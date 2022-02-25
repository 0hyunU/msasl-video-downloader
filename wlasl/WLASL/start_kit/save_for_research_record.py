import traceback
from vid import plot_2D_keypoint_every_move
import os
import glob
import pickle
import time
import logging as log
import imageio

from vid.draw_plot import plot_3D_keypoint_every_move

# clear scatter_gif_dir
png_path_for_glob = os.path.join(os.getcwd(),'scatter_gif',"*.png")
[os.remove(i) for i in glob.glob(png_path_for_glob)]
#log.basicConfig(filename = os.path.join(os.getcwd(),'scatter_gif',f'gif_log_{int(time.time())}.log'), level=log.DEBUG)

def main():
    st_time = time.time()
    for i in glob.glob(os.path.join(os.getcwd(),"**","**","*.mp4.pickle")):
        for with_face in [True,False]:
            try:
                assert len(glob.glob(png_path_for_glob)) == 0
                print("processing file: ",i)
                log.info(f"processing file: {i}")
                file_path = os.path.join(i)
                a = pickle.load(open(file_path,'rb'))
                plot_2D_keypoint_every_move(a, with_face=with_face,save=True)

                imlist = []
                for j in glob.glob(png_path_for_glob):
                    imlist.append(imageio.imread(j))

                pickle_name = i.split(os.sep)[-1]
                if not with_face: save_to = os.path.join(os.path.dirname(png_path_for_glob),pickle_name+'.gif')
                else:
                    save_to = os.path.join(os.path.dirname(png_path_for_glob),pickle_name+'_without_face.gif')
        
                imageio.mimsave(save_to,imlist)
                log.info(f"image saved to {save_to}")
                print(f"save_to: {save_to}")

                [os.remove(i) for i in glob.glob(png_path_for_glob)]
            except Exception as e:
                log.error(traceback.format_exc())
                log.error("error OCCUR")
                log.error(f"file path occurred error: {i}")
            finally:
                [os.remove(i) for i in glob.glob(png_path_for_glob)]

    log.info(f"running time: {st_time - time.time()}")

if __name__ =="__main__":
    # main()
    file_path = os.path.join(os.getcwd(),"train_test",'black','06483.mp4')
    print(file_path)
    from vid import aug_vid as av

    from vid.gen_key_arr import KeyArrGen as KAG

    vid_arr = av.vid2arr(file_path)

    a = KAG(vid_arr).get_keyarr()
    print(a.shape)
    plot_3D_keypoint_every_move(a, True, True)
    # plot_2D_keypoint_every_move(a)
    # plot_2D_keypoint_every_move(a, True, True)
    # imlist = []
    # png_path_for_glob = os.path.join(os.getcwd(),'scatter_gif',"*.png")
    # print
    # for j in glob.glob(png_path_for_glob):
    #     imlist.append(imageio.imread(j))
    # imageio.mimsave(os.path.join(os.getcwd(),'scatter_gif','true.gif'),imlist)
    # # [os.remove(i) for i in glob.glob(png_path_for_glob)]