import os
k = list()
n = dict()
for a,b,c in os.walk("./videos"):
    print(a,b,c)
    k.append(len(c))

print([i for i in k if i>10])
print(n)
print("class #:",len(k))
print("video #:",sum(k))


for a,b,c in os.walk("./raw_videos"):
    pass
    # print(len(c))
    
for a,b,c in os.walk("./raw_videos_mp4"):
    pass
    # print(len(c))