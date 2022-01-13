import os
k = list()
n = dict()
for a,b,c in os.walk("./videos"):
    print(a,b,c)
    k.append(len(c))

print([i for i in k if i>10])
print(n)
print(sum(k))

for a,b,c in os.walk("./raw_videos"):
    print(len(c))
    
for a,b,c in os.walk("./raw_videos_mp4"):
    print(len(c))