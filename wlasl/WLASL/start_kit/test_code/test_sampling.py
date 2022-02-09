import random

def check_random_rate_hist():
    a = [random.uniform(0,0.2) for i in range(1000)]
    import matplotlib.pyplot as plt
    plt.hist(a,bins=10)
    plt.show()