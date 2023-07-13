import random
import matplotlib.pyplot as plt

def small_exp(exp_num, trail, x):
    exp_N = []
    for i in range(exp_num):
        trail_count = []
        for j in range(int(trail)):
            #print("exp: ", i, "trail: ", j)
            sum_x = 0
            count = 0
            while sum_x <= x:
                ran = random.random()
                sum_x += ran
                count += 1
            trail_count.append(count)
        exp_N.append(min(trail_count))
    N = []
    for i in exp_N:
        if i not in N:
            N.append(i)
        else:
            count
    #print(exp_N)
    plt.figure()
    plt.hist(exp_N, bins=len(N))
    plt.title("Hist of %d trails" % exp_num)
    plt.xlabel("random variable N")
    plt.ylabel("times")
    plt.savefig("./fig2/%dtrail_hist.png" % exp_num)

    expect = 0
    for n in N:
        expect += n * exp_N.count(n) / len(exp_N)
    return expect



if __name__ == '__main__':
    ex_100 = small_exp(100, 1, 4)
    ex_1000 = small_exp(1000, 1, 4)
    ex_10000 = small_exp(10000, 1, 4)

    print("expected value of 100 trail is: ", ex_100)
    print("expected value of 1000 trail is: ", ex_1000)
    print("expected value of 10000 trail is: ", ex_10000)