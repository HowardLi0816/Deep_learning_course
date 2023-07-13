import random
import matplotlib.pyplot as plt

def coin_1_exp(head_prob, trail, plt_head_run_len = False):
    if head_prob>1 or head_prob<0:
        print("probability of a Bernoulli trail must between 0 and 1!")
    else:
        coin_result = []
        for i in range(trail):
            head = random.random()
            if head <= head_prob:
                coin_result.append(1)
            else:
                coin_result.append(0)
        num_head = coin_result.count(1)
        longest_head = 0
        continue_head = 0
        head_run_list = []
        for item in coin_result:
            if item == 1:
                continue_head += 1
                if continue_head > longest_head:
                    longest_head = continue_head
            else:
                if continue_head != 0:
                    head_run_list.append(continue_head)
                continue_head = 0
        if plt_head_run_len:
            plt.figure()
            plt.hist(head_run_list)
            plt.title("Hist of head run length when tossing %d trails with P(Head)=%3.1f" % (trail, head_prob))
            plt.xlabel("head run length")
            plt.ylabel("times")
            plt.savefig("./fig/%dtrail_head_run_length_hist.png" % trail)


    return num_head, longest_head

def repeat_exp(n_exp, head_prob, trail):
    exp_head = []
    for exp in range(n_exp):
        num_head, _ = coin_1_exp(head_prob, trail)
        exp_head.append(num_head)
    plt.figure()
    plt.hist(exp_head, bins=trail, range=(0, trail))
    plt.title("%d experiences for tossing %d trails with P(Head)=%3.1f" %(n_exp, trail, head_prob))
    plt.xlabel("head num")
    plt.ylabel("times")
    plt.savefig("./fig/%dexp_hist.png" % n_exp)

if __name__ == '__main__':
    #Question(a)
    num_head_50, longest_head_50 = coin_1_exp(0.7, 50)
    print("There are %d heads and the longest run head is %d" % (num_head_50, longest_head_50))

    #Question(b)
    repeat_exp(20, 0.7, 50)
    repeat_exp(100, 0.7, 50)
    repeat_exp(200, 0.7, 50)
    repeat_exp(1000, 0.7, 50)

    #Question(c)
    _, _ = coin_1_exp(0.7, 500, plt_head_run_len=True)






