import numpy as np

    
def multi_armed_bandit(k):
    return np.random.randn(k)*2 + 10

def calc_reward(A, bandit):
    return np.random.randn(1) + bandit[A]

def main():
    k = 10
    bandit = multi_armed_bandit(k)

    Q = np.zeros(10)
    N = np.zeros(10)
    epsilon = 0.1
    
    n = 10000
    for i in range(n):
        if (np.random.rand(1) < epsilon):
            A = np.argmax(Q)
        else:
            A = np.random.randint(0, k, 1)

        R = calc_reward(A, bandit)
        N[A] = N[A] + 1
        Q[A] = Q[A] + 1/N[A]*(R-Q[A])

    print(np.sum(Q)/n)

if __name__ == "__main__":
    main()
