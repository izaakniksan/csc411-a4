import matplotlib.pyplot as plt

from tictactoe import *
test=1

def main():
    env = Environment()
    policy = Policy(hidden_size=32)
    policy.load_state_dict(torch.load('ttt\policy-50000.pkl'))
    pr=first_move_distr(policy, env)
    pr=np.array((pr))
    pr.shape=(3,3)
    print('The final first move distribution is:')
    print(pr,'\n')
    
    episodes,one,two,three,four,five,six,seven,eight,nine=[],[],[],[],[],[],[],[],[],[]
    import glob
    for file in glob.glob('ttt/*'):
        policy.load_state_dict(torch.load(file))
        episodes.append(int(file[11:-4]))
        
        pr=first_move_distr(policy, env)
        pr=np.array(pr)
        one.append(pr[0][0])
        two.append(pr[0][1])
        three.append(pr[0][2])
        four.append(pr[0][3])
        five.append(pr[0][4])
        six.append(pr[0][5])
        seven.append(pr[0][6])
        eight.append(pr[0][7])
        nine.append(pr[0][8])
    
    print('Graph of the changing first move distribution throughout training:')
    plt.figure(figsize=(12,5))
    plt.plot(*zip(*sorted(zip(episodes, one))),label='Space 0')
    plt.plot(*zip(*sorted(zip(episodes, two))),label='Space 1')
    plt.plot(*zip(*sorted(zip(episodes, three))),label='Space 2')
    plt.plot(*zip(*sorted(zip(episodes, four))),label='Space 3')
    plt.plot(*zip(*sorted(zip(episodes, five))),label='Space 4')
    plt.plot(*zip(*sorted(zip(episodes, six))),label='Space 5')
    plt.plot(*zip(*sorted(zip(episodes, seven))),label='Space 6')
    plt.plot(*zip(*sorted(zip(episodes, eight))),label='Space 7')
    plt.plot(*zip(*sorted(zip(episodes, nine))),label='Space 8')
    plt.xlabel('Episode')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()

    

    
    