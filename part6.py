import matplotlib.pyplot as plt

from tictactoe import *
from part5 import *

def win_lose_tie_ratios(policy, env, n_games=100):
    '''
    Evaluate and return the win, lose, and tie rates for a given policy.
    '''
    wins, losses, ties = 0, 0, 0
    for i in range(n_games):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
        wins += status == Environment.STATUS_WIN
        losses += status == Environment.STATUS_LOSE
        ties += status == Environment.STATUS_TIE
    return wins/n_games, losses/n_games, ties/n_games
    
def main():
    env = Environment()
    policy = Policy(hidden_size=32)
    episodes = []
    win_ratio = []
    lose_ratio=[]
    tie_ratio=[]

    import glob
    for file in glob.glob('ttt/*'):
        policy.load_state_dict(torch.load(file))
        #get the ratios:
        wins,losses,ties = win_lose_tie_ratios(policy,env)
        
        episodes.append(int(file[11:-4]))

        win_ratio.append(wins)
        lose_ratio.append(losses)
        tie_ratio.append(ties)
        
    plt.plot(*zip(*sorted(zip(episodes, win_ratio))),label='Win Ratio')
    plt.plot(*zip(*sorted(zip(episodes, lose_ratio))),label='Loss Ratio')
    plt.plot(*zip(*sorted(zip(episodes, tie_ratio))),label='Tie Ratio')
    plt.xlabel('Episode')
    plt.ylabel('Ratio')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
