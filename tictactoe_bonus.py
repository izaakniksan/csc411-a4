import matplotlib.pyplot as plt

from tictactoe import *

def win_lose_tie_ratios(policy, env, n_games=100, opponent=2):
    '''
    Evaluate and return the win, lose, and tie rates for a given policy.
    '''
    wins, losses, ties = 0, 0, 0
    for i in range(n_games):
        state = env.reset()
        done = False
        if opponent == 1:
            state, status, done = env.random_step()
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action, opponent)
        wins += status == Environment.STATUS_WIN
        losses += status == Environment.STATUS_LOSE
        ties += status == Environment.STATUS_TIE
    return wins/n_games, losses/n_games, ties/n_games

def train(policy, env, gamma=1.0, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    avg_rewards = []

    for i_episode in range(1, 50001):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        opponent_turn = int(np.random.random() > 0.5)
        if opponent_turn == 1:
            state, status, done = env.random_step()
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action, opponent_turn)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            avg_rewards.append(running_reward / log_interval)
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt2/policy-{}.pkl".format(i_episode))

        if i_episode % 1 == 0:  # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return avg_rewards


def win_rate(policy, env, n_games=100000):
    wins = 0
    for i in range(n_games):
        state = env.reset()
        done = False
        if i % 2 == 0:
            state, status, done = env.random_step()
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
        wins += status == Environment.STATUS_WIN
    return wins / n_games


def plot_wins(opponent):
    env = Environment()
    policy = Policy(hidden_size=32)
    episodes = []
    win_ratio = []
    lose_ratio = []
    tie_ratio = []
    import glob
    for file in glob.glob('ttt/*'):
        policy.load_state_dict(torch.load(file))
        # get the ratios:
        wins, losses, ties = win_lose_tie_ratios(policy, env, opponent=opponent)

        episodes.append(int(file[11:-4]))

        win_ratio.append(wins)
        lose_ratio.append(losses)
        tie_ratio.append(ties)

    plt.plot(*zip(*sorted(zip(episodes, win_ratio))), label='Win Ratio')
    plt.plot(*zip(*sorted(zip(episodes, lose_ratio))), label='Loss Ratio')
    plt.plot(*zip(*sorted(zip(episodes, tie_ratio))), label='Tie Ratio')
    plt.xlabel('Episode')
    plt.ylabel('Ratio')
    plt.legend()
    plt.show()


def main():
    env = Environment()
    policy = Policy(hidden_size=32)
    avg_return = train(policy, env)
    plt.plot((np.arange(len(avg_return)) + 1) * 1000, avg_return)
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.show()
    plot_wins(1)
    plot_wins(2)




if __name__ == '__main__':
    main()
