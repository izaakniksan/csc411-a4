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
    return wins / n_games, losses / n_games, ties / n_games


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
        opponent_turn = int(np.random.random() > 0.5) + 1
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


def self_train(policy, env, gamma=1.0, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.SGD(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    avg_rewards = []

    for i_episode in range(1, 50001):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        tracked_turn = int(np.random.random() > 0.5) + 1
        if tracked_turn == 2:
            action, _ = select_action(policy, state)
            state, status, done = env.step(action)
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.step(action)
            if not env.done and env.turn != tracked_turn:
                action, _ = select_action(policy, state)
                state, s2, done = env.step(action)
                if done:
                    if s2 == Environment.STATUS_WIN:
                        status = Environment.STATUS_LOSE
                    elif s2 == Environment.STATUS_TIE:
                        status = Environment.STATUS_TIE
                    else:
                        raise ValueError("???")
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
                       "ttt3/policy-{}.pkl".format(i_episode))

        if i_episode % 1 == 0:  # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return avg_rewards


def plot_wins(opponent, folder='ttt2'):
    env = Environment()
    policy = Policy(hidden_size=32)
    episodes = []
    win_ratio = []
    lose_ratio = []
    tie_ratio = []
    import glob
    for file in glob.glob(folder + '/*'):
        policy.load_state_dict(torch.load(file))
        # get the ratios:
        wins, losses, ties = win_lose_tie_ratios(policy, env, opponent=opponent)

        episodes.append(int(file[12:-4]))

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


def examine_performance(policy, env, n_shown=5, rand=True):
    for i in range(n_shown):
        state = env.reset()
        done = False
        opponent = int(np.random.random() > 0.5) + 1
        if opponent == 1:
            state, status, done = env.random_step()
            env.render()
        while not done:
            action, logprob = select_action(policy, state)
            if rand:
                state, status, done = env.play_against_random(action)
            else:
                state, status, done = env.step(action)
            env.render()
        print()


def main():
    part = input()
    env = Environment()
    policy = Policy(hidden_size=32)
    if part == 'a':
        avg_return = train(policy, env)
        plt.plot((np.arange(len(avg_return)) + 1) * 1000, avg_return)
        plt.xlabel('Episodes')
        plt.ylabel('Average Return')
        plt.show()
        plot_wins(1)
        plot_wins(2)
        policy.load_state_dict(torch.load('ttt2/policy-50000.pkl'))
        examine_performance(policy, env)
    if part == 'b':
        policy.load_state_dict(torch.load('ttt2/policy-50000.pkl'))
        avg_return = self_train(policy, env)
        plt.plot((np.arange(len(avg_return)) + 1) * 1000, avg_return)
        plt.xlabel('Episodes')
        plt.ylabel('Average Return')
        plt.show()
        plot_wins(1, 'ttt3')
        plot_wins(2, 'ttt3')


if __name__ == '__main__':
    main()
