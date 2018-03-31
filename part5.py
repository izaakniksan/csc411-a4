import matplotlib.pyplot as plt

from tictactoe import *


def test_hidden_size(hidden_size):
    np.random.seed(11)
    torch.manual_seed(11)
    print("Testing Hidden Layer Size of ", hidden_size)
    env = Environment()
    policy = Policy(hidden_size=hidden_size)
    avg_return = train(policy, env)
    plt.plot((np.arange(len(avg_return)) + 1) * 1000, avg_return)
    plt.xlabel('Episodes')
    plt.ylabel('Average Return')
    plt.show()
    print('Final: ', win_rate(policy, env))


def win_rate(policy, env, n_games=100000):
    wins = 0
    for i in range(n_games):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
        wins += status == Environment.STATUS_WIN
    return wins / n_games


def examine_performance(policy, env, n_games=100, n_shown=5):
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
    print('{:5} GAMES WON'.format(wins))
    print('{:5} GAMES LOST'.format(losses))
    print('{:5} GAMES TIED'.format(ties))
    for i in range(n_shown):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            env.render()
        print()


def invalid_moves(policy, env, n_games=100):
    invalid = 0
    for i in range(n_games):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            invalid += status == Environment.STATUS_INVALID_MOVE
    return invalid / n_games


def main():
    print('Enter Part (a, b, c, or d):')
    part = input()

    if part == 'a':
        test_hidden_size(32)

    if part == 'b':
        test_hidden_size(32)
        test_hidden_size(64)
        test_hidden_size(128)

    if part == 'c':
        test_hidden_size(32)
        env = Environment()
        policy = Policy(hidden_size=32)
        episodes = []
        invalids = []
        import glob
        for file in glob.glob('ttt/*'):
            policy.load_state_dict(torch.load(file))
            episodes.append(int(file[11:-4]))
            invalids.append(invalid_moves(policy, env))
        plt.plot(*zip(*sorted(zip(episodes, invalids))))
        plt.xlabel('Episodes')
        plt.ylabel('Average Number of Invalid Moves')
        plt.show()

    if part == 'd':
        test_hidden_size(32)
        env = Environment()
        policy = Policy(hidden_size=32)
        policy.load_state_dict(torch.load('ttt/policy-50000.pkl'))
        examine_performance(policy, env)


if __name__ == '__main__':
    main()
