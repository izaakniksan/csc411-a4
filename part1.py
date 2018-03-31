import tictactoe

def main():

    p1_env=tictactoe.Environment()
    print('First turns:')
    p1_env.step(0)
    p1_env.step(1)
    p1_env.render()
    print('Second turns:')
    p1_env.step(8)
    p1_env.step(5)
    p1_env.render()
    print('Third turn (x wins):')
    p1_env.step(4)
    p1_env.render()
    
    
if __name__ == '__main__':
    main()

