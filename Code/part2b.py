import tictactoe  

state=np.array([0,1,2]*3)
print('The grid array format is:')
print(state)
state = torch.from_numpy(state).long().unsqueeze(0)
state = torch.zeros(3,9).scatter_(0,state,1)
print('\nThe 27-dimensional array format is:')
print(state)