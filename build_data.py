from rule_chess_AI import State, AI
import numpy as np
import pickle

piece_indexes = {'r': 0, 'n': 1, 'b': 2, 'q': 3, 'k': 4, 'p': 5, 'R': 6, 'N': 7, 'B': 8, 'Q': 9, 'K': 10, 'P': 11}


def generate_tensor(key):
    board = np.zeros((8, 8, 12))
    rows = key.split('/')
    for r in range(8):
        c = 0
        for piece in rows[r]:
            if piece in '12345678':
                c += int(piece)
            else:
                board[r][c][piece_indexes[piece]] = 1
                c += 1
    return board


Xw = []
yw = []
Xb = []
yb = []


def play():

    ai_white = AI()
    ai_white.make_move()
    ai_black = AI(ai_white.state.key)

    while True:
        message = ai_black.make_move()
        if message == 'Success':
            info = ai_black.state.key
            board = info.split(' ')[0]
            Xb.append(generate_tensor(board))
            yb.append(ai_black.state.value)
            message = ai_white.register_key(info)
            if message != 'Success':
                print('Black tried to cheat', ai_white.state.key, ai_black.state.key)
                break
        else:
            print(message)
            break

        message = ai_white.make_move()
        if message == 'Success':
            info = ai_white.state.key
            board = info.split(' ')[0]
            Xw.append(generate_tensor(board))
            yw.append(ai_white.state.value)
            message = ai_black.register_key(info)
            if message != 'Success':
                print('White tried to cheat', ai_black.state.key, ai_white.state.key)
                break
        else:
            print(message)
            break

for i in range(10):
    print(i)
    play()

pickle.dump(Xb, open('Xb', 'wb'))
pickle.dump(yb, open('yb', 'wb'))
pickle.dump(Xw, open('Xw', 'wb'))
pickle.dump(yw, open('yw', 'wb'))
