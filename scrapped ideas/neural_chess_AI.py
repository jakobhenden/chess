import numpy as np
import pickle
import keras

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


def is_in_bounds(pos):
    return -1 < pos[0] < 8 and -1 < pos[1] < 8


def generate_key(board, move, turn, castle, halv_moves, moves):
    new_board = board.copy()
    new_board[move[0][0]][move[0][1]] = '.'
    capture = board[move[1][0]][move[1][1]] != '.'
    piece = board[move[0][0]][move[0][1]]
    pawn_move = piece == 'P' or piece == 'p'

    # Upgrade pawns
    if pawn_move and (move[1][0] == 0 or move[1][0] == 7):
        new_board[move[1][0]][move[1][1]] = 'Q' if piece == 'P' else 'q'
    else:
        new_board[move[1][0]][move[1][1]] = piece

    # En passant
    en_passant = '-'
    if pawn_move and abs(move[0][0] - move[1][0]) == 2:
        en_passant = '(5,{})'.format(move[0][1]) if piece == 'P' else '(2,{})'.format(move[0][1])

    # Change turns
    if turn == 'b':
        turn = 'w'
        moves = int(moves) + 1
    else:
        turn = 'b'

    if capture or pawn_move:
        halv_moves = '0'
    else:
        halv_moves = int(halv_moves) + 1

    # Castling
    if castle:
        if piece == 'K':
            castle = castle.replace('K', '').replace('Q', '')
        elif piece == 'k':
            castle = castle.replace('k', '').replace('q', '')
        elif piece == 'R':
            if move[0][0] == 7:
                if move[0][1] == 7:
                    castle = castle.replace('K', '')
                elif move[0][1] == 0:
                    castle = castle.replace('Q', '')
        elif piece == 'r':
            if move[0][0] == 0:
                if move[0][1] == 7:
                    castle = castle.replace('k', '')
                elif move[0][1] == 0:
                    castle = castle.replace('q', '')

    key = ''
    for row in new_board:
        empty_count = 0
        for piece in row:
            if piece == '.':
                empty_count += 1
            else:
                if empty_count != 0:
                    key += str(empty_count)
                    empty_count = 0
                key += piece
        if empty_count != 0:
            key += str(empty_count)
        key += '/'
    return key[:-1] + ' ' + turn + ' ' + castle + ' ' + en_passant + ' ' + str(halv_moves) + ' ' + str(moves)


def generate_board(key):
    board = []
    for row in key.split('/'):
        board_row = []
        for piece in row:
            if piece in '12345678':
                for i in range(int(piece)):
                    board_row.append('.')
            else:
                board_row.append(piece)
        board.append(board_row)
    return np.array(board)


class State:
    def __init__(self, key='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        self.key = key
        info = key.split(' ')
        self.board = generate_board(info[0])
        self.turn, self.castle, self.en_passant = info[1], info[2], info[3]
        self.own_pieces = 'RNBQKP' if self.turn == 'w' else 'rnbqkp'
        self.opp_pieces = 'rnbqkp' if self.turn == 'w' else 'RNBQKP'
        self.turn_int = 1 if self.turn == 'w' else -1
        self.half_moves, self.moves = info[4], info[5]
        self.children = []
        self.value = 0
        self.check = False
        self.calc_check_children()

    def get_key(self, move):
        return generate_key(self.board, move, self.turn, self.castle, self.half_moves, self.moves)

    def get_straight_moves(self, pos, reach):
        moves = []

        for i in range(1, reach + 1):
            new_pos = (pos[0] + i, pos[1])
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in self.own_pieces:
                    break
                elif self.board[new_pos[0]][new_pos[1]] == '.':
                    moves.append(new_pos)
                else:
                    moves.append(new_pos)
                    break
            else:
                break

        for i in range(1, reach + 1):
            new_pos = (pos[0], pos[1] + i)
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in self.own_pieces:
                    break
                elif self.board[new_pos[0]][new_pos[1]] == '.':
                    moves.append(new_pos)
                else:
                    moves.append(new_pos)
                    break
            else:
                break

        for i in range(1, reach + 1):
            new_pos = (pos[0] - i, pos[1])
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in self.own_pieces:
                    break
                elif self.board[new_pos[0]][new_pos[1]] == '.':
                    moves.append(new_pos)
                else:
                    moves.append(new_pos)
                    break
            else:
                break

        for i in range(1, reach + 1):
            new_pos = (pos[0], pos[1] - i)
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in self.own_pieces:
                    break
                elif self.board[new_pos[0]][new_pos[1]] == '.':
                    moves.append(new_pos)
                else:
                    moves.append(new_pos)
                    break
            else:
                break
        return moves

    def get_diag_moves(self, pos, reach):
        moves = []

        for i in range(1, reach + 1):
            new_pos = (pos[0] + i, pos[1] + i)
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in self.own_pieces:
                    break
                elif self.board[new_pos[0]][new_pos[1]] == '.':
                    moves.append(new_pos)
                else:
                    moves.append(new_pos)
                    break
            else:
                break

        for i in range(1, reach + 1):
            new_pos = (pos[0] + i, pos[1] - i)
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in self.own_pieces:
                    break
                elif self.board[new_pos[0]][new_pos[1]] == '.':
                    moves.append(new_pos)
                else:
                    moves.append(new_pos)
                    break
            else:
                break

        for i in range(1, reach + 1):
            new_pos = (pos[0] - i, pos[1] + i)
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in self.own_pieces:
                    break
                elif self.board[new_pos[0]][new_pos[1]] == '.':
                    moves.append(new_pos)
                else:
                    moves.append(new_pos)
                    break
            else:
                break

        for i in range(1, reach + 1):
            new_pos = (pos[0] - i, pos[1] - i)
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in self.own_pieces:
                    break
                elif self.board[new_pos[0]][new_pos[1]] == '.':
                    moves.append(new_pos)
                else:
                    moves.append(new_pos)
                    break
            else:
                break
        return moves

    def get_pawn_moves(self, pos):
        moves = []

        for i in [-1, 1]:
            new_pos = (pos[0] - 1 * self.turn_int, pos[1] + i)
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in self.opp_pieces:
                    moves.append(new_pos)
        new_pos = (pos[0] - 1 * self.turn_int, pos[1])
        if is_in_bounds(new_pos):
            if self.board[new_pos[0]][new_pos[1]] == '.':
                moves.append(new_pos)
                if self.turn == 'w' and pos[0] == 6:
                    new_pos = (4, pos[1])
                    if self.board[new_pos[0]][new_pos[1]] == '.':
                        moves.append(new_pos)
                        # TODO add en passant
                elif self.turn == 'b' and pos[0] == 1:
                    new_pos = (3, pos[1])
                    if self.board[new_pos[0]][new_pos[1]] == '.':
                        moves.append(new_pos)
                        # TODO add en passant
        return moves

    def get_knight_moves(self, pos):
        moves = []

        for i in [-2, 2]:
            for c in [-1, 1]:
                new_pos = (pos[0] + i, pos[1] + c)
                if is_in_bounds(new_pos):
                    if self.board[new_pos[0]][new_pos[1]] not in self.own_pieces:
                        moves.append(new_pos)
                new_pos = (pos[0] + c, pos[1] + i)
                if is_in_bounds(new_pos):
                    if self.board[new_pos[0]][new_pos[1]] not in self.own_pieces:
                        moves.append(new_pos)
        return moves

    def get_moves(self, piece, pos):
        moves = []
        if piece == 'r' or piece == 'R':
            moves = self.get_straight_moves(pos, 7)
        elif piece == 'n' or piece == 'N':
            moves = self.get_knight_moves(pos)
        elif piece == 'b' or piece == 'B':
            moves = self.get_diag_moves(pos, 7)
        elif piece == 'q' or piece == 'Q':
            moves = self.get_straight_moves(pos, 7) + self.get_diag_moves(pos, 7)
        elif piece == 'k' or piece == 'K':
            moves = self.get_straight_moves(pos, 1) + self.get_diag_moves(pos, 1)
        elif piece == 'p' or piece == 'P':
            moves = self.get_pawn_moves(pos)
        return moves

    def calc_check_children(self):
        idxs = np.where(self.board != '.')
        for r, c in zip(idxs[0], idxs[1]):
            # TODO add castle and en passant
            if self.board[r][c] in self.own_pieces:
                poss_moves = self.get_moves(self.board[r][c], (r, c))
                king = 'k' if self.turn == 'w' else 'K'
                for move in poss_moves:
                    if self.board[move[0]][move[1]] == king:
                        self.check = True
                    self.children.append(self.get_key(((r, c), move)))


class AI():

    def __init__(self, key='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'):
        self.state = State(key)
        if key.split(' ')[1] == 'w':
            self.model = keras.models.load_model('black_model')
        else:
            self.model = keras.models.load_model('white_model')

    def register_move(self, move):
        # TODO check that opponent doesn't check himself
        new_key = self.state.get_key(move)
        if new_key not in self.state.children:
            return 'Invalid move!'
        self.state = State(new_key)
        return 'Success'

    def register_key(self, key):
        if key not in self.state.children:
            return 'Invalid move!'
        self.state = State(key)
        return 'Success'

    def make_move(self):
        scores = []
        for child in self.state.children:
            tensor = generate_tensor(child.split(' ')[0])
            scores.append(self.model.predict(np.array([tensor, ]))[0][0])
        print(scores)
        try:
            found_move = False
            for i in (-np.array(scores)).argsort():
                state = State(self.state.children[i])
                if not state.check:
                    found_move = True
                    self.state = state
                    break
            if not found_move:
                return 'White wins!' if self.state.turn == 'b' else 'Black wins!'
        except:
            return 'White wins!' if self.state.turn == 'b' else 'Black wins!'
        if int(self.state.half_moves) >= 40: #TODO change to 100
            return 'Tie by fifty-move rule!'
        return 'Success'
