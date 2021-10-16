from math import sqrt, log
import numpy as np
import random
import time

piece_values = {'r': -5, 'n': -3, 'b': -3, 'q': -9, 'k': -9, 'p': -1,
                'R': 5, 'N': 3, 'B': 3, 'Q': 9, 'K': 9, 'P': 1, '.': 0}

pos_values = {(r, c): 1 for r in range(0, 8) for c in range(0, 8)}
for r in range(2, 6):
    for c in range(2, 6):
        if 2 < r < 5 or 2 < c < 5:
            pos_values[(r, c)] = 3
        else:
            pos_values[(r, c)] = 2


def is_in_bounds(pos):
    return -1 < pos[0] < 8 and -1 < pos[1] < 8


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


def generate_key(board, turn, castle, en_passant, half_moves, moves):
    key = ''
    for row in board:
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
    return key[:-1] + ' ' + turn + ' ' + castle + ' ' + en_passant + ' ' + str(half_moves) + ' ' + str(moves)


def generate_state(board, turn, castle, en_passant, half_moves, moves, move=None, which_castle=None):
    if move is not None:
        capture = board[move[1][0]][move[1][1]] != '.'
        piece = board[move[0][0]][move[0][1]]
        pawn_move = piece == 'P' or piece == 'p'

        new_board = board.copy()
        new_board[move[0][0]][move[0][1]] = '.'
        new_board[move[1][0]][move[1][1]] = piece

        double_step = False
        if pawn_move:
            # Upgrade pawns
            if move[1][0] == 0 or move[1][0] == 7:
                new_board[move[1][0]][move[1][1]] = 'Q' if piece == 'P' else 'q'
            # En passant
            if en_passant != '-':
                r = int(en_passant[0])
                c = int(en_passant[1])
                if move[1][0] == r and move[1][1] == c:
                    capture = True
                    backward = 1 if piece == 'P' else -1
                    new_board[r + backward][c] = '.'
            if abs(move[0][0] - move[1][0]) == 2:
                double_step = True
                en_passant = '5{}'.format(move[0][1]) if piece == 'P' else '2{}'.format(move[0][1])

        if not double_step:
            en_passant = '-'

        if capture or pawn_move:
            half_moves = 0
        else:
            half_moves += 1

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
        if not castle:
            castle = '-'

    else:
        half_moves += + 1

        new_board = board.copy()

        # Castling
        if which_castle == 'K':
            new_board[7][4], new_board[7][5], new_board[7][6], new_board[7][7] = '.', 'R', 'K', '.'
            castle = castle.replace('K', '').replace('Q', '')
        elif which_castle == 'Q':
            new_board[7][0], new_board[7][2], new_board[7][3], new_board[7][4] = '.', 'K', 'R', '.'
            castle = castle.replace('K', '').replace('Q', '')
        elif which_castle == 'k':
            new_board[0][4], new_board[0][5], new_board[0][6], new_board[0][7] = '.', 'r', 'k', '.'
            castle = castle.replace('k', '').replace('q', '')
        elif which_castle == 'q':
            new_board[0][0], new_board[0][2], new_board[0][3], new_board[0][4] = '.', 'k', 'r', '.'
            castle = castle.replace('k', '').replace('q', '')

        if not castle:
            castle = '-'

        en_passant = '-'

    # Change turns
    if turn == 'b':
        turn = 'w'
        moves = int(moves) + 1
    else:
        turn = 'b'

    return State(board=new_board, turn=turn, castle=castle, en_passant=en_passant, half_moves=half_moves, moves=moves)


def move_to_str(move):
    return str(move[0][0]) + str(move[0][1]) + str(move[1][0]) + str(move[1][1])


class State:
    def __init__(self, key='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
                 board=None, turn=None, castle=None, en_passant=None, half_moves=None, moves=None):
        if board is None:
            self.key = key
            info = key.split(' ')
            self.board = generate_board(info[0])
            self.turn, self.castle, self.en_passant = info[1], info[2], info[3]
            self.half_moves, self.moves = int(info[4]), int(info[5])
        else:
            self.key = None
            self.board = board
            self.turn, self.castle, self.en_passant = turn, castle, en_passant
            self.half_moves, self.moves = half_moves, moves

        self.own_pieces = 'RNBQKP' if self.turn == 'w' else 'rnbqkp'
        self.opp_pieces = 'rnbqkp' if self.turn == 'w' else 'RNBQKP'
        self.turn_int = 1 if self.turn == 'w' else -1

        self.value = self.calc_simple_value()
        self.visits = 1
        self.children = None
        self.check = None
    
    def get_key(self):
        if self.key is None:
            self.key = generate_key(self.board, self.turn, self.castle, self.en_passant, self.half_moves, self.moves)
        return self.key

    def get_child_state(self, move=None, castle=None):
        if move is not None:
            return generate_state(self.board, self.turn, self.castle, self.en_passant, self.half_moves, self.moves,
                                  move=move)
        else:
            return generate_state(self.board, self.turn, self.castle, self.en_passant, self.half_moves, self.moves,
                                  which_castle=castle)

    def get_straight_moves(self, pos, reach, own):
        moves = []

        for i in range(1, reach + 1):
            new_pos = (pos[0] + i, pos[1])
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in (self.own_pieces if own else self.opp_pieces):
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
                if self.board[new_pos[0]][new_pos[1]] in (self.own_pieces if own else self.opp_pieces):
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
                if self.board[new_pos[0]][new_pos[1]] in (self.own_pieces if own else self.opp_pieces):
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
                if self.board[new_pos[0]][new_pos[1]] in (self.own_pieces if own else self.opp_pieces):
                    break
                elif self.board[new_pos[0]][new_pos[1]] == '.':
                    moves.append(new_pos)
                else:
                    moves.append(new_pos)
                    break
            else:
                break
        return moves

    def get_diag_moves(self, pos, reach, own):
        moves = []

        for i in range(1, reach + 1):
            new_pos = (pos[0] + i, pos[1] + i)
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in (self.own_pieces if own else self.opp_pieces):
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
                if self.board[new_pos[0]][new_pos[1]] in (self.own_pieces if own else self.opp_pieces):
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
                if self.board[new_pos[0]][new_pos[1]] in (self.own_pieces if own else self.opp_pieces):
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
                if self.board[new_pos[0]][new_pos[1]] in (self.own_pieces if own else self.opp_pieces):
                    break
                elif self.board[new_pos[0]][new_pos[1]] == '.':
                    moves.append(new_pos)
                else:
                    moves.append(new_pos)
                    break
            else:
                break
        return moves

    def get_pawn_moves(self, pos, own, pawn):
        moves = []
        forward = -1 if pawn == 'P' else 1

        for i in [-1, 1]:
            new_pos = (pos[0] + forward, pos[1] + i)
            if is_in_bounds(new_pos):
                if self.board[new_pos[0]][new_pos[1]] in (self.own_pieces if not own else self.opp_pieces):
                    moves.append(new_pos)
        new_pos = (pos[0] + forward, pos[1])
        if is_in_bounds(new_pos):
            if self.board[new_pos[0]][new_pos[1]] == '.':
                moves.append(new_pos)
                if pawn == 'P' and pos[0] == 6:
                    new_pos = (4, pos[1])
                    if self.board[new_pos[0]][new_pos[1]] == '.':
                        moves.append(new_pos)
                elif pawn == 'p' and pos[0] == 1:
                    new_pos = (3, pos[1])
                    if self.board[new_pos[0]][new_pos[1]] == '.':
                        moves.append(new_pos)
        if self.en_passant != '-':
            r = int(self.en_passant[0])
            c = int(self.en_passant[1])
            if pos[0] == r - forward and abs(pos[1] - c) == 1:
                moves.append((r, c))

        return moves

    def get_knight_moves(self, pos, own):
        moves = []

        for i in [-2, 2]:
            for c in [-1, 1]:
                new_pos = (pos[0] + i, pos[1] + c)
                if is_in_bounds(new_pos):
                    if self.board[new_pos[0]][new_pos[1]] not in (self.own_pieces if own else self.opp_pieces):
                        moves.append(new_pos)
                new_pos = (pos[0] + c, pos[1] + i)
                if is_in_bounds(new_pos):
                    if self.board[new_pos[0]][new_pos[1]] not in (self.own_pieces if own else self.opp_pieces):
                        moves.append(new_pos)
        return moves

    def get_moves(self, piece, pos, own):
        moves = []
        if piece == 'r' or piece == 'R':
            moves = self.get_straight_moves(pos, 7, own)
        elif piece == 'n' or piece == 'N':
            moves = self.get_knight_moves(pos, own)
        elif piece == 'b' or piece == 'B':
            moves = self.get_diag_moves(pos, 7, own)
        elif piece == 'q' or piece == 'Q':
            moves = self.get_straight_moves(pos, 7, own) + self.get_diag_moves(pos, 7, own)
        elif piece == 'k' or piece == 'K':
            moves = self.get_straight_moves(pos, 1, own) + self.get_diag_moves(pos, 1, own)
        elif piece == 'p' or piece == 'P':
            moves = self.get_pawn_moves(pos, own, piece)
        return moves
    
    def calc_simple_value(self):
        idxs = np.where(self.board != '.')
        value = 0
        for r, c in zip(idxs[0], idxs[1]):
            piece = self.board[r][c]
            value += piece_values[piece] * self.turn_int
        return value

    def calc_children_and_check(self):
        idxs = np.where(self.board != '.')
        check_squares = []
        opp_king = 'k' if self.turn == 'w' else 'K'
        self.children = {}

        for r, c in zip(idxs[0], idxs[1]):
            piece = self.board[r][c]

            if piece in self.own_pieces:
                poss_moves = self.get_moves(piece, (r, c), True)
                for move in poss_moves:
                    if self.board[move[0]][move[1]] == opp_king:
                        self.check = True
                    self.children[move_to_str(((r, c), move))] = self.get_child_state(move=((r, c), move))
            else:
                poss_moves = self.get_moves(piece, (r, c), False)
                for move in poss_moves:
                    check_squares.append(move)

        if self.castle != '-':
            if self.turn == 'w':
                if 'K' in self.castle and self.board[7][5] == '.' and self.board[7][6] == '.':
                    if (7, 4) not in check_squares and (7, 5) not in check_squares and (7, 6) not in check_squares:
                        self.children['7476'] = self.get_child_state(castle='K')
                if 'Q' in self.castle and self.board[7][1] == '.' and self.board[7][2] == '.' and self.board[7][3] == '.':
                    if (7, 2) not in check_squares and (7, 3) not in check_squares and (7, 4) not in check_squares:
                        self.children['7472'] = self.get_child_state(castle='Q')
            else:
                if 'k' in self.castle and self.board[0][5] == '.' and self.board[0][6] == '.':
                    if (0, 4) not in check_squares and (0, 5) not in check_squares and (0, 6) not in check_squares:
                        self.children['0406'] = self.get_child_state(castle='k')
                if 'q' in self.castle and self.board[0][1] == '.' and self.board[0][2] == '.' and self.board[0][3] == '.':
                    if (0, 2) not in check_squares and (0, 3) not in check_squares and (0, 4) not in check_squares:
                        self.children['0402'] = self.get_child_state(castle='q')

    def calc_adv_value(self, weights):
        idxs = np.where(self.board != '.')

        if self.turn == 'w':
            own_pawn, opp_pawn = 'P', 'p'
            own_king, opp_king = 'K', 'k'
        else:
            own_pawn, opp_pawn = 'p', 'P'
            own_king, opp_king = 'k', 'K'

        pawn_value = 0
        material_value = 0
        attack_value = 0
        territory_value = 0
        develop_value = 0

        for r, c in zip(idxs[0], idxs[1]):
            piece = self.board[r][c]
            material_value += piece_values[piece] * self.turn_int

            if piece in self.own_pieces:

                if piece == own_pawn:
                    develop_value += (3.5 - r) * self.turn_int
                    for i, j in zip([-1, 0, 1, 0], [0, -1, 0, 1]):
                        if is_in_bounds((r + i, c + j)):
                            if self.board[r + i][c + j] == own_pawn:
                                pawn_value -= 0.25
                    for i, j in zip([-1, -1, 1, 1], [-1, 1, -1, 1]):
                        if is_in_bounds((r + i, c + j)):
                            if self.board[r + i][c + j] == own_pawn:
                                pawn_value += 0.25

                else:
                    if r % 7 != 0 and piece != own_king:
                        develop_value += 1
                    elif piece == own_king and r % 7 == 0 and (c == 2 or c == 6):
                        develop_value += 3

                poss_moves = self.get_moves(piece, (r, c), True)
                for move in poss_moves:
                    attack_value += ((-1) * piece_values[self.board[move[0]][move[1]]] * self.turn_int)
                    territory_value += pos_values[move]
            else:

                if piece == opp_pawn:
                    develop_value += (3.5 - r) * self.turn_int
                    for i, j in zip([-1, 0, 1, 0], [0, -1, 0, 1]):
                        if is_in_bounds((r + i, c + j)):
                            if self.board[r + i][c + j] == opp_pawn:
                                pawn_value += 0.25
                    for i, j in zip([-1, -1, 1, 1], [-1, 1, -1, 1]):
                        if is_in_bounds((r + i, c + j)):
                            if self.board[r + i][c + j] == opp_pawn:
                                pawn_value -= 0.25

                else:
                    if r % 7 != 0 and piece != opp_king:
                        develop_value -= 1
                    elif piece == opp_king and r % 7 == 0 and (c == 2 or c == 6):
                        develop_value -= 3

                poss_moves = self.get_moves(piece, (r, c), False)
                for move in poss_moves:
                    attack_value -= piece_values[self.board[move[0]][move[1]]] * self.turn_int
                    territory_value -= pos_values[move]

        values = np.array([pawn_value, material_value, attack_value, territory_value, develop_value])
        self.value = sum(values * weights)

    def rollout(self, e, depth, max_depth, weights):
        if depth > max_depth:
            self.visits = 1
            self.calc_adv_value(weights)
            return self.value / self.visits

        if self.children is None:
            self.calc_children_and_check()

        if random.random() < e:
            try:
                while True:
                    move, child = random.choice(list(self.children.items()))
                    if child.children is None:
                        child.calc_children_and_check()
                    if not child.check:
                        self.visits += 1
                        self.value -= child.rollout(e, depth + 1, max_depth, weights)
                        return self.value / self.visits
                    else:
                        del self.children[move]
            except:
                return -100

        values = []
        moves = []
        for move, child in self.children.items():
            values.append(child.value / child.visits - sqrt(log(self.visits) / child.visits))
            moves.append(move)
        for i in np.array(values).argsort():
            child = self.children[moves[i]]
            if child.children is None:
                child.calc_children_and_check()
            if not child.check:
                self.visits += 1
                self.value -= child.rollout(e, depth + 1, max_depth, weights)
                return self.value / self.visits
            else:
                del self.children[moves[i]]
        return -100


class AI:

    def __init__(self, key='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', move_time=5, exploration=0.3,
                 weights=None):
        self.state = State(key)
        self.state.calc_children_and_check()
        self.move_time = move_time
        self.exploration = exploration
        if weights is None:
            weights = np.array([0.5, 1, 0.1, 0.2, 1])
        self.weights = weights
        self.game_stage = 'Early'
        self.dict = {}

    def register_move(self, move):
        if type(move) is not np.str_:
            move = move_to_str(move)
        try:
            state = self.state.children[move]
        except:
            return 'Invalid move!'
        if state.children is None:
            state.calc_children_and_check()
        if state.check:
            return 'Invalid move!'
        self.state = state
        return 'Success'

    def make_move(self):
        self.dict['current_board'] = self.state.board
        self.dict['key'] = self.state.get_key()

        if self.game_stage == 'Early':
            if self.state.moves > 7:
                self.game_stage = 'Mid'
        elif self.game_stage == 'Mid':
            if len(np.where(self.state.board != '.')[0]) < 16:
                self.game_stage = 'Late'

        if self.game_stage == 'Early':  # Look at shallow depth
            max_depth = 4
        elif self.game_stage == 'Mid':  # Look deeper
            max_depth = 7
        else:  # Look even deeper
            max_depth = 10

        values = []
        moves = []
        best_value = -np.inf
        # Look at all possible moves
        for move, child in self.state.children.items():
            # Expand child if not previously expanded
            if child.children is None:
                child.calc_children_and_check()
            # If this is a valid move, proceed
            if not child.check:
                worst_value = np.inf
                best_cmoves = []
                # For every possible counter move
                for cmove, grandchild in child.children.items():
                    # If this is one of the best counter moves yet
                    if grandchild.value <= worst_value:
                        # Expand if not previously expanded
                        if grandchild.children is None:
                            grandchild.calc_children_and_check()
                        # If this is a valid counter move
                        if not grandchild.check:
                            # If this is the best counter move yet
                            if grandchild.value < worst_value:
                                # Replace all inferior counter moves and update the worst value
                                best_cmoves = [cmove]
                                worst_value = grandchild.value
                            # If it's an equal (to the best) counter move
                            else:
                                # Add the counter move to the list
                                best_cmoves.append(cmove)
                # If a lower worst value hasn't been found, the move results in check mate -> chose that move
                if worst_value == np.inf:
                    self.state = self.state.children[move]
                    return ('Check mate!', None)
                # If it's a superior move, replace all other moves
                if worst_value > best_value + 2:
                    best_value = worst_value
                    moves = [(move, best_cmoves)]
                    values = [worst_value]
                # If it's among the best moves, add the move to the list
                elif worst_value > best_value - 2:
                    moves.append((move, best_cmoves))
                    values.append(worst_value)


        try:
            self.dict['considered_moves'] = moves
            nr_moves = len(moves)
            # If there's only one feasible move, no need to roll-out -> chose that move
            if nr_moves == 1:
                print('Only one feasible move found')
                self.dict['chosen_move'] = moves[0][0]
                self.state = self.state.children[moves[0][0]]
            else:
                print('Considering {} moves'.format(nr_moves))
                values = []
                # For every feasible move
                for move, cmoves in moves:
                    nr_cmoves = len(cmoves)
                    # Select maximum 10 of the best (and equal) counter moves
                    if nr_cmoves > 10:
                        cmoves = random.sample(cmoves, 10)
                        nr_cmoves = 10
                    print('   Considering {} countermoves'.format(nr_cmoves))
                    child = self.state.children[move]
                    # For all selected counter moves
                    for cmove in cmoves:
                        # Make roll-outs as long as there's time left
                        end_time = time.time() + self.move_time / nr_moves / nr_cmoves
                        grandchild = child.children[cmove]
                        i = 0
                        while time.time() < end_time:
                            i += 1
                            grandchild.rollout(self.exploration, 1, max_depth, self.weights)
                        print('      Did {} roll-outs with {} to {}'.format(i, move, cmove))
                        child.visits += 1
                        child.value -= grandchild.value / grandchild.visits
                        #child.visits += i
                        #child.value = (child.value * child.visits + grandchild.value * i) / (child.visits + i)
                    # Add value after roll-outs to the list
                    values.append(child.value / child.visits)

                # Chose the best move
                #print(values)
                i = np.array(values).argmin()
                self.dict['chosen_move'] = moves[i][0]
                self.state = self.state.children[moves[i][0]]
        except:
            return ('White wins!', None) if self.state.turn == 'b' else ('Black wins!', None)

        if self.state.half_moves >= 20:  # Should be 100
            return 'Tie by fifty-move rule!', None
        return 'Success', self.dict['chosen_move']

    def print_decision_process(self):
        print('current board:')
        print(self.dict['current_board'])
        print(self.dict['key'])
        print('considered moves/counter moves:')
        print(self.dict['considered_moves'])
        print('chosen move:')
        print(self.dict['chosen_move'])
