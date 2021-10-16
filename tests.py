from chess import State
import numpy as np
'1kr1qb1r/2p4p/2n2p1n/1p1Ppbp1/2B5/2N3P1/1PPPNP1P/R1BKQR2 w - - 0 0'
'4kb2/2pnq3/Pr2pR2/N5p1/1P1P4/N3P3/P7/5R1K w - - 0 31'
'3k4/q2n4/8/N5p1/3pR3/4P3/P7/7K w - - 0 39'
'3k4/3N4/4Q3/4P3/5n2/6K1/8/8 w - - 2 69'
'k7/8/8/8/8/2B4r/1PPP3P/6K1 w - - 0 0'
'r1q1kbQ1/pppbp1pr/2n4p/8/3P2p1/N1P1P3/PP3P1P/R1B1KBNR w KQ - 0 13'
'r1kR1b1r/1p5p/p5P1/1BP1p2Q/5B2/7P/PPP2PP1/2K4R w - - 0 18'
'r2q3r/ppkn3p/6P1/1BbRp2Q/8/4B2P/PPP2PP1/2K4R w - - 0 21'
'r2q3r/ppkn2bp/6P1/1BPRp2Q/8/4B2P/PPP2PP1/2K4R w - - 0 21'
'r2q1b1r/pp1n3p/2k2nP1/2P1p2Q/8/2N1B2P/PPP2PP1/2KR1B1R w - - 0 18'

def make_move(self):
    self.dict['current_board'] = self.state.board
    self.dict['key'] = self.state.get_key()

    if self.game_stage == 'Early':
        if self.state.moves > 7:
            self.game_stage = 'Mid'
    elif self.game_stage == 'Mid':
        if len(np.where(self.state.board != '.')[0]) < 16:
            self.game_stage = 'Late'

    if self.game_stage == 'Early':  # Look at many moves, but shallow depth
        n_moves = 10
        max_depth = 5
    elif self.game_stage == 'Mid':  # Look at fewer moves, but deeper
        n_moves = 5
        max_depth = 10
    else:  # Late game, look at even fewer moves and even deeper?
        n_moves = 3
        max_depth = 5

    values = []
    moves = []
    for move, child in self.state.children.items():
        if child.children is None:
            child.calc_children_and_check()
        if not child.check:
            worst_value = np.inf
            best_cmove = None
            for cmove, grandchild in child.children.items():
                if grandchild.value < worst_value:
                    if grandchild.children is None:
                        grandchild.calc_children_and_check()
                    if not grandchild.check:
                        worst_value = grandchild.value
                        best_cmove = cmove
            values.append(worst_value)
            moves.append((move, best_cmove))

    try:
        moves = np.array(moves)[np.array(values).argsort()[-n_moves:]]
        self.dict['considered_moves'] = moves
        values = []
        for move, cmove in moves:
            end_time = time.time() + self.move_time / n_moves
            child = self.state.children[move]
            grandchild = child.children[cmove]
            while time.time() < end_time:
                grandchild.rollout(self.exploration, 1, max_depth, self.weights)
            child.visits += 1
            child.value -= grandchild.value / grandchild.visits
            values.append(child.value / child.visits)
        i = np.array(values).argmin()
        self.dict['chosen_move'] = moves[i][0]
        self.state = self.state.children[moves[i][0]]
    except:
        return ('White wins!', None) if self.state.turn == 'b' else ('Black wins!', None)

    if self.state.half_moves >= 20:  # Should be 100
        return 'Tie by fifty-move rule!', None
    return 'Success', self.dict['chosen_move']
