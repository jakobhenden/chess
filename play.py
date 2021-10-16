from chess import AI
import PySimpleGUI as sg
import os

blank = os.path.join('textures/blank.png')
bbishop = os.path.join('textures/bbishop.png')
wbishop = os.path.join('textures/wbishop.png')
bpawn = os.path.join('textures/bpawn.png')
wpawn = os.path.join('textures/wpawn.png')
bknight = os.path.join('textures/bknight.png')
wknight = os.path.join('textures/wknight.png')
brook = os.path.join('textures/brook.png')
wrook = os.path.join('textures/wrook.png')
bqueen = os.path.join('textures/bqueen.png')
wqueen = os.path.join('textures/wqueen.png')
bking = os.path.join('textures/bking.png')
wking = os.path.join('textures/wking.png')

images = {'b': bbishop, 'B': wbishop, 'p': bpawn, 'P': wpawn, 'n': bknight, 'N': wknight,
          'r': brook, 'R': wrook, 'k': bking, 'K': wking, 'q': bqueen, 'Q': wqueen, '.': blank}


def render_square(image, location):
    if (location[0] + location[1]) % 2:
        color = '#B58863'
    else:
        color = '#F0D9B5'
    return sg.RButton('', image_filename=image, size=(1, 1), button_color=('white', color), pad=(0, 0), key=location)


def redraw_board(window, board):
    for r in range(8):
        for c in range(8):
            color = '#B58863' if (r + c) % 2 else '#F0D9B5'
            piece_image = images[board[r][c]]
            elem = window.FindElement(key=(r, c))
            elem.Update(button_color=('white', color), image_filename=piece_image)


def play_as_white(debug=False, move_time=5):

    ai = AI(move_time=move_time)
    board = ai.state.board

    board_layout = []
    for r in range(8):
        row = []
        for c in range(8):
            piece_image = images[board[r][c]]
            row.append(render_square(piece_image, location=(r, c)))
        board_layout.append(row)

    if debug:
        layout = [[sg.Column(board_layout), sg.Column([[sg.RButton('Why did you\n do that?', key='print')]])]]
    else:
        layout = board_layout

    window = sg.Window('Chess').Layout(layout)

    move = (None, None)
    while True:
        button, value = window.Read()
        if button == 'print':
            ai.print_decision_process()
        elif button == sg.WINDOW_CLOSED:
            break
        else:
            if not move[0]:
                move = (button, None)
            elif not move[1]:
                move = (move[0], button)
                message = ai.register_move(move)
                if message == 'Success':
                    board = ai.state.board
                    redraw_board(window, board)
                    window.refresh()
                    message, move = ai.make_move()
                    if message == 'Success':
                        board = ai.state.board
                        redraw_board(window, board)
                        window.refresh()
                    elif message == 'Check mate!':
                        board = ai.state.board
                        redraw_board(window, board)
                        window.refresh()
                        sg.Popup(message, keep_on_top=True, font=('Arial', 15, 'bold'))
                        break
                    else:
                        sg.Popup(message, keep_on_top=True, font=('Arial', 15, 'bold'))
                        break
                else:
                    sg.Popup(message, keep_on_top=True, font=('Arial', 15, 'bold'))
                move = (None, None)


def play(debug=False, move_time=5):

    ai_white = AI(move_time=move_time)
    board = ai_white.state.board

    board_layout = []
    for r in range(8):
        row = []
        for c in range(8):
            piece_image = images[board[r][c]]
            row.append(render_square(piece_image, location=(r, c)))
        board_layout.append(row)

    if debug:
        layout = [[sg.Column(board_layout), sg.Column([[sg.RButton('Print key', key='print')]])]]
    else:
        layout = board_layout

    window = sg.Window('Chess').Layout(layout)
    window.read()

    ai_white.make_move()
    board = ai_white.state.board
    redraw_board(window, board)
    window.refresh()
    ai_black = AI(key=ai_white.state.get_key(), move_time=move_time)

    while True:
        button, value = window.Read(timeout=10)
        if button == 'print':
            print(ai_black.state.get_key())
        elif button == sg.WINDOW_CLOSED:
            break
        message, move = ai_black.make_move()
        if message == 'Success':
            board = ai_black.state.board
            message = ai_white.register_move(move)
            if message == 'Success':
                redraw_board(window, board)
                window.Refresh()
            else:
                print('Black tried to cheat', ai_white.state.get_key(), ai_black.state.get_key())
                break
        else:
            sg.Popup(message, keep_on_top=True, font=('Arial', 15, 'bold'))
            break

        message, move = ai_white.make_move()
        if message == 'Success':
            board = ai_white.state.board
            message = ai_black.register_move(move)
            if message == 'Success':
                redraw_board(window, board)
                window.Refresh()
            else:
                print('White tried to cheat', ai_black.state.get_key(), ai_white.state.get_key())
                break
        else:
            sg.Popup(message, keep_on_top=True, font=('Arial', 15, 'bold'))
            break


play_as_white(move_time=5)
