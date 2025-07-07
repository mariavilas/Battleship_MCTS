"""
app.py
=============
A Flask web application for a Battleship game with MCTS and ML-MCTS AI bots.
"""
import os
from flask import Flask, request, jsonify, render_template
import time
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

from tfg.game.board import Board, SHIPS
from tfg.algorithms.mcts import MCTS
from tfg.ai.mcts_ml import NeuralMCTS

app = Flask(__name__, instance_relative_config=True)

# ─── Database configuration ────────────────────────────────────────────────────
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///game.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class GameResult(db.Model):
    """GameResult represents a record of a completed game in the database.

    Args:
        db.Model: Inherits from SQLAlchemy's Model class to define a database model.
    """
    __tablename__ = 'game_results'
    id  = db.Column(db.Integer,   primary_key=True)
    timestamp = db.Column(db.DateTime,  default=datetime.utcnow)
    game_mode = db.Column(db.String(50), nullable=False)
    winner = db.Column(db.String(50), nullable=False)
    duration = db.Column(db.Float,     nullable=False)

with app.app_context():
    db.create_all()

# ─── AI bots ────────────────────────────────────────────────────────────────────
vanilla_mcts = MCTS(iterations=5)
ml_mcts      = NeuralMCTS('model.pth', iters=100, c_puct=1.0)

# ─── Global game state ─────────────────────────────────────────────────────────
user_board = None
pc_board = None
user_turn = True
game_over = False
message = ''

game_mode = "uservsmcts"
boat_placement = "random"
current_turn = "player1"

manual_phase = False
placement_index = 0

# track when a game starts so we can compute its duration
start_time = None

# ─── Helpers ────────────────────────────────────────────────────────────────────
def init_game():
    """Initialize the game state for a new match.
    """
    global user_board, pc_board, user_turn, game_over, message
    global current_turn, manual_phase, placement_index, start_time

    user_board = Board()
    pc_board   = Board()

    if boat_placement == "random":
        user_board.place_fleet()
        pc_board.place_fleet()
        manual_phase = False
    else:
        manual_phase    = True
        placement_index = 0
        pc_board.place_fleet()

    if game_mode == "mcts_vs_ml_mcts":
        user_turn    = False
        current_turn = "player1"
        message      = 'MCTS vs ML-MCTS in progress.'
    else:
        user_turn = True
        if manual_phase:
            message = f'Place your boat of length {SHIPS[placement_index]}.'
        else:
            message = 'Game started. Your move!'

    game_over  = False
    start_time = time.time()

def mask_board(board):
    """Mask the board for display, hiding unshot cells unless game is over or in MCTS vs ML-MCTS mode.

    Args:
        board (list): The game board represented as a 2D list.

    Returns:
        list: A masked version of the board where unshot cells are replaced with '?'.
    """
    masked = []
    for row in board:
        new = []
        for c in row:
            if game_mode == "mcts_vs_ml_mcts" or game_over:
                new.append(c)
            else:
                new.append(c if c in ['X','O'] else '?')
        masked.append(new)
    return masked

def record_result(winner_label):
    """Record the result of a game in the database.

    Args:
        winner_label (str): The label of the winning player ('user' or 'MCTS').
    """
    duration = time.time() - start_time
    gr = GameResult(
        game_mode = game_mode,
        winner    = winner_label,
        duration  = duration
    )
    db.session.add(gr)
    db.session.commit()

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Render the main game page.

    Returns:
        str: Rendered HTML template for the game index page.
    """
    return render_template('index.html')

@app.route('/set_options', methods=['POST'])
def set_options():
    """Set game options based on user input.

    Returns:
        json: JSON response containing the initial game state.
    """
    global game_mode, boat_placement
    data           = request.get_json()
    game_mode      = data.get("game_mode", "uservsmcts")
    boat_placement = data.get("boat_placement", "random")

    init_game()
    vanilla_mcts.root = None  # Reset MCTS tree
    
    return jsonify({
        'user_board':   user_board.board,
        'pc_board':     mask_board(pc_board.board),
        'manual_phase': manual_phase,
        'user_turn':    user_turn,
        'game_over':    game_over,
        'message':      message,
        'user_boats':   user_board.get_boats_status(),
        'pc_boats':     pc_board.get_boats_status()
    })

@app.route('/start', methods=['POST'])
def start():
    """Start a new game with the current settings.

    Returns:
        json: JSON response containing the initial game state.
    """
    init_game()
    vanilla_mcts.root = None  # Reset MCTS tree
    return jsonify({
        'user_board':   user_board.board,
        'pc_board':     mask_board(pc_board.board),
        'manual_phase': manual_phase,
        'user_turn':    user_turn,
        'game_over':    game_over,
        'message':      message,
        'user_boats':   user_board.get_boats_status(),
        'pc_boats':     pc_board.get_boats_status()
    })

@app.route('/state', methods=['GET'])
def state():
    """Get the current game state.

    Returns:
        json: JSON response containing the current game state.
    """
    return jsonify({
        'user_board':   user_board.board,
        'pc_board':     mask_board(pc_board.board),
        'manual_phase': manual_phase,
        'user_turn':    user_turn,
        'game_over':    game_over,
        'message':      message,
        'user_boats':   user_board.get_boats_status(),
        'pc_boats':     pc_board.get_boats_status()
    })

@app.route('/manual_place', methods=['POST'])
def manual_place():
    """Place a ship manually during the manual placement phase.

    Returns:
        json: JSON response containing the updated game state after placing a ship.
    """
    global manual_phase, placement_index, user_board, message, user_turn, game_over

    if not manual_phase:
        return jsonify({'message': 'Manual placement phase is over.'})

    data  = request.get_json()
    start = data.get("start", {})
    end   = data.get("end", {})

    sx, sy = int(start.get("x",-1)), int(start.get("y",-1))
    ex, ey = int(end.get("x",-1)),   int(end.get("y",-1))
    if sx != ex and sy != ey:
        return jsonify({'message': 'Cells must be in a straight line.'})

    if sx == ex:
        if sy > ey: sy,ey = ey,sy
        cells = [(sx, j) for j in range(sy, ey+1)]
        orient = 'H'
    else:
        if sx > ex: sx,ex = ex,sx
        cells = [(i, sy) for i in range(sx, ex+1)]
        orient = 'V'

    req_len = SHIPS[placement_index]
    if len(cells) != req_len:
        return jsonify({'message': f'Boat length must be {req_len}.'})
    if not user_board.can_place_ship(cells[0][0], cells[0][1], orient, req_len):
        return jsonify({'message': 'Invalid placement. Try again.'})

    user_board.place_ship(cells[0][0], cells[0][1], orient, req_len)
    user_board.boats.append({"value":str(req_len), "positions": cells})

    placement_index += 1
    if placement_index >= len(SHIPS):
        manual_phase = False
        user_turn    = True
        message      = 'All boats placed. Your move!'
    else:
        message = f'Boat placed. Next: length {SHIPS[placement_index]}.'

    return jsonify({
        'user_board':   user_board.board,
        'message':      message,
        'manual_phase': manual_phase,
        'placement_index': placement_index,
        'current_boat': SHIPS[placement_index] if placement_index < len(SHIPS) else None,
        'user_boats':   user_board.get_boats_status()
    })

@app.route('/user_move', methods=['POST'])
def user_move():
    """Handle a user's move in the game.

    Returns:
        json: JSON response containing the updated game state after the user's move.
    """
    global user_turn, game_over, message

    data = request.get_json()
    x, y = int(data['x']), int(data['y'])

    # Player fires
    hit = pc_board.shoot(x, y)
    if hit:
        # If that shot sinks the last PC ship, end immediately
        if pc_board.has_won():
            game_over = True
            message   = 'You win!'
            record_result('user')
            return jsonify({
                'user_board':   user_board.board,
                'pc_board':     mask_board(pc_board.board),
                'user_turn':    user_turn,
                'game_over':    game_over,
                'message':      message,
                'manual_phase': manual_phase,
                'user_boats':   user_board.get_boats_status(),
                'pc_boats':     pc_board.get_boats_status()
            })

        # Otherwise user hit but game continues, they shoot again
        user_turn = True
        message   = 'You hit! Shoot again.'
        return jsonify({
            'user_board':   user_board.board,
            'pc_board':     mask_board(pc_board.board),
            'user_turn':    user_turn,
            'game_over':    game_over,
            'message':      message,
            'manual_phase': manual_phase,
            'user_boats':   user_board.get_boats_status(),
            'pc_boats':     pc_board.get_boats_status()
        })

    # User missed so hand turn to AI
    user_turn = False
    message   = 'You missed. PC turn.'

    # Check for user win (in case last shot was a miss but still sank the final ship—unlikely)
    if pc_board.has_won():
        game_over = True
        message   = 'You win!'
        record_result('user')
        return jsonify({
            'user_board':   user_board.board,
            'pc_board':     mask_board(pc_board.board),
            'user_turn':    user_turn,
            'game_over':    game_over,
            'message':      message,
            'manual_phase': manual_phase,
            'user_boats':   user_board.get_boats_status(),
            'pc_boats':     pc_board.get_boats_status()
        })

    # AI moves (only after a miss) — User vs MCTS
    if game_mode == 'uservsmcts':
        while True:
            # get AI move
            mx, my = vanilla_mcts.run(user_board)

            # snapshot tree & summary
            full_tree = vanilla_mcts.root.to_dict()
            summary = [
                {
                    'action':   {'x': c['action'][0], 'y': c['action'][1]},
                    'visits':   c['visits'],
                    'wins':     c['wins'],
                    'win_rate': round(c['win_rate'], 3)
                }
                for c in full_tree['children']
            ]

            # apply AI shot
            hit2 = user_board.shoot(mx, my)

            # advance root (optional)
            vanilla_mcts.update_with_move((mx, my))

            # check for AI win
            if hit2 and user_board.has_won():
                game_over = True
                message   = 'PC wins!'
                record_result('MCTS')
                break

            # on miss, hand back to user
            if not hit2:
                user_turn = True
                message   = 'PC missed. Your turn.'
                break

            # on hit, loop and fire again
            message = 'PC hit!'

        # return everything, including tree & summary
        return jsonify({
            'user_board':   user_board.board,
            'pc_board':     mask_board(pc_board.board),
            'user_turn':    user_turn,
            'game_over':    game_over,
            'message':      message,
            'manual_phase': manual_phase,
            'user_boats':   user_board.get_boats_status(),
            'pc_boats':     pc_board.get_boats_status(),
            'tree':         full_tree,
            'summary':      summary
        })

    # Fallback response
    return jsonify({
        'user_board':   user_board.board,
        'pc_board':     mask_board(pc_board.board),
        'user_turn':    user_turn,
        'game_over':    game_over,
        'message':      message,
        'manual_phase': manual_phase,
        'user_boats':   user_board.get_boats_status(),
        'pc_boats':     pc_board.get_boats_status()
    })


    
    
@app.route('/auto_move', methods=['POST'])
def auto_move():
    """Handle an automatic move by the AI bot (MCTS or ML-MCTS).

    Returns:
        json: JSON response containing the updated game state after the AI's move.
    """
    global current_turn, game_over, message, user_board, pc_board

    if user_board is None or pc_board is None:
        init_game()
        vanilla_mcts.root = None

    if current_turn == "player1":
        mover, target, label = vanilla_mcts, pc_board,   "MCTS"
    else:
        mover, target, label = ml_mcts,      user_board, "ML-MCTS"

    while True:
        move = mover.run(target)
        if move is None:
            game_over = True
            message   = f"No moves for {label}."
            record_result(label)
            break

        mx, my = move
        hit    = target.shoot(mx, my)

        if mover is vanilla_mcts:
            vanilla_mcts.update_with_move((mx, my))

        if hit and target.has_won():
            game_over = True
            message   = f"{label} wins!"
            record_result(label)
            break

        if not hit:
            message = f"{label} missed."
            current_turn = 'player2' if current_turn == 'player1' else 'player1'
            break

        message = f"{label} hit!"

    return jsonify({
        'user_board':   mask_board(user_board.board),
        'pc_board':     mask_board(pc_board.board),
        'game_over':    game_over,
        'message':      message,
        'user_boats':   user_board.get_boats_status(),
        'pc_boats':     pc_board.get_boats_status(),
        'current_turn': current_turn
    })



@app.route('/stats')
def stats():
    """Get game statistics from the database.

    Returns:
        json: JSON response containing game statistics.
    """
    rows = GameResult.query.order_by(GameResult.timestamp).all()
    out = {'uservsmcts': [], 'mcts_vs_ml_mcts': []}
    for r in rows:
        out[r.game_mode].append({
            'winner':   r.winner,
            'duration': r.duration
        })
    return jsonify(out)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # 5000 is the default port if not specified
    app.run(host='0.0.0.0', port=port)
