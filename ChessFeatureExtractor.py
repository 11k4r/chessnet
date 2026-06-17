import chess
import os
import json

class ChessFeatureExtractor:
    def __init__(self, opening_book_path: str = None):
        """
        Initializes the extractor and loads the opening books if a path is provided.
        """
        self.opening_book = self._load_opening_books(opening_book_path) if opening_book_path else {}

    def extract_features(self, board: chess.Board, user_side: str = 'w') -> dict:
        """
        Extracts base features, opening names, and endgame types from a chess.Board object.
        """
        fen = board.fen()
        counts = self._get_material_counts(board)
        passed_dict = self._get_passed_pawns(board)
        pins_dict = self._get_pins(board)
        skewers_dict = self._get_skewers(board)
        threats_dict = self._get_check_and_mate_threats(board)
        
        return {
            "fen": fen,
            "turn": self._get_turn(board),
            "move_counters": self._get_move_counters(board),
            "castling_rights": self._get_castling_rights(board),
            "is_terminal": self._get_terminal_states(board),
            "opening_name": self._get_opening_name(fen),
            "material_counts": counts,
            "raw_scores": self._get_raw_scores(counts),
            "has_bishop_pair": self._get_bishop_pair(counts),
            "minor_piece_imbalance": self._get_minor_imbalance(counts),
            "legal_moves": self._get_legal_moves(board),
            "center_control": self._get_center_control(board),
            "extended_center_control": self._get_extended_center_control(board),
            "safe_mobility": self._get_safe_mobility(board),
            "passed_pawns": passed_dict,
            "isolated_pawns": self._get_isolated_pawns(board),
            "doubled_pawns": self._get_doubled_pawns(board),
            "backward_pawns": self._get_backward_pawns(board),
            "pawn_islands": self._get_pawn_islands(board),
            "pawn_rams": self._get_pawn_rams(board),
            "pawn_chains": self._get_pawn_chains(board),
            "pawn_levers": self._get_levers(board), #Also called pawn break
            "phalanx_pawns": self._get_phalanx_pawns(board),
            "connected_passers": self._get_connected_passers(passed_dict),
            "protected_passers": self._get_protected_passers(board, passed_dict),
            "candidate_passers": self._get_candidate_passers(board, passed_dict),
            "pawn_majorities": self._get_pawn_majorities(board),
            "in_check": board.is_check(),
            "checkers": [chess.square_name(sq) for sq in board.checkers()],
            "king_ring_attacks": self._get_king_ring_attacks(board),
            "pawn_shield_integrity": self._get_pawn_shield_integrity(board),
            "open_files_near_king": self._get_open_files_near_king(board),
            "hanging_pieces": self._get_hanging_pieces(board),
            "capture_tension": self._get_capture_tension(board),
            "absolute_pins": pins_dict["absolute"],
            "relative_pins": pins_dict["relative"],
            "discovered_attacks": self._get_discovered_attacks(board),
            "forks": self._get_forks(board),
            "batteries": self._get_batteries(board),
            "xrays": self._get_xrays(board),
            "absolute_skewers": skewers_dict["absolute"],
            "relative_skewers": skewers_dict["relative"],
            "undeveloped_minors": self._get_undeveloped_minors(board),
            "rooks_connected": self._get_rooks_connected(board),
            "outposts": self._get_outposts(board),
            "piece_connectivity": self._get_piece_connectivity(board),
            "king_activity": self._get_king_activity(board),
            "rule_of_the_square": self._get_rule_of_the_square(board, passed_dict),
            "opposition": self._get_opposition(board),
            "check_threats": threats_dict["checks"],
            "checkmate_threats": threats_dict["mates"],
            "fianchetto_structures": self._get_fianchetto_structures(board),
            "opposite_castling": self._get_opposite_castling(board),
        }

    # ==========================================
    # Base Features
    # ==========================================
    def _get_turn(self, board: chess.Board) -> str:
        return "White" if board.turn == chess.WHITE else "Black"

    def _get_move_counters(self, board: chess.Board) -> dict:
        return {
            "half_moves": board.halfmove_clock,
            "full_moves": board.fullmove_number
        }

    def _get_castling_rights(self, board: chess.Board) -> dict:
        return {
            "white_ks": board.has_kingside_castling_rights(chess.WHITE),
            "white_qs": board.has_queenside_castling_rights(chess.WHITE),
            "black_ks": board.has_kingside_castling_rights(chess.BLACK),
            "black_qs": board.has_queenside_castling_rights(chess.BLACK)
        }

    def _get_terminal_states(self, board: chess.Board) -> dict:
        return {
            "checkmate": board.is_checkmate(),
            "stalemate": board.is_stalemate(),
            "insufficient_material": board.is_insufficient_material()
        }

    # ==========================================
    # Openings Extraction (Sourced from helpers.py)
    # ==========================================
    def _load_opening_books(self, folder_name: str) -> dict:
        json_filepaths = [
            os.path.join(folder_name, "ecoA.json"),
            os.path.join(folder_name, "ecoB.json"),
            os.path.join(folder_name, "ecoC.json"),
            os.path.join(folder_name, "ecoD.json"),
            os.path.join(folder_name, "ecoE.json")
        ]
        
        opening_book = {}
        for filepath in json_filepaths:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    eco_data = json.load(f)
                
                if isinstance(eco_data, dict):
                    for fen_key, opening_data in eco_data.items():
                        parts = fen_key.split()
                        epd = " ".join(parts[:4])
                        opening_book[epd] = opening_data
                        
                        # Fallback key: Force en passant square to '-'
                        if len(parts) >= 4 and parts[3] != '-':
                            parts[3] = '-'
                            opening_book[" ".join(parts[:4])] = opening_data
                            
            except FileNotFoundError:
                print(f"Warning: Could not find {filepath}.")
                
        return opening_book

    def _get_opening_name(self, fen: str) -> str:
        if not self.opening_book or not fen:
            return "Unknown Opening"
            
        parts = fen.split()
        if len(parts) < 4:
            return "Unknown Opening"

        # Try exact match
        epd_exact = " ".join(parts[:4])
        
        # Try fallback match 
        parts_no_ep = parts[:4]
        parts_no_ep[3] = '-'
        epd_no_ep = " ".join(parts_no_ep)
        
        if epd_exact in self.opening_book:
            return self.opening_book[epd_exact].get("name", "Unknown Opening")
        elif epd_no_ep in self.opening_book:
            return self.opening_book[epd_no_ep].get("name", "Unknown Opening")
            
        return "Unknown Opening"


    # ==========================================
    # Material & Imbalance Features
    # ==========================================
    def _get_material_counts(self, board: chess.Board) -> dict:
        """Counts exact occurrences of each piece type by color."""
        return {
            "white_pawns": len(board.pieces(chess.PAWN, chess.WHITE)),
            "white_knights": len(board.pieces(chess.KNIGHT, chess.WHITE)),
            "white_bishops": len(board.pieces(chess.BISHOP, chess.WHITE)),
            "white_rooks": len(board.pieces(chess.ROOK, chess.WHITE)),
            "white_queens": len(board.pieces(chess.QUEEN, chess.WHITE)),
            "black_pawns": len(board.pieces(chess.PAWN, chess.BLACK)),
            "black_knights": len(board.pieces(chess.KNIGHT, chess.BLACK)),
            "black_bishops": len(board.pieces(chess.BISHOP, chess.BLACK)),
            "black_rooks": len(board.pieces(chess.ROOK, chess.BLACK)),
            "black_queens": len(board.pieces(chess.QUEEN, chess.BLACK)),
        }

    def _get_raw_scores(self, counts: dict) -> dict:
        """Calculates traditional material score (Q=9, R=5, B=3, N=3, P=1)."""
        w_score = (counts["white_queens"] * 9 + counts["white_rooks"] * 5 + 
                   counts["white_bishops"] * 3 + counts["white_knights"] * 3 + 
                   counts["white_pawns"] * 1)
                   
        b_score = (counts["black_queens"] * 9 + counts["black_rooks"] * 5 + 
                   counts["black_bishops"] * 3 + counts["black_knights"] * 3 + 
                   counts["black_pawns"] * 1)
                   
        return {"white": w_score, "black": b_score}

    def _get_bishop_pair(self, counts: dict) -> dict:
        """Checks if a player has 2 or more bishops."""
        return {
            "white": counts["white_bishops"] >= 2,
            "black": counts["black_bishops"] >= 2
        }

    def _get_minor_imbalance(self, counts: dict) -> dict:
        """Returns the minor pieces possessed by each side as a structured dictionary."""
        w_n = counts["white_knights"]
        w_b = counts["white_bishops"]
        b_n = counts["black_knights"]
        b_b = counts["black_bishops"]
        
        return {
            "white": ("N" * w_n) + ("B" * w_b),
            "black": ("N" * b_n) + ("B" * b_b)
        }

    # ==========================================
    # Tactical & Spatial Features
    # ==========================================

    def _get_center_control(self, board: chess.Board) -> dict:
        """Returns the exact pieces (and their squares) attacking d4, e4, d5, e5."""
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        return self._extract_attackers(board, center_squares)

    def _get_extended_center_control(self, board: chess.Board) -> dict:
        """Returns the exact pieces attacking the 16 central squares."""
        extended_squares = [
            chess.C3, chess.D3, chess.E3, chess.F3,
            chess.C4, chess.D4, chess.E4, chess.F4,
            chess.C5, chess.D5, chess.E5, chess.F5,
            chess.C6, chess.D6, chess.E6, chess.F6
        ]
        return self._extract_attackers(board, extended_squares)

    def _extract_attackers(self, board: chess.Board, target_squares: list) -> dict:
        """Helper method to map target squares to an array of exact piece symbols attacking them."""
        w_attackers = {}
        b_attackers = {}

        for sq in target_squares:
            sq_name = chess.square_name(sq)
            w_attackers[sq_name] = []
            b_attackers[sq_name] = []
            
            for attacker_sq in board.attackers(chess.WHITE, sq):
                piece = board.piece_at(attacker_sq)
                if piece:
                    w_attackers[sq_name].append(piece.symbol())
                    
            for attacker_sq in board.attackers(chess.BLACK, sq):
                piece = board.piece_at(attacker_sq)
                if piece:
                    b_attackers[sq_name].append(piece.symbol())

        return {
            "white": w_attackers,
            "black": b_attackers
        }


    def _get_legal_moves(self, board: chess.Board) -> dict:
        moves_dict = {"white": {}, "black": {}}
    
        def extract_moves(b: chess.Board):
            d = {}
            for move in b.legal_moves:
                from_sq = chess.square_name(move.from_square)
                to_sq = chess.square_name(move.to_square)
                
                if from_sq not in d:
                    d[from_sq] = []
                    
                # Prevent adding the same destination square 4 times during promotions
                if to_sq not in d[from_sq]:
                    d[from_sq].append(to_sq)
            return d
    
        current_color = "white" if board.turn == chess.WHITE else "black"
        moves_dict[current_color] = extract_moves(board)
    
        # Safely generate hypothetical opponent moves
        if not board.is_check():
            temp_board = board.copy(stack=False) 
            
            # Pushing a null move safely flips the turn, updates clocks, 
            # and clears en passant caches so the move generator doesn't crash.
            temp_board.push(chess.Move.null())
            
            other_color = "black" if current_color == "white" else "white" 
            moves_dict[other_color] = extract_moves(temp_board)
    
        return moves_dict

    def _get_safe_mobility(self, board: chess.Board) -> dict:
        safe_moves_dict = {"white": {}, "black": {}}
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
        }

        def extract_safe_moves(b: chess.Board, active_color: chess.Color):
            d = {}
            enemy_color = not active_color
            
            # Define piece values inside or pass them in (assuming they are in scope)
            piece_values = {
                chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
            }
        
            for move in b.legal_moves:
                from_sq = chess.square_name(move.from_square)
                to_sq = chess.square_name(move.to_square)
                
                # 1. Evaluate the moving piece
                moving_piece = b.piece_at(move.from_square)
                if not moving_piece:
                    continue
                    
                if move.promotion:
                    moving_val = piece_values[move.promotion]
                else:
                    moving_val = piece_values[moving_piece.piece_type]
        
                # 2. Evaluate the captured piece (if any) BEFORE the push
                captured_val = 0
                if b.is_en_passant(move):
                    captured_val = piece_values[chess.PAWN]
                elif b.is_capture(move):
                    captured_piece = b.piece_at(move.to_square)
                    if captured_piece:
                        captured_val = piece_values[captured_piece.piece_type]
        
                # 3. Push the move to simulate the future board state
                b.push(move)
                
                # 4. Evaluate safety on the new board state
                enemy_attackers = b.attackers(enemy_color, move.to_square)
                is_safe = True
                
                if enemy_attackers:
                    defenders = b.attackers(active_color, move.to_square)
                    
                    # If we lose the piece but captured something better/equal, it's still safe
                    net_material_change = captured_val - moving_val
        
                    if not defenders:
                        # We have no defenders. Will we lose more than we just gained?
                        if net_material_change < 0:
                            is_safe = False 
                    else:
                        # We have defenders. The enemy will likely attack with their cheapest piece.
                        # Using a generator (no brackets) inside min() is slightly faster
                        min_attacker_val = min(piece_values[b.piece_at(sq).piece_type] for sq in enemy_attackers)
                        
                        # If they attack with a cheaper piece, and our initial capture didn't cover the loss
                        if min_attacker_val <= moving_val and net_material_change < 0:
                            is_safe = False
        
                # 5. ALWAYS pop the move to restore the board
                b.pop()
        
                if is_safe:
                    if from_sq not in d:
                        d[from_sq] = []
                    if to_sq not in d[from_sq]:
                        d[from_sq].append(to_sq)
                        
            return d

        return safe_moves_dict

    def _get_check_and_mate_threats(self, board: chess.Board) -> dict:
        checks = {"white": [], "black": []}
        mates = {"white": [], "black": []}

        def evaluate_threats(b: chess.Board, color_key: str):
            for move in b.legal_moves:
                if b.gives_check(move):
                    move_pair = [chess.square_name(move.from_square), chess.square_name(move.to_square)]
                    
                    # PERFORMANCE FIX 1: Deduplicate promotion moves.
                    # A pawn promoting to Q, R, B, or N might all give check on the same square.
                    if move_pair not in checks[color_key]:
                        checks[color_key].append(move_pair)
                    
                    b.push(move)
                    try:
                        if b.is_checkmate():
                            if move_pair not in mates[color_key]:
                                mates[color_key].append(move_pair)
                    finally:
                        # CRASH FIX 1: Always pop the move, even if is_checkmate() errors out
                        b.pop()

        current_color = "white" if board.turn == chess.WHITE else "black"
        other_color = "black" if current_color == "white" else "white"
        
        # 1. Evaluate active player's immediate threats
        evaluate_threats(board, current_color)

        # 2. Evaluate opponent's threats
        if not board.is_check():
            temp_board = board.copy(stack=False)
            
            # CRASH FIX 2: Use null move to safely flip the turn and clear en passant caches
            temp_board.push(chess.Move.null())
            
            evaluate_threats(temp_board, other_color)

        return {"checks": checks, "mates": mates}


        # ==========================================
    # Pawn Structure Features
    # ==========================================
    def _get_passed_pawns(self, board: chess.Board) -> dict:
        passed = {"white": [], "black": []}
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            enemy_color = not color
            forward = 1 if color == chess.WHITE else -1

            for sq in board.pieces(chess.PAWN, color):
                f, r = chess.square_file(sq), chess.square_rank(sq)
                is_passed = True
                
                # Check ranks ahead for enemy pawns on the same or adjacent files
                check_ranks = range(r + forward, 8) if color == chess.WHITE else range(r + forward, -1, -1)
                for cr in check_ranks:
                    for check_f in [f - 1, f, f + 1]:
                        if 0 <= check_f <= 7:
                            enemy_sq = chess.square(check_f, cr)
                            p = board.piece_at(enemy_sq)
                            if p and p.piece_type == chess.PAWN and p.color == enemy_color:
                                is_passed = False
                                break
                    if not is_passed: break
                    
                if is_passed:
                    passed[color_key].append(chess.square_name(sq))
        return passed

    def _get_isolated_pawns(self, board: chess.Board) -> dict:
        isolated = {"white": [], "black": []}
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            pawns = board.pieces(chess.PAWN, color)
            pawn_files = {chess.square_file(sq) for sq in pawns}

            for sq in pawns:
                f = chess.square_file(sq)
                if (f - 1) not in pawn_files and (f + 1) not in pawn_files:
                    isolated[color_key].append(chess.square_name(sq))
        return isolated

    def _get_doubled_pawns(self, board: chess.Board) -> dict:
        doubled = {"white": [], "black": []}
        files_str = "abcdefgh"
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            file_counts = {}
            for sq in board.pieces(chess.PAWN, color):
                f = chess.square_file(sq)
                file_counts[f] = file_counts.get(f, 0) + 1

            for f, count in file_counts.items():
                if count >= 2:
                    doubled[color_key].append(files_str[f]) # Returns the letter (e.g. "c")
        return doubled


    def _get_pawn_islands(self, board: chess.Board) -> dict:
        """
        Groups pawns into islands (arrays of pawns). 
        An island is a group of pawns separated from others of the same color by at least one empty file.
        """
        islands = {"white": [], "black": []}
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            
            current_island = []
            # Sweep left-to-right across the board files (0 to 7)
            for f in range(8):
                # Find all pawns of this color on the current file
                pawns_on_file = [
                    chess.square_name(sq) for sq in board.pieces(chess.PAWN, color) 
                    if chess.square_file(sq) == f
                ]
                
                if pawns_on_file:
                    # If pawns exist on this file, they belong to the current contiguous island
                    current_island.extend(pawns_on_file)
                elif current_island:
                    # If this file is empty but we were tracking an island, the gap closes the island
                    islands[color_key].append(current_island)
                    current_island = []
                    
            # Catch any island that reached the edge of the board
            if current_island:
                islands[color_key].append(current_island)
                
        return islands

    def _get_pawn_rams(self, board: chess.Board) -> list:
        """
        Finds pawns that are locked head-to-head.
        Returns a global list of paired squares, e.g., [["d4", "d5"], ["e4", "e5"]].
        """
        rams = []
        for sq in board.pieces(chess.PAWN, chess.WHITE):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            
            # FIXED: Explicitly ensure the pawn isn't magically on rank 8
            if r + 1 <= 7: 
                front_sq = chess.square(f, r + 1)
                piece = board.piece_at(front_sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                    rams.append([chess.square_name(sq), chess.square_name(front_sq)])
                    
        return rams

    def _get_pawn_chains(self, board: chess.Board) -> dict:
        """Finds arrays of connected diagonal pawns."""
        chains = {"white": [], "black": []}
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            pawns = list(board.pieces(chess.PAWN, color))
            visited = set()
            
            for sq in pawns:
                if sq not in visited:
                    # Depth-First Search to find connected components
                    stack = [sq]
                    current_chain = []
                    
                    while stack:
                        curr = stack.pop()
                        if curr not in visited:
                            visited.add(curr)
                            current_chain.append(chess.square_name(curr))
                            cf, cr = chess.square_file(curr), chess.square_rank(curr)
                            
                            for nsq in pawns:
                                if nsq not in visited:
                                    nf, nr = chess.square_file(nsq), chess.square_rank(nsq)
                                    # Connected diagonally
                                    if abs(cf - nf) == 1 and abs(cr - nr) == 1:
                                        stack.append(nsq)
                    
                    if len(current_chain) >= 2:
                        # Sort by rank so the string reads top-to-bottom or bottom-to-top nicely
                        current_chain.sort(key=lambda s: s[1])
                        chains[color_key].append(current_chain)
        return chains


    # ==========================================
    # Advanced Pawn Tactics & Structure
    # ==========================================
    def _get_levers(self, board: chess.Board) -> list:
        """
        Finds Pawn Breaks / Levers (Pawns that can capture each other).
        Returns a global list of paired squares, e.g., [["d4", "e5"]].
        """
        levers = []
        for w_sq in board.pieces(chess.PAWN, chess.WHITE):
            # If a White pawn is attacked by a Black pawn, it's a lever
            for attacker_sq in board.attackers(chess.BLACK, w_sq):
                if board.piece_at(attacker_sq).piece_type == chess.PAWN:
                    pair = [chess.square_name(w_sq), chess.square_name(attacker_sq)]
                    levers.append(pair)
        return levers

    def _get_phalanx_pawns(self, board: chess.Board) -> dict:
        """
        Finds pawns of the same color standing side-by-side on the same rank.
        Returns pairs of squares, e.g., {"white": [["d4", "e4"]], "black": []}.
        """
        phalanx = {"white": [], "black": []}
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            pawns = list(board.pieces(chess.PAWN, color))
            
            for sq1 in pawns:
                f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
                for sq2 in pawns:
                    if sq1 < sq2: # Avoid duplicates (e.g., d4-e4 and e4-d4)
                        f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
                        # Same rank, adjacent files
                        if r1 == r2 and abs(f1 - f2) == 1:
                            phalanx[color_key].append([chess.square_name(sq1), chess.square_name(sq2)])
        return phalanx

    def _get_connected_passers(self, passed_dict: dict) -> dict:
        """
        Filters passed pawns to only return those on adjacent files.
        """
        connected = {"white": [], "black": []}
        for color_key in ["white", "black"]:
            passers = passed_dict[color_key]
            
            # Extract file integers (0 to 7) for all passed pawns of this color
            passer_files = {chess.parse_square(sq) % 8 for sq in passers} 
            
            for sq_name in passers:
                f = chess.parse_square(sq_name) % 8
                if (f - 1) in passer_files or (f + 1) in passer_files:
                    connected[color_key].append(sq_name)
        return connected

    def _get_protected_passers(self, board: chess.Board, passed_dict: dict) -> dict:
        """
        Filters passed pawns to only return those defended by a friendly pawn.
        """
        protected = {"white": [], "black": []}
        for color, color_key in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
            for sq_name in passed_dict[color_key]:
                sq = chess.parse_square(sq_name)
                
                # Check if any attacker of this square is a friendly pawn (meaning it defends the square)
                defenders = board.attackers(color, sq)
                is_protected = any(board.piece_at(d).piece_type == chess.PAWN for d in defenders)
                
                if is_protected:
                    protected[color_key].append(sq_name)
        return protected


    def _get_candidate_passers(self, board: chess.Board, passed_dict: dict) -> dict:
        candidates = {"white": [], "black": []}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            enemy_color = not color
            forward = 1 if color == chess.WHITE else -1
            
            for sq in board.pieces(chess.PAWN, color):
                sq_name = chess.square_name(sq)
                if sq_name in passed_dict[color_key]:
                    continue 
                    
                f, r = chess.square_file(sq), chess.square_rank(sq)
                check_ranks = range(r + forward, 8) if color == chess.WHITE else range(r + forward, -1, -1)
                
                # 1. Ensure NO pawn (friendly or enemy) is directly blocking the file ahead
                blocked_ahead = False
                for cr in check_ranks:
                    p = board.piece_at(chess.square(f, cr))
                    # If ANY pawn is directly in front, this specific pawn cannot be the candidate
                    if p and p.piece_type == chess.PAWN:
                        blocked_ahead = True
                        break
                        
                if not blocked_ahead:
                    local_files = [af for af in [f - 1, f, f + 1] if 0 <= af <= 7]
                    
                    # 2. Friendly majority counts ALL friendly pawns on the local files
                    # This ensures we count supporting pawns behind the candidate
                    friendly_local = sum(
                        1 for af in local_files for cr in range(8)
                        if board.color_at(chess.square(af, cr)) == color and 
                           board.piece_type_at(chess.square(af, cr)) == chess.PAWN
                    )
                    
                    # 3. Enemy blockers ONLY matter if they are strictly ahead of the pawn
                    # Pawns on the same rank or behind are geometrically irrelevant
                    enemy_local = sum(
                        1 for af in local_files for cr in check_ranks
                        if board.color_at(chess.square(af, cr)) == enemy_color and 
                           board.piece_type_at(chess.square(af, cr)) == chess.PAWN
                    )
                    
                    if friendly_local > enemy_local:
                        candidates[color_key].append(sq_name)
                        
        return candidates

    def _get_backward_pawns(self, board: chess.Board) -> dict:
        backward = {"white": [], "black": []}
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            enemy_color = not color
            forward = 1 if color == chess.WHITE else -1
            pawns = list(board.pieces(chess.PAWN, color))
            
            for sq in pawns:
                f, r = chess.square_file(sq), chess.square_rank(sq)
                
                is_rearmost = True
                for other_sq in pawns:
                    if chess.square_file(other_sq) == f:
                        if (color == chess.WHITE and chess.square_rank(other_sq) < r) or \
                           (color == chess.BLACK and chess.square_rank(other_sq) > r):
                            is_rearmost = False
                            break
                if not is_rearmost: continue
                
                has_support_potential = False
                for other_sq in pawns:
                    if abs(chess.square_file(other_sq) - f) == 1:
                        if (color == chess.WHITE and chess.square_rank(other_sq) <= r) or \
                           (color == chess.BLACK and chess.square_rank(other_sq) >= r):
                            has_support_potential = True
                            break
                if has_support_potential: continue

                # BOUNDARY FIX: Check bounds before asking chess.square to parse it
                if 0 <= r + forward <= 7:
                    front_sq = chess.square(f, r + forward)
                    attackers = board.attackers(enemy_color, front_sq)
                    is_held_back = any(board.piece_at(atk).piece_type == chess.PAWN for atk in attackers)
                    
                    if is_held_back:
                        backward[color_key].append(chess.square_name(sq))
                        
        return backward

    def _get_skewers(self, board: chess.Board) -> dict:
        skewers = {"absolute": [], "relative": []}
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
        
        for sq, piece in board.piece_map().items():
            color = piece.color
            enemy_color = not color
            
            enemy_sliders = chess.SquareSet(
                board.pieces(chess.QUEEN, enemy_color) | 
                board.pieces(chess.ROOK, enemy_color) | 
                board.pieces(chess.BISHOP, enemy_color)
            )
            
            for slider_sq in enemy_sliders:
                if sq in board.attacks(slider_sq):
                    
                    board.remove_piece_at(sq)
                    new_attacks = board.attacks(slider_sq)
                    board.set_piece_at(sq, piece)
                    
                    old_attacks = board.attacks(slider_sq)
                    revealed_targets = new_attacks - old_attacks
                    
                    for target_sq in revealed_targets:
                        target = board.piece_at(target_sq)
                        
                        if target and target.color == color:
                            val_front = piece_values[piece.piece_type]
                            val_back = piece_values[target.piece_type]
                            val_attacker = piece_values[board.piece_at(slider_sq).piece_type]
                            
                            # LOGIC FIX: Gather defenders, but completely ignore the skewered piece itself
                            defenders = set(board.attackers(color, target_sq))
                            defenders.discard(sq)
                            is_protected = len(defenders) > 0
                            
                            if is_protected and val_attacker > val_back:
                                continue
                            
                            skewer_data = {
                                "skewerer": chess.square_name(slider_sq),
                                "skewered": chess.square_name(sq),
                                "target": chess.square_name(target_sq),
                                "color": "white" if enemy_color == chess.WHITE else "black"
                            }
                            
                            if piece.piece_type == chess.KING:
                                skewers["absolute"].append(skewer_data)
                            elif val_front > val_back:
                                skewers["relative"].append(skewer_data)
                                
        return skewers


    def _get_pawn_majorities(self, board: chess.Board) -> dict:
        """
        Determines which side holds a pawn majority on the Queenside, Center, and Kingside.
        Returns lists of regions where a side has a majority, e.g., {"white": ["queenside"], "black": ["kingside"]}.
        """
        majorities = {"white": [], "black": []}

        # Helper to count pawns for a specific color across given files
        def count_pawns(color, files):
            return sum(1 for sq in board.pieces(chess.PAWN, color) if chess.square_file(sq) in files)

        # 0,1,2 = a,b,c | 3,4 = d,e | 5,6,7 = f,g,h
        w_qs = count_pawns(chess.WHITE, [0, 1, 2])
        w_c  = count_pawns(chess.WHITE, [3, 4])
        w_ks = count_pawns(chess.WHITE, [5, 6, 7])

        b_qs = count_pawns(chess.BLACK, [0, 1, 2])
        b_c  = count_pawns(chess.BLACK, [3, 4])
        b_ks = count_pawns(chess.BLACK, [5, 6, 7])

        # Evaluate Queenside
        if w_qs > b_qs:
            majorities["white"].append("queenside")
        elif b_qs > w_qs:
            majorities["black"].append("queenside")

        # Evaluate Center
        if w_c > b_c:
            majorities["white"].append("central")
        elif b_c > w_c:
            majorities["black"].append("central")

        # Evaluate Kingside
        if w_ks > b_ks:
            majorities["white"].append("kingside")
        elif b_ks > w_ks:
            majorities["black"].append("kingside")

        return majorities


    # ==========================================
    # King Safety Features
    # ==========================================
    def _get_king_ring_attacks(self, board: chess.Board) -> dict:
        """
        Returns a list of squares containing enemy pieces that are attacking 
        the 8 squares immediately surrounding the king.
        """
        ring_attackers = {"white": [], "black": []}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            enemy_color = not color
            king_sq = board.king(color)
            
            if king_sq is None: 
                continue
            
            kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
            
            # Calculate the 8 adjacent squares
            adj_squares = [
                chess.square(f, r) for f in [kf-1, kf, kf+1] for r in [kr-1, kr, kr+1]
                if 0 <= f <= 7 and 0 <= r <= 7 and not (f == kf and r == kr)
            ]
            
            attackers = set()
            for sq in adj_squares:
                attackers.update(board.attackers(enemy_color, sq))
            
            # Convert internal square integers to names (e.g., 'f3')
            ring_attackers[color_key] = [chess.square_name(sq) for sq in attackers]
            
        return ring_attackers

    def _get_pawn_shield_integrity(self, board: chess.Board) -> dict:
        """
        Returns a dictionary separating the friendly shield pawns into 'rank_1' 
        (immediately in front of the king) and 'rank_2' (pushed one square forward).
        """
        shield_pawns = {
            "white": {"rank_1": [], "rank_2": []}, 
            "black": {"rank_1": [], "rank_2": []}
        }
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            king_sq = board.king(color)
            
            if king_sq is None: 
                continue
            
            kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
            forward = 1 if color == chess.WHITE else -1
            
            # Check the file the King is on, and the two adjacent files
            for f in [kf-1, kf, kf+1]:
                if 0 <= f <= 7:
                    # 1. Check Rank 1 immediately in front of the King
                    r1 = kr + forward
                    found_rank_1 = False
                    if 0 <= r1 <= 7:
                        sq1 = chess.square(f, r1)
                        p1 = board.piece_at(sq1)
                        if p1 and p1.piece_type == chess.PAWN and p1.color == color:
                            shield_pawns[color_key]["rank_1"].append(chess.square_name(sq1))
                            found_rank_1 = True
                            
                    # 2. Check Rank 2 (pushed pawn shield)
                    # We usually only consider it a 'rank 2 shield' if the pawn hasn't already been found on rank 1 for this file
                    if not found_rank_1:
                        r2 = kr + (2 * forward)
                        if 0 <= r2 <= 7:
                            sq2 = chess.square(f, r2)
                            p2 = board.piece_at(sq2)
                            if p2 and p2.piece_type == chess.PAWN and p2.color == color:
                                shield_pawns[color_key]["rank_2"].append(chess.square_name(sq2))
                                    
        return shield_pawns

    def _get_open_files_near_king(self, board: chess.Board) -> dict:
        """
        Returns a list of files near the King (King file + adjacent files) that have absolutely NO pawns on them.
        """
        open_files = {"white": [], "black": []}
        files_str = "abcdefgh"
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            king_sq = board.king(color)
            
            if king_sq is None: 
                continue
            
            kf = chess.square_file(king_sq)
            
            for f in [kf-1, kf, kf+1]:
                if 0 <= f <= 7:
                    pawn_found = False
                    # Scan the entire file for ANY pawn (White or Black)
                    for r in range(8):
                        p = board.piece_at(chess.square(f, r))
                        if p and p.piece_type == chess.PAWN:
                            pawn_found = True
                            break
                            
                    # If the file is completely empty of pawns, it is dangerously open
                    if not pawn_found:
                        open_files[color_key].append(files_str[f])
                        
        return open_files

    # ==========================================
    # Deep Tactical Features
    # ==========================================
    def _get_hanging_pieces(self, board: chess.Board) -> dict:
        """Finds pieces that are under attack but have ZERO defenders, explicitly excluding the King."""
        hanging = {"white": [], "black": []}
        
        for sq, piece in board.piece_map().items():
            # The King cannot be "hanging" in a material sense
            if piece.piece_type == chess.KING:
                continue
                
            color_key = "white" if piece.color == chess.WHITE else "black"
            enemy_color = not piece.color
            
            # If the piece is attacked by an enemy
            if board.attackers(enemy_color, sq):
                # And has no friendly defenders
                if not board.attackers(piece.color, sq):
                    hanging[color_key].append(chess.square_name(sq))
                    
        return hanging

    def _get_capture_tension(self, board: chess.Board) -> dict:
        """
        Returns all possible pseudo-legal captures for both sides using pure bitboard math,
        explicitly excluding captures of the King.
        """
        tension = {"white": [], "black": []}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            enemy_color = not color
            
            # 1. Standard Piece Captures (O(1) Bitwise Intersection)
            # Iterate strictly over squares occupied by this color
            for sq in chess.SquareSet(board.occupied_co[color]):
                
                # board.attacks() calculates rays regardless of whose turn it is
                attacks = board.attacks(sq)
                enemy_targets = attacks & board.occupied_co[enemy_color]
                
                for target_sq in enemy_targets:
                    # MICRO-OPTIMIZATION: Filter out King captures via integer comparison
                    if board.piece_type_at(target_sq) != chess.KING:
                        tension[color_key].append([
                            chess.square_name(sq), 
                            chess.square_name(target_sq)
                        ])
                        
            # 2. En Passant Captures (Only pseudo-legal for the active player)
            if color == board.turn and board.ep_square is not None:
                ep_sq = board.ep_square
                
                # Intersect our pawns with the pieces attacking the EP square
                ep_attackers = board.attackers(color, ep_sq) & board.pieces(chess.PAWN, color)
                
                for attacker_sq in ep_attackers:
                    tension[color_key].append([
                        chess.square_name(attacker_sq), 
                        chess.square_name(ep_sq)
                    ])
                    
        return tension


    def _get_pins(self, board: chess.Board) -> dict:
        """Evaluates both Absolute (pinned to King) and Relative (pinned to Queen/Rook) pins."""
        abs_pins = []
        rel_pins = []
        
        # MICRO-OPTIMIZATION: Calculate the sliders once per board state, not inside the loop
        white_sliders = chess.SquareSet(
            board.pieces(chess.QUEEN, chess.WHITE) | 
            board.pieces(chess.ROOK, chess.WHITE) | 
            board.pieces(chess.BISHOP, chess.WHITE)
        )
        black_sliders = chess.SquareSet(
            board.pieces(chess.QUEEN, chess.BLACK) | 
            board.pieces(chess.ROOK, chess.BLACK) | 
            board.pieces(chess.BISHOP, chess.BLACK)
        )
    
        for sq, piece in board.piece_map().items():
            color = piece.color
            
            # LOGIC FIX 1: The King cannot act as a shield. Skip it.
            if piece.piece_type == chess.KING:
                continue
                
            enemy_sliders = black_sliders if color == chess.WHITE else white_sliders
    
            # 1. Check Absolute Pin (Native python-chess method)
            if board.is_pinned(color, sq):
                pin_ray = board.pin(color, sq)
                pinner_mask = pin_ray & enemy_sliders
                
                if pinner_mask:
                    pinner_sq = pinner_mask.pop()
                    abs_pins.append({
                        "pinned": chess.square_name(sq), 
                        "pinner": chess.square_name(pinner_sq), 
                        "color": "white" if color == chess.WHITE else "black"
                    })
                    
            # 2. Check Relative Pin (Simulation Method)
            else:
                if not enemy_sliders:
                    continue
    
                # PERFORMANCE FIX: Pre-calculate all old attacks BEFORE modifying the board
                old_attacks_dict = {slider: board.attacks(slider) for slider in enemy_sliders}
                
                # Remove the piece exactly ONCE
                board.remove_piece_at(sq)
            
                try:
                    for enemy_slider in enemy_sliders:
                        new_attacks = board.attacks(enemy_slider)
                        old_attacks = old_attacks_dict[enemy_slider]
                        
                        revealed_targets = new_attacks - old_attacks
                        for target_sq in revealed_targets:
                            target = board.piece_at(target_sq)
                            
                            if target and target.color == color:
                                is_queen = target.piece_type == chess.QUEEN and piece.piece_type != chess.QUEEN
                                is_valuable_rook = target.piece_type == chess.ROOK and piece.piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP]
                                
                                if is_queen or is_valuable_rook:
                                    rel_pins.append({
                                        "pinned": chess.square_name(sq), 
                                        "pinner": chess.square_name(enemy_slider), 
                                        "target": chess.square_name(target_sq),
                                        "color": "white" if color == chess.WHITE else "black"
                                    })
                finally:
                    # Restore the piece exactly ONCE, no matter what happens
                    board.set_piece_at(sq, piece)
                
        return {"absolute": abs_pins, "relative": rel_pins}


    def _get_discovered_attacks(self, board: chess.Board) -> list:
        """Finds pieces that, if moved, reveal an attack from a friendly sliding piece."""
        discovered = []
        
        white_sliders = chess.SquareSet(
            board.pieces(chess.QUEEN, chess.WHITE) | 
            board.pieces(chess.ROOK, chess.WHITE) | 
            board.pieces(chess.BISHOP, chess.WHITE)
        )
        black_sliders = chess.SquareSet(
            board.pieces(chess.QUEEN, chess.BLACK) | 
            board.pieces(chess.ROOK, chess.BLACK) | 
            board.pieces(chess.BISHOP, chess.BLACK)
        )
    
        all_sliders = white_sliders | black_sliders
        if not all_sliders:
            return discovered
    
        # MAJOR PERFORMANCE FIX: Calculate initial slider attacks exactly ONCE per board state
        base_attacks = {slider: board.attacks(slider) for slider in all_sliders}
    
        for sq, piece in board.piece_map().items():
            color = piece.color
            enemy_color = not color
            
            # LOGIC FIX: A piece absolutely pinned to its King cannot move to execute a discovered attack
            if board.is_pinned(color, sq):
                continue
                
            friendly_sliders = white_sliders if color == chess.WHITE else black_sliders
            
            sliders_to_check = friendly_sliders.copy()
            sliders_to_check.discard(sq) # A piece cannot discover an attack from itself
            
            if not sliders_to_check:
                continue
            
            # Remove the piece exactly ONCE
            board.remove_piece_at(sq)
            
            try:
                for slider_sq in sliders_to_check:
                    new_attacks = board.attacks(slider_sq)
                    old_attacks = base_attacks[slider_sq]
                    
                    revealed_targets = new_attacks - old_attacks
                    for target_sq in revealed_targets:
                        
                        # MICRO-OPTIMIZATION: Bitwise color check avoids Piece instantiation
                        if board.color_at(target_sq) == enemy_color:
                            discovered.append({
                                "mover": chess.square_name(sq), 
                                "slider": chess.square_name(slider_sq), 
                                "target": chess.square_name(target_sq), 
                                "color": "white" if color == chess.WHITE else "black"
                            })
            finally:
                # Restore the piece exactly ONCE, no matter what happens
                board.set_piece_at(sq, piece)
                
        return discovered

    
    def _get_forks(self, board: chess.Board) -> list:
        """
        Finds instances where a piece attacks 2+ vulnerable targets.
        A target is vulnerable if it is the King, undefended, or of strictly greater value.
        """
        forks = []
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
        }
        
        for sq, piece in board.piece_map().items():
            color = piece.color
            enemy_color = not color
            
            # LOGIC FIX: An absolutely pinned piece can NEVER execute a fork.
            # A fork implies multiple divergent threats; a pinned piece is restricted to one ray.
            if board.is_pinned(color, sq):
                continue
                
            val = piece_values[piece.piece_type]
            
            # PERFORMANCE FIX: Bitwise intersection instantly filters to only enemy-occupied attacked squares
            attacks = board.attacks(sq)
            enemy_targets = attacks & board.occupied_co[enemy_color]
            
            valid_targets = []
            
            for target_sq in enemy_targets:
                # MICRO-OPTIMIZATION: Avoid piece_at() object instantiation
                target_type = board.piece_type_at(target_sq)
                target_val = piece_values[target_type]
                
                # 1. The target is the King (always a massive threat)
                is_king = target_type == chess.KING
                
                # 2. The target is worth strictly more than our attacking piece
                is_higher_value = target_val > val
                
                # 3. The target is completely hanging (no friendly defenders)
                is_undefended = not board.attackers(enemy_color, target_sq)
                
                if is_king or is_higher_value or is_undefended:
                    valid_targets.append(chess.square_name(target_sq))
            
            # A fork requires at least two vulnerable targets
            if len(valid_targets) >= 2:
                forks.append({
                    "forking_piece": chess.square_name(sq), 
                    "targets": valid_targets, 
                    "color": "white" if color == chess.WHITE else "black"
                })
                
        return forks

    def _get_batteries(self, board: chess.Board) -> list:
        """Finds stacked pieces on the same line (e.g. Rook defending a Rook on a file)."""
        batteries = []
        
        for color in [chess.WHITE, chess.BLACK]:
            sliders = chess.SquareSet(board.pieces(chess.QUEEN, color) | board.pieces(chess.ROOK, color) | board.pieces(chess.BISHOP, color))
            
            for sq in sliders:
                piece = board.piece_at(sq)
                attacks = board.attacks(sq)
                
                for target_sq in attacks:
                    target = board.piece_at(target_sq)
                    if target and target.color == color and target.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP]:
                        # Are they functionally aligned?
                        # 1. Orthogonal Battery (Rooks/Queens)
                        if piece.piece_type in [chess.ROOK, chess.QUEEN] and target.piece_type in [chess.ROOK, chess.QUEEN]:
                            if chess.square_file(sq) == chess.square_file(target_sq) or chess.square_rank(sq) == chess.square_rank(target_sq):
                                batteries.append(sorted([chess.square_name(sq), chess.square_name(target_sq)]))
                        # 2. Diagonal Battery (Bishops/Queens)
                        if piece.piece_type in [chess.BISHOP, chess.QUEEN] and target.piece_type in [chess.BISHOP, chess.QUEEN]:
                            if abs(chess.square_file(sq) - chess.square_file(target_sq)) == abs(chess.square_rank(sq) - chess.square_rank(target_sq)):
                                batteries.append(sorted([chess.square_name(sq), chess.square_name(target_sq)]))
        
        # Strip duplicates (since A defends B, and B defends A, we sorted the array to make them identical)
        unique_batteries = []
        for b in batteries:
            if b not in unique_batteries:
                unique_batteries.append(b)
                
        return unique_batteries

    def _get_xrays(self, board: chess.Board) -> dict:
        """
        Finds all enemy pieces that lie on the exact same ray (orthogonal or diagonal) 
        as a sliding piece, but are blocked by at least one piece.
        """
        xrays = {"white": {}, "black": {}}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            enemy_color = not color
            
            # Get all friendly sliding pieces
            sliders = chess.SquareSet(
                board.pieces(chess.QUEEN, color) | 
                board.pieces(chess.ROOK, color) | 
                board.pieces(chess.BISHOP, color)
            )
            
            for slider_sq in sliders:
                piece = board.piece_at(slider_sq)
                xrayed_enemies = []
                
                # Check against ALL enemy pieces on the board
                for enemy_sq in board.pieces(chess.PAWN, enemy_color) | board.pieces(chess.KNIGHT, enemy_color) | \
                                board.pieces(chess.BISHOP, enemy_color) | board.pieces(chess.ROOK, enemy_color) | \
                                board.pieces(chess.QUEEN, enemy_color) | board.pieces(chess.KING, enemy_color):
                    
                    is_on_line = False
                    file_diff = abs(chess.square_file(slider_sq) - chess.square_file(enemy_sq))
                    rank_diff = abs(chess.square_rank(slider_sq) - chess.square_rank(enemy_sq))
                    
                    # Mathematical check: Are they on the same rank/file?
                    if piece.piece_type in [chess.ROOK, chess.QUEEN]:
                        if file_diff == 0 or rank_diff == 0:
                            is_on_line = True
                            
                    # Mathematical check: Are they on the same perfect diagonal?
                    if piece.piece_type in [chess.BISHOP, chess.QUEEN]:
                        if file_diff == rank_diff:
                            is_on_line = True
                            
                    if is_on_line:
                        # If it is directly attacked, it's not an x-ray (it's a normal attack). 
                        # We only want targets that are visually aligned but mathematically blocked.
                        if enemy_sq not in board.attacks(slider_sq):
                            xrayed_enemies.append(chess.square_name(enemy_sq))
                
                if xrayed_enemies:
                    xrays[color_key][chess.square_name(slider_sq)] = xrayed_enemies
                    
        return xrays

    # ==========================================
    # Development & Positional Features
    # ==========================================
    def _get_undeveloped_minors(self, board: chess.Board) -> dict:
        """Finds Knights and Bishops that are still resting on their starting squares."""
        undeveloped = {"white": [], "black": []}
        
        # Standard starting squares for minors
        starting_squares = {
            chess.WHITE: {chess.B1: chess.KNIGHT, chess.G1: chess.KNIGHT, chess.C1: chess.BISHOP, chess.F1: chess.BISHOP},
            chess.BLACK: {chess.B8: chess.KNIGHT, chess.G8: chess.KNIGHT, chess.C8: chess.BISHOP, chess.F8: chess.BISHOP}
        }
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            for sq, piece_type in starting_squares[color].items():
                p = board.piece_at(sq)
                # Ensure the piece hasn't been replaced by something else returning to that square
                if p and p.piece_type == piece_type and p.color == color:
                    undeveloped[color_key].append(chess.square_name(sq))
                    
        return undeveloped

    def _get_rooks_connected(self, board: chess.Board) -> dict:
        """
        Evaluates if rooks are connected (meaning they share a rank/file 
        with NO pieces between them).
        """
        connected = {"white": False, "black": False}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            rooks = list(board.pieces(chess.ROOK, color))
            
            if len(rooks) >= 2:
                for r_sq in rooks:
                    # In python-chess, board.attacks() includes friendly pieces in its bitboard ray.
                    # So if Rook A's attack ray hits Rook B, they are perfectly connected!
                    if any(target in rooks for target in board.attacks(r_sq)):
                        connected[color_key] = True
                        break
                        
        return connected

    def _get_outposts(self, board: chess.Board) -> dict:
        """
        Finds strong outposts: Knights/Bishops on advanced ranks, protected by a pawn,
        and IMPOSSIBLE to be attacked by an enemy pawn (no enemy pawns on adjacent files ahead of it).
        """
        outposts = {"white": [], "black": []}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            enemy_color = not color
            
            # Combine knights and bishops
            minors = list(board.pieces(chess.KNIGHT, color)) + list(board.pieces(chess.BISHOP, color))

            for sq in minors:
                f, r = chess.square_file(sq), chess.square_rank(sq)

                # 1. Must be advanced (Rank 4+ for White, Rank 5- for Black)
                if color == chess.WHITE and r < 3: continue
                if color == chess.BLACK and r > 4: continue

                # 2. Must be protected by a friendly pawn
                defenders = board.attackers(color, sq)
                if not any(board.piece_at(d).piece_type == chess.PAWN for d in defenders):
                    continue

                # 3. Cannot be driven away by enemy pawns
                # Check adjacent files for enemy pawns that are still physically capable of advancing
                is_safe = True
                adj_files = [af for af in [f - 1, f + 1] if 0 <= af <= 7]
                
                for af in adj_files:
                    # White fears black pawns on ranks ahead of it. Black fears white pawns on ranks below it.
                    check_ranks = range(r + 1, 8) if color == chess.WHITE else range(r - 1, -1, -1)
                    
                    for cr in check_ranks:
                        p = board.piece_at(chess.square(af, cr))
                        if p and p.piece_type == chess.PAWN and p.color == enemy_color:
                            is_safe = False
                            break
                    if not is_safe: break

                if is_safe:
                    outposts[color_key].append(chess.square_name(sq))

        return outposts

    def _get_piece_connectivity(self, board: chess.Board) -> dict:
        """
        Creates a map of all occupied squares and the friendly pieces defending them.
        Returns: {"white": {"e4": ["d3", "f3"]}, "black": {"d5": ["c6", "e6", "g8"]}}
        """
        connectivity = {"white": {}, "black": {}}
        
        for sq, piece in board.piece_map().items():
            color_key = "white" if piece.color == chess.WHITE else "black"
            
            # Use python-chess to find all friendly attackers (defenders) of this square
            defenders = board.attackers(piece.color, sq)
            
            connectivity[color_key][chess.square_name(sq)] = [chess.square_name(d) for d in defenders]
                
        return connectivity


    # ==========================================
    # Endgame Specific Features
    # ==========================================
    def _get_king_activity(self, board: chess.Board) -> dict:
        """
        Calculates the King's distance to the center of the board (d4, e4, d5, e5).
        Returns a distance score where 0 is dead center, and 3 is stuck in the corner.
        """
        activity = {"white": 0, "black": 0}
        center_files = [3, 4] # d and e files
        center_ranks = [3, 4] # 4th and 5th ranks (0-indexed)
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            k_sq = board.king(color)
            
            if k_sq is not None:
                kf, kr = chess.square_file(k_sq), chess.square_rank(k_sq)
                # Calculates Chebyshev (King-step) distance to the closest central square
                dist = min(max(abs(kf - cf), abs(kr - cr)) for cf in center_files for cr in center_ranks)
                activity[color_key] = dist
                
        return activity

    def _get_rule_of_the_square(self, board: chess.Board, passed_dict: dict) -> list:
        """
        Evaluates the 'Rule of the Square' for every passed pawn.
        Uses a timeline-interception algorithm to determine if the King can catch 
        the pawn on ANY square along its path to promotion, not just the final square.
        """
        results = []
        
        for color, color_key in [(chess.WHITE, "white"), (chess.BLACK, "black")]:
            enemy_color = not color
            enemy_king = board.king(enemy_color)
            
            if not enemy_king: 
                continue
            
            ek_f, ek_r = chess.square_file(enemy_king), chess.square_rank(enemy_king)
            forward = 1 if color == chess.WHITE else -1
            promo_r = 7 if color == chess.WHITE else 0
            start_r = 1 if color == chess.WHITE else 6
            
            for sq_name in passed_dict[color_key]:
                sq = chess.parse_square(sq_name)
                pf, pr = chess.square_file(sq), chess.square_rank(sq)
                
                can_catch = False
                
                # 1. Immediate capture or block check
                # If it's the defending king's turn and it is standing right next to the pawn
                if board.turn == enemy_color and max(abs(ek_f - pf), abs(ek_r - pr)) <= 1:
                    can_catch = True
                else:
                    # 2. Build the pawn's timeline to promotion
                    current_pr = pr
                    moves_taken = 0
                    
                    while current_pr != promo_r:
                        # Pawns on the starting rank can double-step
                        if current_pr == start_r:
                            current_pr += 2 * forward
                        else:
                            current_pr += forward
                            
                        moves_taken += 1
                        
                        # Calculate how many moves the King has to reach this intercept square.
                        # If the defending king moves first, it gets an extra tempo to intercept.
                        available_moves = moves_taken if board.turn == color else moves_taken + 1
                        
                        # King's Chebyshev distance to the target intercept square
                        k_dist = max(abs(ek_f - pf), abs(ek_r - current_pr))
                        
                        # If the King can reach the square at or before the pawn gets there
                        if k_dist <= available_moves:
                            can_catch = True
                            break
                            
                results.append({
                    "pawn": sq_name,
                    "pawn_color": color_key,
                    "catcher_king": "black" if color == chess.WHITE else "white",
                    "can_catch": can_catch
                })
                
        return results

    def _get_opposition(self, board: chess.Board) -> dict:
        """
        Determines if the Kings face each other directly with an odd number of squares between them.
        Returns a boolean, the type of opposition, and crucially, WHO currently holds it.
        """
        wk = board.king(chess.WHITE)
        bk = board.king(chess.BLACK)
        
        if wk is None or bk is None:
            return {"is_opposition": False, "type": None, "who_has_it": None}
            
        wk_f, wk_r = chess.square_file(wk), chess.square_rank(wk)
        bk_f, bk_r = chess.square_file(bk), chess.square_rank(bk)
        
        f_diff = abs(wk_f - bk_f)
        r_diff = abs(wk_r - bk_r)
        
        is_opposition = False
        opp_type = None
        
        # An odd number of squares BETWEEN the kings means the strict square distance is an EVEN number (2, 4, 6)
        if f_diff == 0 and r_diff > 0 and r_diff % 2 == 0:
            is_opposition = True
            opp_type = "direct_file"
        elif r_diff == 0 and f_diff > 0 and f_diff % 2 == 0:
            is_opposition = True
            opp_type = "direct_rank"
        elif f_diff == r_diff and f_diff > 0 and f_diff % 2 == 0:
            is_opposition = True
            opp_type = "diagonal"
            
        if is_opposition:
            # In chess, the player who DOES NOT have to move is the one who "holds" the opposition
            who_has_it = "black" if board.turn == chess.WHITE else "white"
            return {
                "is_opposition": True, 
                "type": opp_type, 
                "who_has_it": who_has_it
            }
            
        return {"is_opposition": False, "type": None, "who_has_it": None}


    def _get_fianchetto_structures(self, board: chess.Board) -> dict:
        """
        Detects fianchetto structures on the queenside (b-file) and kingside (g-file).
        Returns detailed status of the structure (occupied by a bishop, or a weakness/hole).
        """
        fianchettos = {"white": [], "black": []}
        
        for color in [chess.WHITE, chess.BLACK]:
            color_key = "white" if color == chess.WHITE else "black"
            home_rank = 0 if color == chess.WHITE else 7
            forward = 1 if color == chess.WHITE else -1
            
            for flank_name, flank_file in [("queenside", 1), ("kingside", 6)]:
                pawn_start_sq = chess.square(flank_file, home_rank + forward)
                pawn_on_start = board.piece_at(pawn_start_sq) == chess.Piece(chess.PAWN, color)
                
                # LOGIC FIX: The fianchetto structure exists fundamentally because the pawn 
                # vacated its start square (whether by pushing or capturing).
                pawn_advanced = not pawn_on_start
                
                bishop_sq = pawn_start_sq 
                bishop_present = board.piece_at(bishop_sq) == chess.Piece(chess.BISHOP, color)
                
                if pawn_advanced:
                    fianchettos[color_key].append({
                        "flank": flank_name,
                        "bishop_square": chess.square_name(bishop_sq),
                        "is_occupied": bishop_present,
                        "is_hole": not bishop_present 
                    })
                    
        return fianchettos


    def _get_opposite_castling(self, board: chess.Board) -> bool:
        """
        Determines if the players have castled on opposite sides of the board.
        Uses king file position as a heuristic (Queenside = files a/b/c, Kingside = files g/h).
        """
        wk_sq = board.king(chess.WHITE)
        bk_sq = board.king(chess.BLACK)
        
        if wk_sq is None or bk_sq is None:
            return False
            
        wk_file = chess.square_file(wk_sq)
        bk_file = chess.square_file(bk_sq)
        
        # Define castled flank boundaries: 
        # Queenside is file <= 2 (a, b, c)
        # Kingside is file >= 6 (g, h)
        w_castled_qs = wk_file <= 2
        w_castled_ks = wk_file >= 6
        
        b_castled_qs = bk_file <= 2
        b_castled_ks = bk_file >= 6
        
        # Return True if they are on explicitly opposite flanks
        return (w_castled_qs and b_castled_ks) or (w_castled_ks and b_castled_qs)