import io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Tetris Sequential Optimizer",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Styling
# -------------------------
st.markdown(
    """
<style>
.block-container {
    max-width: 1380px;
    padding-top: 2.2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: clamp(2rem, 2.6vw, 3rem);
    font-weight: 800;
    color: #0f172a;
    line-height: 1.2;
    margin: 0;
    padding-top: 0.15rem;
    padding-bottom: 0.15rem;
    word-break: keep-all;
}

.sub-title {
    color: #475569;
    font-size: 1rem;
    margin-top: 0.35rem;
    margin-bottom: 1.25rem;
}

.section-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.15rem 1.15rem 1rem 1.15rem;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    margin-bottom: 1rem;
}

.section-title {
    margin: 0 0 0.8rem 0;
    font-size: 1.05rem;
    font-weight: 800;
    color: #0f172a;
}

.info-chip {
    display: inline-block;
    padding: 0.30rem 0.65rem;
    border-radius: 999px;
    background: #ecfeff;
    color: #155e75;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 0.35rem;
    margin-bottom: 0.45rem;
}

.metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 0.9rem 1rem;
}

.small-muted {
    color: #64748b;
    font-size: 0.88rem;
}

.guide-list {
    margin: 0;
    padding-left: 1.2rem;
}

.guide-list li {
    margin-bottom: 0.45rem;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 12px;
    font-weight: 700;
}

h1, h2, h3, h4, h5, h6 {
    scroll-margin-top: 100px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Constants
# -------------------------
DEFAULT_WIDTH = 10
DEFAULT_HEIGHT = 20
DEFAULT_BEAM_WIDTH = 20
DEFAULT_TOP_K_MOVES = 12

BASE_PIECES: Dict[str, List[Tuple[int, int]]] = {
    "I": [(0, 0), (1, 0), (2, 0), (3, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}

PIECE_COLORS = {
    "I": "#38bdf8",
    "O": "#facc15",
    "T": "#c084fc",
    "S": "#4ade80",
    "Z": "#f87171",
    "J": "#60a5fa",
    "L": "#fb923c",
}
PIECE_CODE = {name: i + 1 for i, name in enumerate(BASE_PIECES.keys())}
CODE_TO_PIECE = {v: k for k, v in PIECE_CODE.items()}
VALID_SHAPES = list(BASE_PIECES.keys())


# -------------------------
# Core logic
# -------------------------
def normalize_cells(cells: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    min_x = min(x for x, y in cells)
    min_y = min(y for x, y in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def rotate_90(cells: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    rotated = [(y, -x) for x, y in cells]
    return normalize_cells(rotated)


def generate_rotations(base_cells: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], ...]]:
    rots = []
    current = normalize_cells(base_cells)
    for _ in range(4):
        if current not in rots:
            rots.append(current)
        current = rotate_90(list(current))
    return rots


PIECE_ROTATIONS = {name: generate_rotations(cells) for name, cells in BASE_PIECES.items()}


@dataclass
class State:
    board: np.ndarray
    heuristic_total: float
    game_score_total: int
    total_lines: int
    placements: list


def make_empty_board(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width), dtype=int)


def piece_dimensions(cells) -> Tuple[int, int]:
    max_x = max(x for x, y in cells)
    max_y = max(y for x, y in cells)
    return max_x + 1, max_y + 1


def can_place(board: np.ndarray, cells, px: int, py: int) -> bool:
    h, w = board.shape
    for x, y in cells:
        bx = px + x
        by = py + y
        if bx < 0 or bx >= w or by < 0 or by >= h:
            return False
        if board[by, bx] != 0:
            return False
    return True


def hard_drop_y(board: np.ndarray, cells, px: int) -> Optional[int]:
    py = 0
    if not can_place(board, cells, px, py):
        return None
    while True:
        next_py = py + 1
        if can_place(board, cells, px, next_py):
            py = next_py
        else:
            return py


def place_piece(board: np.ndarray, cells, px: int, py: int, piece_code_int: int) -> np.ndarray:
    new_board = board.copy()
    for x, y in cells:
        new_board[py + y, px + x] = piece_code_int
    return new_board


def clear_lines(board: np.ndarray) -> Tuple[np.ndarray, int]:
    full_mask = np.all(board != 0, axis=1)
    cleared = int(np.sum(full_mask))
    if cleared == 0:
        return board.copy(), 0
    remaining = board[~full_mask]
    new_rows = np.zeros((cleared, board.shape[1]), dtype=int)
    return np.vstack([new_rows, remaining]), cleared


def column_heights(board: np.ndarray) -> List[int]:
    h, w = board.shape
    heights = []
    for c in range(w):
        filled = np.where(board[:, c] != 0)[0]
        heights.append(0 if len(filled) == 0 else h - filled[0])
    return heights


def count_holes(board: np.ndarray) -> int:
    h, w = board.shape
    holes = 0
    for c in range(w):
        filled_seen = False
        for r in range(h):
            if board[r, c] != 0:
                filled_seen = True
            elif filled_seen:
                holes += 1
    return holes


def bumpiness(heights: List[int]) -> int:
    return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))


def max_height(heights: List[int]) -> int:
    return max(heights) if heights else 0


def line_clear_game_score(lines_cleared: int) -> int:
    return {0: 0, 1: 100, 2: 300, 3: 500}.get(lines_cleared, 800)


def evaluate_board_tetris(board: np.ndarray, lines_cleared_recent: int = 0) -> float:
    heights = column_heights(board)
    holes = count_holes(board)
    bump = bumpiness(heights)
    mh = max_height(heights)
    agg_height = sum(heights)
    score = (
        8.0 * lines_cleared_recent
        - 4.5 * holes
        - 0.7 * bump
        - 0.45 * agg_height
        - 0.8 * mh
    )
    return score


def generate_valid_moves(board: np.ndarray, piece_name: str) -> list:
    moves = []
    for ridx, cells in enumerate(PIECE_ROTATIONS[piece_name]):
        pw, _ = piece_dimensions(cells)
        for px in range(0, board.shape[1] - pw + 1):
            py = hard_drop_y(board, cells, px)
            if py is None:
                continue
            placed = place_piece(board, cells, px, py, PIECE_CODE[piece_name])
            board_after, lines = clear_lines(placed)
            heuristic_score = evaluate_board_tetris(board_after, lines_cleared_recent=lines)
            moves.append({
                "piece": piece_name,
                "rotation_idx": ridx,
                "cells": cells,
                "x": px,
                "y": py,
                "board_after": board_after,
                "lines_cleared": lines,
                "immediate_score": heuristic_score,
            })
    moves.sort(key=lambda m: (m["immediate_score"], m["lines_cleared"]), reverse=True)
    return moves


def beam_search_tetris(sequence: List[str], width: int, height: int, beam_width: int, top_k_moves: int):
    init_state = State(
        board=make_empty_board(width, height),
        heuristic_total=0.0,
        game_score_total=0,
        total_lines=0,
        placements=[],
    )
    beam = [init_state]

    for piece_name in sequence:
        next_beam = []
        for state in beam:
            valid_moves = generate_valid_moves(state.board, piece_name)
            if not valid_moves:
                next_beam.append(
                    State(
                        board=state.board.copy(),
                        heuristic_total=state.heuristic_total - 1000,
                        game_score_total=state.game_score_total,
                        total_lines=state.total_lines,
                        placements=state.placements + [{"piece": piece_name, "failed": True}],
                    )
                )
                continue

            for mv in valid_moves[:top_k_moves]:
                add_game_score = line_clear_game_score(mv["lines_cleared"])
                next_beam.append(
                    State(
                        board=mv["board_after"],
                        heuristic_total=state.heuristic_total + mv["immediate_score"],
                        game_score_total=state.game_score_total + add_game_score,
                        total_lines=state.total_lines + mv["lines_cleared"],
                        placements=state.placements + [{
                            "piece": piece_name,
                            "rotation_idx": mv["rotation_idx"],
                            "x": mv["x"],
                            "y": mv["y"],
                            "lines_cleared": mv["lines_cleared"],
                            "heuristic_score": mv["immediate_score"],
                            "game_score_gain": add_game_score,
                            "failed": False,
                        }],
                    )
                )

        next_beam.sort(key=lambda s: (s.heuristic_total, s.game_score_total, s.total_lines), reverse=True)
        beam = next_beam[:beam_width]

    return max(beam, key=lambda s: (s.heuristic_total, s.game_score_total, s.total_lines))


def replay_sequence(sequence: List[str], placements: list, width: int, height: int, max_steps=None):
    board = make_empty_board(width, height)
    history = []
    steps = min(len(sequence), len(placements))
    if max_steps is not None:
        steps = min(steps, max_steps)

    total_lines = 0
    total_game_score = 0

    for i in range(steps):
        piece = sequence[i]
        pl = placements[i]
        if pl.get("failed", False):
            break
        cells = PIECE_ROTATIONS[piece][pl["rotation_idx"]]
        py = hard_drop_y(board, cells, pl["x"])
        if py is None:
            break
        board = place_piece(board, cells, pl["x"], py, PIECE_CODE[piece])
        board, lines = clear_lines(board)
        total_lines += lines
        total_game_score += line_clear_game_score(lines)
        history.append({
            "board": board.copy(),
            "piece": piece,
            "rotation_idx": pl["rotation_idx"],
            "x": pl["x"],
            "cleared": lines,
        })

    return board, history, total_lines, total_game_score


def plot_board(board: np.ndarray, title: str = "Board"):
    h, w = board.shape
    fig, ax = plt.subplots(figsize=(max(4.5, w * 0.52), max(7.2, h * 0.38)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")
    ax.set_title(title, fontsize=14, weight="bold", pad=10)

    for r in range(h):
        for c in range(w):
            val = board[r, c]
            color = "white" if val == 0 else PIECE_COLORS[CODE_TO_PIECE[val]]
            rect = plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#1f2937", linewidth=0.8)
            ax.add_patch(rect)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(w + 1))
    ax.set_yticks(range(h + 1))
    ax.tick_params(labelsize=8)
    return fig


def make_template_xlsx() -> bytes:
    sample = pd.DataFrame({"shape": ["I", "O", "T", "L", "S", "Z", "J", "T"]})
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        sample.to_excel(writer, index=False, sheet_name="input")
    bio.seek(0)
    return bio.getvalue()


def parse_uploaded_excel(uploaded_file) -> List[str]:
    df = pd.read_excel(uploaded_file)
    col = None
    for c in df.columns:
        if str(c).strip().lower() == "shape":
            col = c
            break
    if col is None:
        col = df.columns[0]

    seq = df[col].dropna().astype(str).str.strip().str.upper().tolist()
    invalid = [s for s in seq if s not in VALID_SHAPES]
    if invalid:
        raise ValueError(f"지원하지 않는 블록 종류가 있습니다: {sorted(set(invalid))}")
    if not seq:
        raise ValueError("유효한 블록 시퀀스가 없습니다.")
    return seq


def placements_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="placements")
    bio.seek(0)
    return bio.getvalue()


# -------------------------
# Sidebar
# -------------------------
st.sidebar.markdown("### ⚙️ Settings")
board_width = st.sidebar.number_input("Board width", min_value=6, max_value=20, value=DEFAULT_WIDTH, step=1)
board_height = st.sidebar.number_input("Board height", min_value=10, max_value=40, value=DEFAULT_HEIGHT, step=1)
beam_width = st.sidebar.slider("Beam width", min_value=5, max_value=60, value=DEFAULT_BEAM_WIDTH, step=1)
top_k_moves = st.sidebar.slider("Top-K moves per state", min_value=4, max_value=30, value=DEFAULT_TOP_K_MOVES, step=1)
show_steps_limit = st.sidebar.slider("Replay max steps", min_value=5, max_value=100, value=20, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Supported blocks")
st.sidebar.markdown(" ".join([f"`{s}`" for s in VALID_SHAPES]))

# -------------------------
# Header
# -------------------------
st.markdown('<div class="main-title">🧩 Tetris Sequential Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">엑셀로 입력한 블록 시퀀스를 실제 테트리스 규칙(위에서 아래로 낙하, 줄 삭제 적용)으로 최적 배치합니다.</div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns([1.3, 1.1], gap="large")

with col_a:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Input Guide</div>', unsafe_allow_html=True)
    st.markdown("**엑셀 입력 방법**")
    st.markdown(
        """
<ol class="guide-list">
    <li>첫 번째 열 이름을 가능하면 <strong><code>shape</code></strong> 로 만듭니다.</li>
    <li>각 행에 블록 종류를 한 개씩 입력합니다.</li>
    <li>허용값은 <strong>I, O, T, S, Z, J, L</strong> 입니다.</li>
    <li>입력 순서가 곧 <strong>블록이 떨어지는 순서</strong>입니다.</li>
    <li>빈 셀은 무시됩니다.</li>
</ol>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**예시**")
    st.dataframe(
        pd.DataFrame({"shape": ["I", "O", "T", "L", "S", "Z", "J"]}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**유의사항**")
    st.markdown(
        """
- 소문자 입력도 가능하지만 자동으로 대문자로 변환됩니다.
- 지원하지 않는 블록 문자가 있으면 실행되지 않습니다.
- 블록 수가 많을수록 탐색 시간이 증가합니다.
- Beam width와 Top-K가 클수록 더 좋은 해를 찾을 수 있지만 실행 시간이 늘어납니다.
        """
    )

    st.download_button(
        label="📄 엑셀 템플릿 다운로드",
        data=make_template_xlsx(),
        file_name="tetris_input_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚀 Upload & Run</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "엑셀 파일 업로드 (.xlsx)",
        type=["xlsx"],
        help="shape 열 또는 첫 번째 열에 I,O,T,S,Z,J,L 순서를 입력하세요.",
    )

    example_text = " / ".join([f"{k}" for k in VALID_SHAPES])
    st.markdown(f"<span class='info-chip'>Allowed: {example_text}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='info-chip'>Board: {board_width} × {board_height}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='info-chip'>Beam width: {beam_width}</span>", unsafe_allow_html=True)
    st.markdown(f"<span class='info-chip'>Top-K: {top_k_moves}</span>", unsafe_allow_html=True)

    run = st.button("▶ Run Optimization", type="primary", use_container_width=True)
    st.markdown(
        '<p class="small-muted">실행 후 최종 보드, 단계별 replay, placements 파일을 다운로드할 수 있습니다.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Run section
# -------------------------
if run:
    if uploaded_file is None:
        st.error("엑셀 파일을 먼저 업로드해줘.")
    else:
        try:
            sequence = parse_uploaded_excel(uploaded_file)
        except Exception as e:
            st.error(f"입력 파일을 읽는 중 오류가 발생했어: {e}")
            st.stop()

        with st.spinner("블록 시퀀스를 최적 배치 중이야..."):
            best = beam_search_tetris(
                sequence=sequence,
                width=board_width,
                height=board_height,
                beam_width=beam_width,
                top_k_moves=top_k_moves,
            )
            final_board, history, replay_lines, replay_game_score = replay_sequence(
                sequence,
                best.placements,
                width=board_width,
                height=board_height,
            )

        filled_cells = int(np.sum(final_board != 0))
        fill_ratio = filled_cells / (board_width * board_height)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Blocks in sequence", len(sequence))
        with m2:
            st.metric("Lines cleared", replay_lines)
        with m3:
            st.metric("Game score", replay_game_score)
        with m4:
            st.metric("Board fill ratio", f"{fill_ratio:.1%}")

        tab1, tab2, tab3, tab4 = st.tabs(["Final Board", "Replay", "Placements", "Input Summary"])

        with tab1:
            c1, c2 = st.columns([1.25, 0.9], gap="large")
            with c1:
                fig = plot_board(final_board, title=f"Final Tetris Board | lines={replay_lines}, game_score={replay_game_score}")
                st.pyplot(fig, use_container_width=True)
            with c2:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Result Summary</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
- **Board size:** {board_width} × {board_height}
- **Input blocks:** {len(sequence)}
- **Lines cleared:** {replay_lines}
- **Game score:** {replay_game_score}
- **Search heuristic total:** {best.heuristic_total:.2f}
- **Failed placements:** {sum(1 for p in best.placements if p.get('failed', False))}
                    """
                )
                st.markdown("### Interpretation")
                st.markdown(
                    """
- 이 결과는 **실제 테트리스 규칙**으로 재생된 최종 보드다.
- 한 줄이 가득 차면 즉시 삭제된다.
- 각 블록은 선택된 회전과 x 위치에서 **위에서 아래로 hard drop** 된다.
                    """
                )
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            if history:
                max_step = min(len(history), show_steps_limit)
                step = st.slider("Replay step", min_value=1, max_value=max_step, value=min(5, max_step), step=1)
                item = history[step - 1]
                fig = plot_board(
                    item["board"],
                    title=f"Step {step} | piece={item['piece']} | rot={item['rotation_idx']} | x={item['x']} | cleared={item['cleared']}",
                )
                st.pyplot(fig, use_container_width=True)
            else:
                st.warning("표시할 replay history가 없어.")

        with tab3:
            placements_df = pd.DataFrame(best.placements)
            st.dataframe(placements_df, use_container_width=True, hide_index=True)
            csv_bytes = placements_df.to_csv(index=False).encode("utf-8-sig")
            xlsx_bytes = placements_to_xlsx_bytes(placements_df)
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "⬇️ Download placements CSV",
                    data=csv_bytes,
                    file_name="tetris_mode_placements.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with d2:
                st.download_button(
                    "⬇️ Download placements XLSX",
                    data=xlsx_bytes,
                    file_name="tetris_mode_placements.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

        with tab4:
            s1, s2 = st.columns([0.9, 1.1], gap="large")
            with s1:
                st.markdown("### Uploaded sequence")
                st.dataframe(pd.DataFrame({"shape": sequence}), use_container_width=True, hide_index=True)
            with s2:
                counts = pd.Series(sequence).value_counts().reindex(VALID_SHAPES, fill_value=0)
                st.markdown("### Block counts")
                st.bar_chart(counts)
else:
    st.markdown("---")
    st.info("엑셀 파일을 업로드한 뒤 **Run Optimization**을 누르면 결과가 표시돼.")
