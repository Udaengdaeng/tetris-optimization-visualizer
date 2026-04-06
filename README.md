# Tetris Sequential Optimizer

A Streamlit web app that optimizes a Tetris block sequence under real Tetris rules:
- pieces arrive in a fixed order
- pieces fall from top to bottom (hard drop)
- complete lines are cleared
- the app searches over rotations and x-positions to choose strong placements

## Files
- `app.py`: Streamlit app
- `requirements.txt`: dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input Excel format
Create an `.xlsx` file with a column named `shape`.
Each row should contain one Tetris block code:

- `I`
- `O`
- `T`
- `S`
- `Z`
- `J`
- `L`

Example:

| shape |
|------|
| I |
| O |
| T |
| L |

If there is no `shape` column, the app uses the first column automatically.

## Notes for users
- Input order = falling order.
- Empty cells are ignored.
- Lowercase input is converted to uppercase.
- Invalid block codes will stop execution.
- Larger search settings may improve quality but increase runtime.

## Deploy
You can upload this repository to GitHub and deploy using:
- Streamlit Community Cloud
- Render
- other Streamlit-compatible hosting
