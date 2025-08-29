## Receptive Field Calculator (Flask)

A very simple Flask web app to interactively compute the cumulative receptive field (RF) and effective stride (“jump”) of a CNN as you add layers (Conv2D and MaxPool2D).

### What it does
- Add layers one by one (Conv2D or MaxPool2D).
- Shows cumulative RF size and effective stride after each layer.
- You can remove the last layer or reset all.

---

## Quick Start (Windows/PowerShell)

From the workspace root (`C:\Users\dines\Learning\TSAI\Session 2`):

```bash
# 1) (Optional) create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 2) Install Flask
pip install flask

# 3) Run the app
python .\class_code\main.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Using the App (Step by Step)

1) Go to `http://127.0.0.1:5000`.

2) In the form:
- Select a layer type: `Conv2D` or `MaxPool2D`.
- Enter:
  - `Kernel (k)` (e.g., 3)
  - `Stride (s)` (e.g., 1 or 2)
  - `Padding (p)` (kept for completeness; doesn’t change RF size)
  - `Dilation (d)` (Conv2D only)

3) Click “Add Layer”.

4) The table shows each layer and the cumulative:
- RF size (how many input pixels influence one output position)
- Effective stride (“jump”) at that layer

5) Use:
- “Remove Last Layer” to undo the most recent addition
- “Reset All” to clear all layers

---

## How RF is Computed

We track two values as we go through layers:
- RF (receptive field size), starts at 1
- jump (effective stride), starts at 1

For a layer with kernel \(k\), stride \(s\), dilation \(d\):
- RF_out = RF_in + (k - 1) × d × jump_in
- jump_out = jump_in × s

Notes:
- Padding does not change the RF size (it affects alignment only).
- For `MaxPool2D`, dilation is effectively 1.

---

## Examples

- Add `Conv2D(k=3, s=1, d=1)`:
  - RF: 3, jump: 1
- Then add `MaxPool2D(k=2, s=2)`:
  - RF: 4, jump: 2
- Then add `Conv2D(k=3, s=1, d=1)`:
  - RF: 8, jump: 2

---

## Project Files

- `class_code/main.py`: Flask app (routes and RF logic).
- `class_code/templates/index.html`: The UI template.

---

## Troubleshooting

- Flask not installed:
  ```bash
  pip install flask
  ```
- Port already in use (5000):
  - Stop the other process or run:
    ```bash
    set FLASK_RUN_PORT=5001
    python .\class_code\main.py
    ```
- Layers don’t persist between requests:
  - Make sure `app.secret_key` is set in `class_code/main.py` (already provided).

---

## Optional: Virtual Environment Tips

```bash
# Create venv
python -m venv .venv

# Activate (PowerShell)
.\.venv\Scripts\activate

# Deactivate
deactivate
```

---

## Next Steps (Easy Extensions)

- Show output feature map size given an input size.
- Add average pooling and general padding effects on alignment.
- Export/import the layer list as JSON.
