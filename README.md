## Receptive Field Calculator (Flask)

A very simple Flask web app to interactively compute the cumulative receptive field (RF), effective stride, spatial sizes, and channels of a CNN as you add layers (Conv2D and MaxPool2D). Includes a 3-panel visualization showing how a kernel reads the input to produce outputs.

### What it does
- Add layers one by one (Conv2D or MaxPool2D).
- Shows cumulative RF, effective stride, output sizes, and channels.
- New: Visualize up to the first 3 layers with:
  - Input grid (numbers)
  - Kernel window at a chosen output position
  - Output grid with highlighted cell

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

2) Set input image shape (H, W, C) at the top and click “Set Input”.

3) In “Add Layer”:
- Select `Conv2D` or `MaxPool2D`.
- Enter `Kernel (k)`, `Stride (s)`, `Padding (p)`, and (for Conv2D) `Dilation (d)` and optional `Out Channels`.

4) Click “Add Layer”. The table updates with RF, effective stride, and output sizes.

5) Scroll to “3-Layer Visualizer”:
- Choose the layer to visualize (1..3).
- Pick an output coordinate `(Y, X)` for that layer.
- The visualization shows:
  - Left: The input grid to that layer (numbers).
  - Middle: The kernel window (cells used to compute the selected output cell). Dilation/stride/padding are all reflected.
  - Right: The output grid; your selected `(Y, X)` cell is highlighted.

Notes:
- The visual uses a small single-channel preview (max 8×8) with unit conv weights to keep it readable.
- For `MaxPool2D`, the output cell is the max of the highlighted input patch.
- For `Conv2D`, the output cell is the sum of the highlighted input patch (unit weights).

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

## Next Steps (Ideas)

- Allow choosing kernel weights and show weighted sum.
- Animate the kernel sweeping across the feature map.
- Show multi-channel overlays (sum across input channels).