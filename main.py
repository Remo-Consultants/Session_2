# class_code/rf_app/app.py
from flask import Flask, render_template, request, redirect, url_for, session
import math

app = Flask(__name__)
app.secret_key = "dev-secret-key"  # replace for production


def conv_out_size(n_in, k, s, p, d):
    # PyTorch-style formula
    # floor((n + 2p - d*(k-1) - 1) / s + 1)
    return max(0, math.floor((n_in + 2 * p - d * (k - 1) - 1) / s + 1))


def compute_rf_and_shapes(layers, input_shape):
    """
    Compute cumulative receptive field, effective stride (jump),
    and spatial/channel sizes after each layer.

    input_shape: dict {h, w, c}
    Assumptions: square kernels/strides/padding/dilation.
    """
    results = []
    rf = 1
    jump = 1

    h = input_shape.get("h", 224)
    w = input_shape.get("w", 224)
    c = input_shape.get("c", 3)

    for idx, layer in enumerate(layers, start=1):
        k = layer.get("kernel", 1)
        s = layer.get("stride", 1)
        d = layer.get("dilation", 1)
        p = layer.get("padding", 0)
        t = layer.get("type", "Conv2D")
        out_c = layer.get("out_channels", None)

        # RF update (padding does not change RF size)
        rf = rf + (k - 1) * d * jump
        jump = jump * s

        # Spatial size update
        h = conv_out_size(h, k, s, p, d)
        w = conv_out_size(w, k, s, p, d)

        # Channels
        if t == "Conv2D":
            # If not provided, default to keep the same channels
            c = int(out_c) if (out_c is not None and str(out_c).isdigit()) else c
        else:
            # MaxPool2D keeps channels
            pass

        results.append({
            "index": idx,
            "type": t,
            "kernel": k,
            "stride": s,
            "dilation": d,
            "padding": p,
            "rf": rf,
            "jump": jump,
            "h": h,
            "w": w,
            "c": c,
            "out_channels": c,
        })

    return results


# ---------- Visualization helpers (single-channel, small grids) ----------

def make_grid(h, w):
    # Simple numbered grid for visualization (0..h*w-1)
    return [[r * w + c for c in range(w)] for r in range(h)]


def pad_grid(grid, p):
    if p <= 0:
        return grid
    h, w = len(grid), len(grid[0]) if grid else 0
    new_h, new_w = h + 2 * p, w + 2 * p
    out = [[0 for _ in range(new_w)] for _ in range(new_h)]
    for r in range(h):
        for c in range(w):
            out[r + p][c + p] = grid[r][c]
    return out


def conv2d_unit(grid, k, s, p, d):
    # unit weights, valid conv with padding p and dilation d, stride s
    g = pad_grid(grid, p)
    gh, gw = len(g), len(g[0]) if g else 0
    out_h = conv_out_size(len(grid), k, s, p, d)
    out_w = conv_out_size(len(grid[0]) if grid else 0, k, s, p, d)
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for orow in range(out_h):
        for ocol in range(out_w):
            acc = 0
            # top-left index inside padded grid
            base_r = orow * s
            base_c = ocol * s
            for kr in range(k):
                for kc in range(k):
                    r = base_r + kr * d
                    c = base_c + kc * d
                    if 0 <= r < gh and 0 <= c < gw:
                        acc += g[r][c]
            out[orow][ocol] = acc
    return out


def maxpool2d(grid, k, s, p):
    # dilation not used for pooling, standard max pooling
    g = pad_grid(grid, p)
    gh, gw = len(g), len(g[0]) if g else 0
    out_h = conv_out_size(len(grid), k, s, p, 1)
    out_w = conv_out_size(len(grid[0]) if grid else 0, k, s, p, 1)
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for orow in range(out_h):
        for ocol in range(out_w):
            base_r = orow * s
            base_c = ocol * s
            m = -10**9
            for kr in range(k):
                for kc in range(k):
                    r = base_r + kr
                    c = base_c + kc
                    if 0 <= r < gh and 0 <= c < gw:
                        if g[r][c] > m:
                            m = g[r][c]
            out[orow][ocol] = m if m != -10**9 else 0
    return out


def build_viz(layers, input_shape, limit_layers=3, cap=8):
    """
    Build simple single-channel feature maps for up to 'limit_layers' layers,
    using a small input size cap x cap (or the real size if smaller).
    Returns:
      {
        'layers': [
            {
              'type','k','s','p','d',
              'in': 2D list,
              'out': 2D list,
              'out_h','out_w'
            }, ...
        ]
      }
    """
    # pick a small visualization size (so SVG is readable)
    vh = min(input_shape.get("h", 8), cap)
    vw = min(input_shape.get("w", 8), cap)
    cur = make_grid(vh, vw)

    viz_layers = []
    for i, layer in enumerate(layers[:limit_layers]):
        t = layer.get("type", "Conv2D")
        k = int(layer.get("kernel", 3))
        s = int(layer.get("stride", 1))
        p = int(layer.get("padding", 0))
        d = int(layer.get("dilation", 1))
        entry = {"type": t, "k": k, "s": s, "p": p, "d": d, "in": cur}

        if t == "Conv2D":
            out = conv2d_unit(cur, k, s, p, d)
        else:
            out = maxpool2d(cur, k, s, p)
        entry["out"] = out
        entry["out_h"] = len(out)
        entry["out_w"] = len(out[0]) if out else 0

        viz_layers.append(entry)
        cur = out
    return {"layers": viz_layers}


@app.route("/", methods=["GET"])
def index():
    if "layers" not in session:
        session["layers"] = []
    if "input_shape" not in session:
        session["input_shape"] = {"h": 224, "w": 224, "c": 3}

    layers = session["layers"]
    input_shape = session["input_shape"]
    computed = compute_rf_and_shapes(layers, input_shape)

    # current feature shape: output of the last layer or input if none
    if computed:
        current = {"h": computed[-1]["h"], "w": computed[-1]["w"], "c": computed[-1]["c"]}
    else:
        current = input_shape

    # --- visualization controls (GET params) ---
    viz_layer = int(request.args.get("viz_layer", "1"))
    viz_layer = max(1, min(viz_layer, max(1, min(3, len(layers)))))  # clamp to 1..min(3, len))
    viz = build_viz(layers, input_shape, limit_layers=3)
    viz_layers = viz["layers"]
    selected = viz_layers[viz_layer - 1] if viz_layers and viz_layer - 1 < len(viz_layers) else None

    oy_max = (selected["out_h"] - 1) if selected else 0
    ox_max = (selected["out_w"] - 1) if selected else 0
    oy = max(0, min(int(request.args.get("oy", "0")), oy_max))
    ox = max(0, min(int(request.args.get("ox", "0")), ox_max))

    return render_template(
        "index.html",
        layers=layers,
        computed=computed,
        input_shape=input_shape,
        current=current,
        viz_layers=viz_layers,
        viz_layer=viz_layer,
        oy=oy,
        ox=ox,
        oy_max=oy_max,
        ox_max=ox_max
    )


@app.route("/set_input", methods=["POST"])
def set_input():
    h = int(request.form.get("in_h", "224"))
    w = int(request.form.get("in_w", "224"))
    c = int(request.form.get("in_c", "3"))
    session["input_shape"] = {"h": h, "w": w, "c": c}
    session.modified = True
    return redirect(url_for("index"))


@app.route("/add", methods=["POST"])
def add_layer():
    if "layers" not in session:
        session["layers"] = []

    layer_type = request.form.get("type", "Conv2D")
    kernel = int(request.form.get("kernel", "3"))
    stride = int(request.form.get("stride", "1"))
    padding = int(request.form.get("padding", "0"))
    dilation = int(request.form.get("dilation", "1"))
    out_channels = request.form.get("out_channels", None)

    if layer_type == "MaxPool2D":
        dilation = 1
        out_channels = None  # ignore for pooling

    session["layers"].append({
        "type": layer_type,
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "out_channels": int(out_channels) if (out_channels and out_channels.isdigit()) else None,
    })
    session.modified = True
    return redirect(url_for("index"))


@app.route("/pop", methods=["POST"])
def pop_layer():
    if "layers" in session and session["layers"]:
        session["layers"].pop()
        session.modified = True
    return redirect(url_for("index"))


@app.route("/reset", methods=["POST"])
def reset():
    session["layers"] = []
    session.modified = True
    return redirect(url_for("index"))


if __name__ == "__main__":
app.run(host="0.0.0.0", port=5000, debug=True)