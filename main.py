# class_code/rf_app/app.py
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "dev-secret-key"  # replace for production


def compute_rf(layers):
    """
    Compute cumulative receptive field and effective stride (jump) after each layer.
    Assumptions:
      - Square kernels/strides/padding/dilation.
      - 'padding' does not change RF size (it affects alignment, not the size).
      - RF formula:
          rf_out   = rf_in + (k - 1) * d * jump_in
          jump_out = jump_in * s
    """
    results = []
    rf = 1
    jump = 1

    for idx, layer in enumerate(layers, start=1):
        k = layer.get("kernel", 1)
        s = layer.get("stride", 1)
        d = layer.get("dilation", 1)
        # padding kept for completeness; not used in rf size
        _p = layer.get("padding", 0)

        rf = rf + (k - 1) * d * jump
        jump = jump * s

        results.append({
            "index": idx,
            "type": layer["type"],
            "kernel": k,
            "stride": s,
            "dilation": d,
            "padding": _p,
            "rf": rf,
            "jump": jump,
        })
    return results


@app.route("/", methods=["GET"])
def index():
    if "layers" not in session:
        session["layers"] = []
    layers = session["layers"]
    computed = compute_rf(layers)
    return render_template("index.html", layers=layers, computed=computed)


@app.route("/add", methods=["POST"])
def add_layer():
    if "layers" not in session:
        session["layers"] = []

    layer_type = request.form.get("type", "Conv2D")
    kernel = int(request.form.get("kernel", "3"))
    stride = int(request.form.get("stride", "1"))
    padding = int(request.form.get("padding", "0"))
    dilation = int(request.form.get("dilation", "1"))

    # For MaxPool, dilation is effectively 1
    if layer_type == "MaxPool2D":
        dilation = 1

    session["layers"].append({
        "type": layer_type,
        "kernel": kernel,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
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
    app.run(debug=True)
