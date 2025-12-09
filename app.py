from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import model1  # trains model, creates model1.model, model1.scaler, model1.X

app = Flask(__name__)

# Use objects created in model1.py
model = model1.model
scaler = model1.scaler
feature_cols = list(model1.X.columns)  # all feature columns used by the model


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", feature_cols=feature_cols, table_html=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return render_template("index.html",
                                   feature_cols=feature_cols,
                                   table_html=None,
                                   error="No file part in request.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html",
                                   feature_cols=feature_cols,
                                   table_html=None,
                                   error="No file selected.")

        # Read CSV into DataFrame
        df_input = pd.read_csv(file)

        # Check that all required columns are present
        missing = set(feature_cols) - set(df_input.columns)
        if missing:
            return render_template(
                "index.html",
                feature_cols=feature_cols,
                table_html=None,
                error=f"Missing columns in CSV: {', '.join(missing)}",
            )

        # Use only the feature columns, in the correct order
        X_new = df_input[feature_cols].values

        # Scale and predict
        X_scaled = scaler.transform(X_new)
        preds = model.predict(X_scaled)

        # Optional: map 0/1 to text
        label_map = {0: "Not Placed", 1: "Placed"}
        df_input["Predicted_Placement"] = [label_map[int(p)] for p in preds]

        # Convert to HTML table
        table_html = df_input.to_html(classes="table", index=False)

        return render_template("index.html",
                               feature_cols=feature_cols,
                               table_html=table_html,
                               error=None)

    except Exception as e:
        return render_template("index.html",
                               feature_cols=feature_cols,
                               table_html=None,
                               error=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
