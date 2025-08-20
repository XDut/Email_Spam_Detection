#!/usr/bin/env python3
"""
app.py

Streamlit app to load a saved model pipeline (CountVectorizer + MultinomialNB or embeddings+classifier),
accept text input, and classify messages as Spam/Ham when the user clicks the "Classify" button.
"""

from pathlib import Path
import joblib
import streamlit as st
import pandas as pd

MODEL_DEFAULT_PATH = Path("models/spam_classifier.pkl")
FEEDBACK_CSV = Path("models/feedback.csv")


@st.cache_resource
def load_model(path: str):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    pipeline = joblib.load(model_path)
    return pipeline


def predict_batch(pipeline, texts):
    # Ensure texts is list-like
    if not isinstance(texts, (list, tuple, pd.Series)):
        texts = [texts]
    preds = pipeline.predict(texts)
    probs = None
    if hasattr(pipeline, "predict_proba"):
        try:
            probs_raw = pipeline.predict_proba(texts)
            # probability of being spam (class 1)
            if hasattr(pipeline, "classes_"):
                classes = getattr(pipeline, "classes_")
                try:
                    idx = list(classes).index(1)
                except ValueError:
                    idx = -1
                    for i, c in enumerate(classes):
                        if str(c).lower() in {"1", "spam", "s", "true", "yes", "y"}:
                            idx = i
                            break
                    if idx == -1:
                        idx = 1 if probs_raw.shape[1] > 1 else 0
                probs = [p[idx] if idx < len(p) else max(p) for p in probs_raw]
            else:
                probs = [p[1] if len(p) > 1 else 0 for p in probs_raw]
        except Exception:
            probs = None
    return preds, probs


def append_feedback(text, true_label):
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"Message": text, "TrueLabel": true_label}])
    if FEEDBACK_CSV.exists():
        df.to_csv(FEEDBACK_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(FEEDBACK_CSV, index=False)


def main():
    st.set_page_config(page_title="Spam/Ham Classifier", layout="centered")
    st.title("📨 Spam / Ham Classifier")
    st.markdown(
        "Paste one or more messages (one message per line) and click **Classify**. "
        "You can also mark a message as spam/ham and save it as feedback."
    )

    st.sidebar.header("Model")
    # Only allow specifying a path (no file upload)
    model_path = st.sidebar.text_input("Path to saved model", value=str(MODEL_DEFAULT_PATH))

    try:
        with st.spinner("Loading model..."):
            model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    else:
        st.sidebar.success(f"Loaded model from `{model_path}`")

    st.subheader("Input")
    input_text = st.text_area("Enter message(s) here (one message per line):", height=180, placeholder="Type or paste text to classify...")

    # Single visible classify button — user must click it to run predictions
    if st.button("Classify"):
        if not input_text or not input_text.strip():
            st.info("Please enter one or more messages to classify.")
        else:
            lines = [l.strip() for l in input_text.splitlines() if l.strip()]
            if len(lines) == 0:
                st.info("No non-empty lines found.")
            else:
                preds, probs = predict_batch(model, lines)
                results = []
                for text, pred, prob in zip(lines, preds, probs or [None] * len(lines)):
                    label = "Spam" if int(pred) == 1 else "Ham"
                    prob_str = f"{prob:.2%}" if prob is not None else "N/A"
                    results.append({"Message": text, "Predicted": label, "SpamProbability": prob_str})

                df_results = pd.DataFrame(results)
                st.subheader("Results")
                st.dataframe(df_results)

                # show single result details if only one message
                if len(lines) == 1:
                    st.metric(label="Prediction", value=df_results.loc[0, "Predicted"], delta=f"Spam prob: {df_results.loc[0, 'SpamProbability']}")
                    try:
                        st.write("Pipeline steps:", list(model.named_steps.keys()))
                    except Exception:
                        pass

                # Feedback area: allow user to mark any row and save
                st.markdown("---")
                st.subheader("Mark / Save feedback")
                idx = st.selectbox("Select message to mark", options=list(range(len(lines))), format_func=lambda i: lines[i][:80])
                true_label = st.radio("Mark as", options=["spam", "ham"], index=0)
                if st.button("Save feedback"):
                    append_feedback(lines[idx], true_label)
                    st.success("Feedback saved to: " + str(FEEDBACK_CSV))

    st.sidebar.markdown("---")
    st.sidebar.write("Tips:")
    st.sidebar.write("- If `predict_proba` exists it will show spam probability.")
    st.sidebar.write("- Provide the path to your saved model in the sidebar; no uploads are supported.")
    st.sidebar.write("- Feedback is appended to `models/feedback.csv` for later inspection.")

    st.markdown("---")
    st.caption("Model: CountVectorizer + MultinomialNB pipeline (trained using training script)")

if __name__ == "__main__":
    main()
