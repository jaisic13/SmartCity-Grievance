import streamlit as st
import joblib
import re
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
import pandas as pd
from operator import itemgetter

# instantiate VADER
sia = SentimentIntensityAnalyzer()

# ---------------------------
# LOAD MODELS
# ---------------------------
tfidf = joblib.load("tfidf_vectorizer.pkl")
cat_model = joblib.load("category_tfidf_lr.pkl")
sev_model = joblib.load("severity_final_calibrated.pkl")

with st.sidebar:
    st.header("SmartCity Grievance — Overview")
    st.markdown("**One-liner:** Routes civic complaints to the right department and estimates severity + urgency using a hybrid ML + rule system.")
    st.markdown("**Purpose:** Fast triage of incoming complaints so municipal teams prioritize safety-critical incidents and allocate resources effectively.")

    with st.expander("Technical pipeline (concise)"):
        st.markdown("""
        **1) Input:** raw complaint text (single/multi-line)  
        **2) Preprocessing:** lowercase, remove punctuation, normalize whitespace  
        **3) Feature extraction:**  
          • TF-IDF text vectors (1–2 gram)  
          • Engineered numeric features: VADER compound & negative scores, character count, word count, negative-keyword count, duration (days)  
        **4) Category model:** TF-IDF → Logistic Regression ⇒ department label (Electricity, Water, Roads, ...)  
        **5) Severity model:** TF-IDF + numeric features → RandomForest (calibrated probabilities)  
        **6) Hybrid safety layer:** deterministic rules to escalate severity when critical keywords or long durations are detected  
        **7) Output:** Department, human-readable category, Sentiment (VADER), Severity (Low/Medium/High), Urgency score (0–1)
        """)

    with st.expander("Models & files used"):
        st.markdown("""
        • `tfidf_vectorizer.pkl` — TF-IDF transformer used to vectorize text.  
        • `category_tfidf_lr.pkl` — LogisticRegression classifier for routing.  
        • `severity_final_calibrated.pkl` — Calibrated RandomForest for severity (outputs probabilities).  
        • Note: numeric features are appended to the TF-IDF sparse matrix before severity prediction.
        """)


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

import re
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack

# instantiate VADER
sia = SentimentIntensityAnalyzer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_feats_array(text):
    """
    Produce numeric features exactly like training:
    [vader_compound, vader_neg, char_count, word_count, neg_count, duration_days]
    (order matters)
    """
    t = str(text)
    # sentiment (VADER)
    s = sia.polarity_scores(t)
    vader_compound = float(s.get("compound", 0.0))
    vader_neg = float(s.get("neg", 0.0))

    # counts
    char_count = float(len(t))
    word_count = float(len(t.split()))

    # negative-keyword count (same lexicon used in training)
    neg_words = ["not","no","broken","fire","accident","urgent","unsafe","burst","leak","bribe","injur","injury","fatal"]
    neg_count = float(sum(1 for w in neg_words if w in t.lower()))

    # duration extraction in days (match training logic)
    duration_days = 0.0
    m = re.search(r"(\d+)\s*(days|day|weeks|week|months|month|hours|hour)", t.lower())
    if m:
        v = int(m.group(1))
        unit = m.group(2)
        if "month" in unit:
            duration_days = float(v * 30)
        elif "week" in unit:
            duration_days = float(v * 7)
        elif "hour" in unit:
            # keep hours as fraction of day if you like; original model used raw number — keep consistent:
            duration_days = float(v / 24.0)
        else:
            duration_days = float(v)

    # return as numpy 2D array (1 x 6)
    return np.array([[vader_compound, vader_neg, char_count, word_count, neg_count, duration_days]], dtype=float)

def predict_severity(text):
    """
    Build exact input like training:
      - tfidf sparse vector (1 x N)
      - numeric features (1 x 6)
    then horizontally stack and call the saved classifier.
    """
    clean = preprocess(text)
    tfidf_vec = tfidf.transform([clean])           # sparse (1 x N)
    feats = extract_feats_array(text)              # numpy (1 x 6)

    # stack sparse + dense correctly
    X_full = hstack([tfidf_vec, feats])            # result is sparse

    # prediction
    probs = sev_model.predict_proba(X_full)[0]
    # map to class names -> dict
    classes = list(sev_model.classes_)
    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    pred_label = max(prob_dict, key=prob_dict.get)
    return pred_label, prob_dict


def rule_override(text, model_sev):
    critical_words = ["accident","injury","unsafe","fire","sparking","fatal","collapse","emergency"]
    long_duration = ["weeks","week","month"]

    t = text.lower()

    if any(w in t for w in critical_words):
        return "High", "critical_condition"

    if any(w in t for w in long_duration):
        return "Medium", "duration_override"

    return model_sev, "model_prediction"
# ---------------------------
# PREDICT SINGLE COMPLAINT (CATEGORY + SEVERITY + HYBRID LOGIC)
# ---------------------------

def predict_single(text):
    """Runs category classifier + severity classifier + hybrid logic.
       Returns a dict with: category, model_severity, final_severity, probabilities.
    """
    # Clean text
    clean = preprocess(text)

    # ---- CATEGORY prediction ----
    category = cat_model.predict(tfidf.transform([clean]))[0]

    # ---- SEVERITY prediction ----
    sev_pred, sev_probs = predict_severity(text)  # uses the calibrated model

    # ---- HYBRID LOGIC (Your boosting rules) ----
    text_low = text.lower()

    critical_keywords = [
        "fire", "accident", "unsafe", "electric spark", "sparking", "collapsed",
        "manhole open", "fatal", "explosion", "burst", "flooded", "flooding"
    ]
    duration_keywords = ["days", "weeks", "months"]

    # duration extraction
    duration_days = 0
    m = re.search(r"(\d+)\s*(days|day|weeks|week|months|month)", text_low)
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        if "month" in unit:
            duration_days = num * 30
        elif "week" in unit:
            duration_days = num * 7
        else:
            duration_days = num

    final_sev = sev_pred
    reason = "model"

    # Rule 1 — critical keywords → ALWAYS HIGH
    if any(k in text_low for k in critical_keywords):
        final_sev = "High"
        reason = "critical_condition"

    # Rule 2 — duration override
    elif duration_days >= 21:
        final_sev = "Medium"
        reason = "duration_override_21d"
    elif duration_days >= 10 and final_sev == "Low":
        final_sev = "Medium"
        reason = "duration_10d_boost"

    return {
        "text": text,
        "category": category,
        "model_severity": sev_pred,
        "final_severity": final_sev,
        "reason": reason,
        "probabilities": sev_probs
    }

# ---------------------------
# STREAMLIT APP UI
# ---------------------------

# small mapping for friendly department names and category detail
DEPT_FRIENDLY = {
    "Water": "Water Supply Department",
    "Electricity": "Electricity Department",
    "Sanitation": "Sanitation Department",
    "Roads": "Public Works Department",
    "Health": "Health Department",
    "Environment": "Environment Department",
    "PublicSafety": "Public Safety / Police",
    "Corruption": "Municipal / Anti-Corruption"
}

def category_detail_from_text(cat_model_label, text):
    t = text.lower()
    if "water" in cat_model_label.lower() or "pipe" in t or "burst" in t or "leak" in t:
        return "Pipe Leakage / Water Issue"
    if "electric" in cat_model_label.lower() or "light" in t or "transformer" in t or "power" in t:
        return "Power / Streetlight Issue"
    if "road" in cat_model_label.lower() or "pothole" in t or "manhole" in t:
        return "Road / Pothole"
    if "sanitation" in cat_model_label.lower() or "garbage" in t or "trash" in t:
        return "Sanitation / Waste Management"
    if "health" in cat_model_label.lower() or "clinic" in t or "ambulance" in t:
        return "Health / Medical Access"
    if "corrupt" in cat_model_label.lower() or "bribe" in t:
        return "Corruption / Service Denial"
    if "safety" in cat_model_label.lower() or "unsafe" in t or "accident" in t:
        return "Public Safety"
    return cat_model_label

def sentiment_label_from_vader(compound, neg):
    # tuned for clarity: create "Highly Negative" for strong negative
    if compound <= -0.6 or neg >= 0.6:
        return "Highly Negative"
    if compound < -0.05 or neg > 0.2:
        return "Negative"
    if compound <= 0.05:
        return "Neutral"
    if compound > 0.5:
        return "Highly Positive"
    return "Positive"

def calculate_urgency_from_features(feats):
    # feats expected as dict: {'vader_compound', 'vader_neg', 'char_count', 'word_count', 'neg_count', 'duration_days'}
    u = 0.30  # base
    # sentiment contribution
    if feats['vader_neg'] > 0.5:
        u += 0.35
    elif feats['vader_neg'] > 0.3:
        u += 0.20
    elif feats['vader_compound'] < -0.3:
        u += 0.25
    # negative word count
    if feats['neg_count'] >= 3:
        u += 0.15
    elif feats['neg_count'] >= 2:
        u += 0.10
    elif feats['neg_count'] >= 1:
        u += 0.05
    # duration
    if feats['duration_days'] > 30:
        u += 0.15
    elif feats['duration_days'] > 14:
        u += 0.10
    elif feats['duration_days'] > 7:
        u += 0.05
    # clamp
    return round(min(u, 1.0), 2)

# ---------------------------
# ENHANCED MAIN UI: Hero, KPIs, Examples, Explainability & Visuals
# Paste this below the sidebar block (replace existing main UI if needed)
# ---------------------------
import os
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO

# -- small helper for colored badge
def sev_badge_html(sev):
    color = {"HIGH":"#dc2626","MEDIUM":"#f59e0b","LOW":"#16a34a"}.get(sev.upper(), "#94a3b8")
    return f"<div style='display:inline-block;padding:6px 10px;border-radius:8px;background:{color};color:white;font-weight:700'>{sev.upper()}</div>"

# -- Hero header
st.markdown("<h1 style='margin-bottom:0.1rem'>🛰️ SmartCity Grievance Classifier</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#6b7280;margin-top:0.0rem'>Auto-route public complaints, estimate severity, and prioritize urgent issues — hybrid ML + rule-based system.</div>", unsafe_allow_html=True)
st.markdown("---")


# -- Quick examples area
st.markdown("### Quick examples")
examples = [
    "Water pipeline burst near my house. Whole area flooded.",
    "Street lights are not working for 10 days, area unsafe at night",
    "Huge pothole on the main road for 10 days causing accidents",
    "Garbage not collected for 3 weeks, bad smell everywhere",
    "Transformer sparking near the market, risk of fire"
]
ex_col1, ex_col2 = st.columns([3,1])
with ex_col1:
    for ex in examples:
        if st.button(f"Use: {ex[:40]}...", key=ex):
            # set the text area content via JS (Streamlit doesn't provide direct setter), we copy to clipboard and show instruction
            # fallback: prefill by re-rendering the input box below (we'll store in session_state)
            st.session_state['prefill'] = ex

with ex_col2:
    st.markdown("**Demo actions**")
    if st.button("Clear input"):
        st.session_state['prefill'] = ""
    if st.button("Fill with safety example"):
        st.session_state['prefill'] = examples[0]

# If a prefill exists, set the text_input value prior to the main input
prefill = st.session_state.get('prefill', "")

# -- Input area (single/multiple lines)
text_input = st.text_area(
    "Complaint(s) — one per line",
    value=prefill,
    height=180,
    placeholder="Enter complaint(s) here..."
)

analyze = st.button("Analyze", key="analyze_main")

# -- area for results and explainability
if analyze:
    complaints = [line.strip() for line in text_input.splitlines() if line.strip()]
    if not complaints:
        st.warning("Please enter at least one complaint.")
    else:
        results = []
        for c in complaints:
            pred = predict_single(c)
            # compute features for urgency
            s = sia.polarity_scores(c)
            feats = {
                'vader_compound': float(s.get('compound',0.0)),
                'vader_neg': float(s.get('neg',0.0)),
                'char_count': float(len(c)),
                'word_count': float(len(c.split())),
                'neg_count': float(sum(1 for w in ["not","no","broken","fire","accident","urgent","unsafe","burst","leak","bribe","injur","injury","fatal","smell"] if w in c.lower())),
                'duration_days': (lambda m: (int(m.group(1))*30 if "month" in m.group(2) else (int(m.group(1))*7 if "week" in m.group(2) else float(int(m.group(1))))) if (m:=re.search(r'(\d+)\s*(days|day|weeks|week|months|month|hours|hour)', c.lower())) else 0.0)(None)
            }
            urgency = calculate_urgency_from_features(feats)
            dept = DEPT_FRIENDLY.get(pred['category'], pred['category'])
            cat_detail = category_detail_from_text(pred['category'], c)
            sentiment = sentiment_label_from_vader(feats['vader_compound'], feats['vader_neg'])
            severity = pred.get('final_severity', pred.get('model_severity','Low')).upper()
            results.append({
                'complaint': c,
                'dept': dept,
                'category': cat_detail,
                'sentiment': sentiment,
                'severity': severity,
                'urgency': urgency,
                'raw': pred,
                'feats': feats
            })

        # Sort by urgency
        results_sorted = sorted(results, key=lambda x: x['urgency'], reverse=True)

        # Show top summary card
        top = results_sorted[0]
        st.markdown("### Top priority complaint")
        col1, col2 = st.columns([3,1])
        with col1:
            st.write(top['complaint'])
            st.markdown(f"**Department:** {top['dept']}")
            st.markdown(f"**Category:** {top['category']}")
            st.markdown(f"**Sentiment:** {top['sentiment']}")
        with col2:
            st.markdown("**Severity**")
            st.markdown(sev_badge_html(top['severity']), unsafe_allow_html=True)
            st.markdown(f"**Urgency:** {top['urgency']:.2f}")
            st.progress(int(top['urgency']*100))

        st.markdown("---")

        # Comparison table
        st.subheader("Comparison by urgency")
        dfc = pd.DataFrame([{
            'Complaint': r['complaint'][:120] + ('...' if len(r['complaint'])>120 else ''),
            'Department': r['dept'],
            'Category': r['category'],
            'Sentiment': r['sentiment'],
            'Severity': r['severity'],
            'Urgency': r['urgency']
        } for r in results_sorted])
        st.dataframe(dfc, height=200)

        st.markdown("---")
        st.subheader("Explainability ")
        for i, r in enumerate(results_sorted, 1):
            st.markdown(f"**{i}. Complaint:** {r['complaint']}")
            cols = st.columns([3,2])
            with cols[0]:
                st.markdown(f"- **Department:** {r['dept']}")
                st.markdown(f"- **Category:** {r['category']}")
                st.markdown(f"- **Sentiment:** {r['sentiment']}")
                st.markdown(f"- **Severity :** {r['severity']}")
            with cols[1]:
                # highlight trigger words
                triggers = [w for w in ["accident","unsafe","fire","burst","flood","sparking","leak","smell"] if w in r['complaint'].lower()]
                st.markdown("**Trigger words:** " + (", ".join(triggers) if triggers else "None"))

            st.markdown("---")

    # split complaints by line, remove empty lines
    complaints = [line.strip() for line in text_input.splitlines() if line.strip()]

    if not complaints:
        st.warning("Please enter at least one complaint.")
        st.stop()

    results = []

    for c in complaints:
        pred = predict_single(c)

        # numeric features for urgency
        s = sia.polarity_scores(c)
        vader_comp = float(s.get('compound', 0.0))
        vader_neg = float(s.get('neg', 0.0))

        char_count = float(len(c))
        word_count = float(len(c.split()))

        neg_words = [
            "not","no","broken","fire","accident","urgent","unsafe","burst",
            "leak","bribe","injur","injury","fatal","smell"
        ]
        neg_count = float(sum(1 for w in neg_words if w in c.lower()))

        duration_days = 0.0
        m = re.search(r'(\d+)\s*(days|day|weeks|week|months|month|hours|hour)', c.lower())
        if m:
            v = int(m.group(1))
            unit = m.group(2)
            if "month" in unit:
                duration_days = float(v * 30)
            elif "week" in unit:
                duration_days = float(v * 7)
            elif "hour" in unit:
                duration_days = float(v / 24.0)
            else:
                duration_days = float(v)

        feats = {
            'vader_compound': vader_comp,
            'vader_neg': vader_neg,
            'char_count': char_count,
            'word_count': word_count,
            'neg_count': neg_count,
            'duration_days': duration_days
        }

        urgency = calculate_urgency_from_features(feats)

        # friendly naming
        dept = DEPT_FRIENDLY.get(pred['category'], pred['category'])
        cat_detail = category_detail_from_text(pred['category'], c)
        sentiment = sentiment_label_from_vader(vader_comp, vader_neg)
        severity = pred.get('final_severity', pred.get('model_severity', 'Low')).upper()

        results.append({
            'complaint': c,
            'department': dept,
            'category_detail': cat_detail,
            'sentiment': sentiment,
            'severity': severity,
            'urgency_score': urgency,
            'raw': pred
        })

    