
import os
import pickle
 
import pandas as pd
from flask import Flask, render_template_string, request
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "logistic.pkl")
 
app = Flask(__name__)
 
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 
FEATURES = list(model.feature_names_in_)
 
# Label encodings (alphabetical, as produced by sklearn LabelEncoder).
# If your training notebook used different codes, change them here.
GENDER = {"Female": 0, "Male": 1}
MARRIED = {"No": 0, "Yes": 1}
DEPENDENTS = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED = {"No": 0, "Yes": 1}
CREDIT = {"Good (1)": 1, "None / poor (0)": 0}
PROPERTY = {"Rural": 0, "Semiurban": 1, "Urban": 2}
 
DEFAULTS = {
    "gender": "Male",
    "married": "Yes",
    "dependents": "0",
    "education": "Graduate",
    "self_employed": "No",
    "applicant_income": 5000,
    "coapplicant_income": 0,
    "loan_amount": 130,
    "loan_term": 360,
    "credit_history": "Good (1)",
    "property_area": "Urban",
}
 
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Loan Approval Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --indigo: #2a1f6b;
    --indigo-2: #43349a;
    --emerald: #10a56b;
    --emerald-soft: #def6ea;
    --rose: #e0485e;
    --rose-soft: #fde7ea;
    --gold: #f4b63f;
    --paper: #f5f4fb;
    --card: #ffffff;
    --line: #e1def2;
    --text: #1e1a3d;
    --muted: #716e8f;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Outfit", system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--paper); color: var(--text);
    min-height: 100vh; line-height: 1.5;
  }
  .shell { display: grid; grid-template-columns: 370px 1fr; min-height: 100vh; }
 
  /* ---------- side panel ---------- */
  .side {
    background: linear-gradient(160deg, var(--indigo) 0%, var(--indigo-2) 100%);
    color: #fff; padding: 56px 40px;
    display: flex; flex-direction: column; justify-content: space-between;
    position: relative; overflow: hidden;
  }
  .side::after {
    content: ""; position: absolute; right: -90px; bottom: -90px;
    width: 280px; height: 280px; border-radius: 50%;
    background: radial-gradient(circle, rgba(244,182,63,.45), rgba(244,182,63,0) 70%);
  }
  .logo { display: inline-flex; align-items: center; gap: 10px; font-weight: 700; font-size: 16px; }
  .logo i {
    width: 34px; height: 34px; border-radius: 10px; background: var(--gold); color: var(--indigo);
    display: grid; place-items: center; font-style: normal; font-weight: 800;
  }
  .side h1 { font-size: 38px; line-height: 1.12; font-weight: 800; margin: 56px 0 16px; }
  .side p { color: #cfcaf0; font-size: 15px; max-width: 290px; }
  .legend { display: grid; gap: 12px; margin-top: 34px; font-size: 14px; position: relative; z-index: 1; }
  .legend div { display: flex; align-items: center; gap: 10px; color: #e6e3fa; }
  .dot { width: 12px; height: 12px; border-radius: 50%; }
  .dot.ok { background: var(--emerald); }
  .dot.no { background: var(--rose); }
  .foot { font-size: 12px; color: #a39fd6; position: relative; z-index: 1; }
 
  /* ---------- main ---------- */
  .main { padding: 56px 64px; max-width: 940px; width: 100%; }
  .main h2 { font-size: 26px; font-weight: 700; color: var(--indigo); }
  .main .sub { color: var(--muted); margin: 4px 0 28px; font-size: 14px; }
 
  form.card {
    background: var(--card); border: 1px solid var(--line); border-radius: 20px;
    padding: 32px; box-shadow: 0 12px 32px rgba(42, 31, 107, 0.07);
  }
  .group-title { font-size: 14px; font-weight: 700; color: var(--indigo-2); margin-bottom: 14px; }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px 22px; margin-bottom: 28px; }
  .field label { display: block; font-size: 13px; font-weight: 600; color: var(--indigo); margin-bottom: 6px; }
  .field input, .field select {
    width: 100%; height: 46px; padding: 0 14px;
    border: 1.5px solid var(--line); border-radius: 12px;
    background: #faf9fe; color: var(--text); font: inherit; font-size: 15px;
    transition: border-color .15s, box-shadow .15s, background .15s;
  }
  .field input:focus, .field select:focus {
    outline: none; border-color: var(--indigo-2); background: #fff;
    box-shadow: 0 0 0 4px rgba(67, 52, 154, 0.14);
  }
  .field small { color: var(--muted); font-size: 12px; }
 
  button {
    width: 100%; height: 54px; border: 0; border-radius: 14px;
    background: linear-gradient(90deg, var(--indigo), var(--indigo-2));
    color: #fff; font: inherit; font-weight: 700; font-size: 16px; cursor: pointer;
    box-shadow: 0 10px 22px rgba(42, 31, 107, 0.3);
    transition: transform .12s;
  }
  button:hover { transform: translateY(-1px); }
  button:focus-visible { outline: 3px solid var(--gold); outline-offset: 2px; }
 
  /* ---------- result ---------- */
  .result {
    margin-top: 28px; padding: 28px 32px; border-radius: 20px; border: 1px solid;
    display: grid; grid-template-columns: auto 1fr; gap: 24px; align-items: center;
  }
  .result.ok { background: var(--emerald-soft); border-color: #a7e0c4; }
  .result.no { background: var(--rose-soft); border-color: #f4b9c1; }
  .badge {
    width: 84px; height: 84px; border-radius: 50%;
    display: grid; place-items: center; font-size: 30px; font-weight: 800; color: #fff;
  }
  .ok .badge { background: var(--emerald); }
  .no .badge { background: var(--rose); }
  .result h3 { font-size: 22px; font-weight: 800; color: var(--indigo); }
  .result p { color: var(--muted); font-size: 14px; margin-bottom: 14px; }
  .bar { height: 12px; background: #fff; border-radius: 99px; overflow: hidden; display: flex; }
  .bar span:first-child { background: var(--emerald); }
  .bar span:last-child  { background: var(--rose); }
  .bar-labels { display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-top: 8px; }
  .bar-labels .a { color: var(--emerald); }
  .bar-labels .r { color: var(--rose); }
 
  .error {
    margin-top: 20px; padding: 14px 18px; border-radius: 12px;
    background: #fff4e0; border: 1px solid #f7d79a; color: #8a5a00; font-size: 14px;
  }
 
  @media (max-width: 900px) {
    .shell { grid-template-columns: 1fr; }
    .side { padding: 32px 24px; }
    .side h1 { margin-top: 28px; font-size: 30px; }
    .legend { margin: 24px 0 0; }
    .foot { margin-top: 24px; }
    .main { padding: 32px 20px; }
  }
  @media (max-width: 560px) {
    .grid { grid-template-columns: 1fr; }
    .result { grid-template-columns: 1fr; text-align: center; justify-items: center; }
  }
</style>
</head>
<body>
<div class="shell">
 
  <aside class="side">
    <div>
      <span class="logo"><i>L</i> Loan Approval Predictor</span>
      <h1>Will the loan be approved?</h1>
      <p>Enter the applicant's details and the logistic regression model estimates the chance of approval.</p>
      <div class="legend">
        <div><span class="dot ok"></span> Likely approved</div>
        <div><span class="dot no"></span> Likely rejected</div>
      </div>
    </div>
    <div class="foot">Logistic Regression &middot; scikit-learn</div>
  </aside>
 
  <main class="main">
    <h2>Applicant details</h2>
    <p class="sub">All fields are required.</p>
 
    <form class="card" method="POST" action="/">
      <div class="group-title">Personal</div>
      <div class="grid">
        <div class="field">
          <label for="gender">Gender</label>
          <select id="gender" name="gender">
            {% for o in gender %}<option value="{{ o }}" {{ 'selected' if v.gender == o }}>{{ o }}</option>{% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="married">Married</label>
          <select id="married" name="married">
            {% for o in married %}<option value="{{ o }}" {{ 'selected' if v.married == o }}>{{ o }}</option>{% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="dependents">Dependents</label>
          <select id="dependents" name="dependents">
            {% for o in dependents %}<option value="{{ o }}" {{ 'selected' if v.dependents == o }}>{{ o }}</option>{% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="education">Education</label>
          <select id="education" name="education">
            {% for o in education %}<option value="{{ o }}" {{ 'selected' if v.education == o }}>{{ o }}</option>{% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="self_employed">Self employed</label>
          <select id="self_employed" name="self_employed">
            {% for o in self_employed %}<option value="{{ o }}" {{ 'selected' if v.self_employed == o }}>{{ o }}</option>{% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="property_area">Property area</label>
          <select id="property_area" name="property_area">
            {% for o in property_area %}<option value="{{ o }}" {{ 'selected' if v.property_area == o }}>{{ o }}</option>{% endfor %}
          </select>
        </div>
      </div>
 
      <div class="group-title">Income and loan</div>
      <div class="grid">
        <div class="field">
          <label for="applicant_income">Applicant income (monthly)</label>
          <input id="applicant_income" type="number" name="applicant_income" min="0" step="any" value="{{ v.applicant_income }}" required>
        </div>
        <div class="field">
          <label for="coapplicant_income">Co-applicant income (monthly)</label>
          <input id="coapplicant_income" type="number" name="coapplicant_income" min="0" step="any" value="{{ v.coapplicant_income }}" required>
        </div>
        <div class="field">
          <label for="loan_amount">Loan amount (in thousands)</label>
          <input id="loan_amount" type="number" name="loan_amount" min="1" step="any" value="{{ v.loan_amount }}" required>
          <small>Enter 130 for a loan of 130,000.</small>
        </div>
        <div class="field">
          <label for="loan_term">Loan term (months)</label>
          <input id="loan_term" type="number" name="loan_term" min="12" max="480" step="any" value="{{ v.loan_term }}" required>
        </div>
        <div class="field">
          <label for="credit_history">Credit history</label>
          <select id="credit_history" name="credit_history">
            {% for o in credit %}<option value="{{ o }}" {{ 'selected' if v.credit_history == o }}>{{ o }}</option>{% endfor %}
          </select>
          <small>Good means past debts were repaid on time.</small>
        </div>
      </div>
 
      <button type="submit">Check approval chance</button>
    </form>
 
    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}
 
    {% if result %}
      <section class="result {{ 'ok' if result.approved else 'no' }}">
        <div class="badge">{{ result.headline_pct }}%</div>
        <div>
          <h3>{{ 'Likely approved' if result.approved else 'Likely rejected' }}</h3>
          <p>
            {% if result.approved %}
              The model sees a {{ result.approve_pct }}% chance this loan is approved.
            {% else %}
              The model sees a {{ result.reject_pct }}% chance this loan is rejected.
            {% endif %}
          </p>
          <div class="bar">
            <span style="width: {{ result.approve_pct }}%"></span>
            <span style="width: {{ result.reject_pct }}%"></span>
          </div>
          <div class="bar-labels">
            <span class="a">Approve {{ result.approve_pct }}%</span>
            <span class="r">Reject {{ result.reject_pct }}%</span>
          </div>
        </div>
      </section>
    {% endif %}
  </main>
 
</div>
</body>
</html>
"""
 
 
def render(values, result=None, error=None):
    return render_template_string(
        PAGE,
        v=values,
        result=result,
        error=error,
        gender=GENDER.keys(),
        married=MARRIED.keys(),
        dependents=DEPENDENTS.keys(),
        education=EDUCATION.keys(),
        self_employed=SELF_EMPLOYED.keys(),
        credit=CREDIT.keys(),
        property_area=PROPERTY.keys(),
    )
 
 
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render(DEFAULTS)
 
    form = request.form
    values = {k: form.get(k, DEFAULTS[k]) for k in DEFAULTS}
 
    try:
        row = {
            "Gender": GENDER[values["gender"]],
            "Married": MARRIED[values["married"]],
            "Dependents": DEPENDENTS[values["dependents"]],
            "Education": EDUCATION[values["education"]],
            "Self_Employed": SELF_EMPLOYED[values["self_employed"]],
            "ApplicantIncome": float(values["applicant_income"]),
            "CoapplicantIncome": float(values["coapplicant_income"]),
            "LoanAmount": float(values["loan_amount"]),
            "Loan_Amount_Term": float(values["loan_term"]),
            "Credit_History": CREDIT[values["credit_history"]],
            "Property_Area": PROPERTY[values["property_area"]],
        }
        if row["ApplicantIncome"] < 0 or row["CoapplicantIncome"] < 0 or row["LoanAmount"] <= 0:
            raise ValueError
    except (KeyError, ValueError):
        return render(values, error="Please check your inputs. Every field needs a valid value.")
 
    X = pd.DataFrame([row])[FEATURES]
    proba = model.predict_proba(X)[0]
    approve_p = float(proba[list(model.classes_).index(1)])
 
    approve_pct = round(approve_p * 100, 1)
    reject_pct = round(100 - approve_pct, 1)
    approved = approve_p >= 0.5
 
    result = {
        "approved": approved,
        "approve_pct": approve_pct,
        "reject_pct": reject_pct,
        "headline_pct": round(approve_pct if approved else reject_pct),
    }
    return render(values, result=result)
 
 
if __name__ == "__main__":
    app.run(debug=True)
