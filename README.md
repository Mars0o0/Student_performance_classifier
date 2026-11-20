# Student Performance Classifier (Shiny for Python)

Predicts whether a student's LMS activity looks like a **High** or **Low** rating, and explains the data and the model.
Labels are the dataset's own High/Low; no Pass/Fail mapping is assumed.

## Key result (from `train_model.py`, see `model_metrics.json`)
- 3,602 rows supplied, 882 exact repeats, **2,720 unique** (repeats removed before splitting).
- Random Forest (300 trees), stratified 80/20 split, `random_state=42`.
- Test accuracy on unseen unique records: **77.8%** (95% range 74.1% to 81.1%).
- The 90% target is only reached when repeated rows leak across the split (90.6% on the file as supplied). The app reports both.

## Files
| File | Purpose |
|---|---|
| `app.py` | Shiny app: Home, Predict, Insights, About. Edit developer names at the top. |
| `train_model.py` | Inspection, model comparison, tuning, evaluation, insights. Writes the files below. |
| `charts.py` | Chart functions shared by the app and the training script. |
| `model.pkl` | Saved scikit-learn pipeline (loaded by the app, never retrained). |
| `model_metrics.json`, `feature_importance.csv` | Real results shown in the app. |
| `data/`, `assets/styles.css`, `outputs/` | Dataset, stylesheet, saved PNG charts. |

## Run locally
```
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
shiny run app.py
```
Open the address printed in the terminal (usually http://127.0.0.1:8000).
Retraining (optional, takes about 10 minutes): `python train_model.py`.

`model.pkl` must be loaded with the pinned `scikit-learn==1.9.1`. Use Python 3.12 where possible.

## Deploy to Posit Connect Cloud (GitHub method)
1. Create a free GitHub account and a **public** repository (free Connect Cloud accounts need a public repo).
2. Upload everything in this folder to the repository root (`app.py` and `requirements.txt` must be at the top level).
   `model.pkl` is about 18 MB, which is fine for GitHub's 100 MB limit.
3. Sign in at https://connect.posit.cloud/ and click **Publish**. Install the GitHub App if asked.
4. Choose **Shiny**, pick your repository and branch (`main`), set **Primary file** to `app.py`.
5. If a Python version option appears, choose 3.12. Click **Publish**.
6. Watch the log while it installs. If it fails, read the first error: a missing package means add it to `requirements.txt`; a missing file means it was not committed.
7. Open the URL (`https://<id>.share.connect.posit.cloud`), run a prediction, then open Insights and About.

Official guide: https://docs.posit.co/connect-cloud/how-to/python/shiny-python.html
Free plans have limited resources, so the first load can be slow and the app may sleep when idle.

## Notes
- Insights text is generated from the saved results, but some sentences describe this training run; re-read them if you retrain.
- Developer names, course, university and academic year are placeholders in `app.py`.
