"""Student Performance Classifier - Shiny for Python app.

Loads the trained pipeline (model.pkl) and the results saved by train_model.py.
The model is NOT retrained when the app starts.
"""
import base64
import functools
import io
import json
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import App, reactive, render, ui

import charts

ROOT = Path(__file__).parent

#Developer's info
DEVELOPERS = ["Librea, Marrah Princess C."]
COURSE_SECTION = "BSCS-DS 3"
UNIVERSITY = "University of Southeastern Philippines"
ACADEMIC_YEAR = "2026-2027"
# =====================================================================

model = joblib.load(ROOT / "model.pkl")
M = json.loads((ROOT / "model_metrics.json").read_text())
IMP = pd.read_csv(ROOT / "feature_importance.csv")
RAW = pd.read_csv(ROOT / "data" / "augmented_balanced_dataset_m.csv")
DF = RAW.drop_duplicates().reset_index(drop=True)  # the 2,720 unique records

FEATURES = M["features"]
L = charts.LABELS
T = M["test"]
D = M["dataset"]
CORR = pd.DataFrame(M["correlation"]).loc[FEATURES, FEATURES]
STATS = {s["feature"]: s for s in M["feature_stats"]}
IMP_ORDER = list(IMP.sort_values("perm_mean", ascending=False)["feature"])
IMP_MAP = IMP.set_index("feature")
DROP = {d["feature"]: d["cv_accuracy_without"] for d in M["ablation"]["drop_one"]}
BANDS = {b["band"]: b for b in M["confidence_bands"]}

IDS = {"Submission.Actions": "submission", "Quiz.Actions": "quiz", "Forum.Actions": "forum",
       "Create.Actions": "create", "Read.Actions": "read", "Update.Actions": "update",
       "Delete.Actions": "delete", "Days.Logged.In": "days"}
MEDIAN_ALL = DF[FEATURES].median()
MEDIAN_HIGH = DF[DF.rating == "High"][FEATURES].median()
MEDIAN_LOW = DF[DF.rating == "Low"][FEATURES].median()
SORTED = {f: np.sort(DF[f].values) for f in FEATURES}


def fig_html(fig, alt):
    """Render a matplotlib figure as an inline PNG so its size/fonts are exactly as designed."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return ui.HTML(f'<img src="{src}" alt="{alt}" style="max-width:100%;height:auto;display:block;margin:0 auto">')


def plot_ui(alt):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper():
            return fig_html(fn(), alt)
        return render.ui(wrapper)
    return deco


def pctl(f, v):
    return 100.0 * np.searchsorted(SORTED[f], v, side="right") / len(SORTED[f])


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


def pval(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def table(headers, rows, num=(), hl=None):
    th = "".join(f'<th class="{"num" if i in num else ""}">{h}</th>' for i, h in enumerate(headers))
    body = ""
    for r_i, r in enumerate(rows):
        cls = ' class="hl"' if hl is not None and r_i == hl else ""
        body += f"<tr{cls}>" + "".join(
            f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
    return ui.HTML(f'<div class="tbl-wrap"><table class="t"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')


def kv(k, v):
    return ui.HTML(f'<div class="kv"><span class="k">{k}</span><span class="v">{v}</span></div>')


def callout(kind, *children):
    return ui.div(*children, class_=f"callout {kind}")


def stat(label, number, detail):
    return ui.div(ui.div(label, class_="l"), ui.div(number, class_="n"), ui.div(detail, class_="d"), class_="stat")


def icon(name):
    paths = {
        "home": '<path d="M3 11l9-8 9 8v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
        "predict": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
        "insights": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
        "about": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
    }
    return ui.HTML(f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
                   f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>')


def nav_button(key, label, active=False):
    return ui.input_action_button(f"nav_{key}", ui.TagList(icon(key), ui.span(label)),
                                  class_="nav-btn" + (" active" if active else ""))


# ---------------------------------------------------------------- sidebar
sidebar = ui.div(
    ui.div(
        ui.div(ui.HTML('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" '
                       'stroke-linecap="round" stroke-linejoin="round"><path d="M3 20h18M6 20V11M12 20V5M18 20v-6"/></svg>'),
               class_="brand-mark"),
        ui.div(ui.div("Student Performance", class_="brand-name"),
               ui.div("LMS activity classifier", class_="brand-sub")),
        class_="brand"),
    ui.div(nav_button("home", "Home", True), nav_button("predict", "Predict"),
           nav_button("insights", "Insights"), nav_button("about", "About"), class_="nav"),
    ui.div(ui.div("Test accuracy (unseen, unique records)", class_="k"),
           ui.div(pc(T["accuracy"]), class_="v"),
           ui.p("An estimate for guidance only. Not a guarantee of any student's outcome."),
           class_="side-card"),
    class_="side")

# ---------------------------------------------------------------- HOME
home = ui.div(
    ui.div(ui.h1("Student Performance Classifier"),
           ui.p("Estimate whether a student's Learning Management System (LMS) activity looks like students "
                "rated High or Low, and see what the model learned along the way."), class_="page-head"),
    ui.div(
        stat("Unique records", f"{D['unique_rows']:,}", f"{D['rows_supplied']:,} rows in the file, {D['exact_duplicate_copies']} repeats"),
        stat("Predictors", f"{D['n_predictors']}", "LMS activity counts"),
        stat("Model", M["selected_model"], f"{M['selected_params']['n_estimators']} trees"),
        stat("Test accuracy", pc(T["accuracy"]), f"95% range {pc(T['accuracy_ci95'][0], 0)} to {pc(T['accuracy_ci95'][1], 0)}"),
        class_="stat-strip"),
    ui.div(
        ui.h2("How to use this app"),
        ui.p("Predict turns eight activity numbers into a High or Low estimate. Insights explains the data and the "
             "model so the estimate is easier to judge.", class_="sub"),
        ui.div(
            ui.div(ui.div("1", class_="no"), ui.HTML("<b>Enter activity</b><span>Type a student's eight LMS counts, or start from a typical High or Low profile.</span>"), class_="step"),
            ui.div(ui.div("2", class_="no"), ui.HTML("<b>Read the estimate</b><span>See the predicted rating, how confident the model is, and what pushed it that way.</span>"), class_="step"),
            ui.div(ui.div("3", class_="no"), ui.HTML("<b>Check the context</b><span>Use Insights to see which activities matter, and when to be careful.</span>"), class_="step"),
            class_="steps"),
        ui.div(ui.input_action_button("cta_predict", "Make a prediction", class_="btn-cta p"),
               ui.input_action_button("cta_insights", "Explore insights", class_="btn-cta s"), class_="cta-row"),
        class_="panel"),
    ui.div(
        ui.h2("About the accuracy figure"),
        callout("warn", ui.HTML(
            f"<b>The honest number is {pc(T['accuracy'])}, not 90%.</b> The supplied file has {D['exact_duplicate_copies']} exact repeated rows "
            f"({D['duplicate_copies_by_class']['Low']} of them Low), which largely accounts for its 50/50 balance. "
            f"When repeats are split across training and testing, the same model scores {pc(M['raw_split_comparison']['test_accuracy'])}, "
            f"because {M['raw_split_comparison']['test_rows_with_exact_copy_in_train']} of {M['raw_split_comparison']['test_rows']} test rows have an identical copy in the training set. "
            "This app removes repeats before splitting, so the test result reflects records the model has never seen.")),
        ui.p(f"For context, always guessing the larger class (High) would be right about {pc(D['majority_baseline_train'], 0)} of the time on the unique data."),
        class_="panel"),
    class_="page")

# ---------------------------------------------------------------- PREDICT
def field(f):
    i = IDS[f]
    return ui.div(
        ui.input_numeric(f"in_{i}", L[f], value=int(MEDIAN_ALL[f]), min=0, step=1),
        ui.div(f"Typical: {MEDIAN_ALL[f]:g}", class_="hint"), class_="field")


predict_page = ui.div(
    ui.div(ui.h1("Predict"), ui.p("Enter a student's LMS activity counts. The saved model estimates a High or Low rating."), class_="page-head"),
    ui.div(
        ui.div(
            ui.h2("Student LMS activity"),
            ui.p("Whole numbers, 0 or more.", class_="sub"),
            ui.div("Quick start", class_="preset-lab"),
            ui.div(ui.input_action_button("preset_high", "Typical High values", class_="btn-ghost"),
                   ui.input_action_button("preset_low", "Typical Low values", class_="btn-ghost"),
                   ui.input_action_button("preset_reset", "Reset", class_="btn-ghost"), class_="preset-row"),
            ui.div(*[field(f) for f in FEATURES], class_="field-grid"),
            ui.input_action_button("predict", "Predict rating", class_="btn-primary-x"),
            class_="panel"),
        ui.div(
            ui.output_ui("result_panel"),
            ui.output_ui("result_charts"),
        ),
        class_="grid-predict"),
    ui.div(
        ui.h2("The model behind the prediction"),
        ui.p("A single saved model is loaded when the app opens. It is not retrained.", class_="sub"),
        ui.div(
            ui.div(
                kv("Algorithm", M["selected_model"]),
                kv("Trees", f"{M['selected_params']['n_estimators']}"),
                kv("Trained on", f"{D['train_rows']:,} unique records (80%)"),
                kv("Tested on", f"{D['test_rows']:,} unseen records (20%)"),
                kv("Labels predicted", "High and Low, as in the dataset"),
            ),
            ui.div(
                ui.div(ui.div(ui.div("Accuracy", class_="k"), ui.div(pc(T["accuracy"]), class_="v"), class_="mini"),
                       ui.div(ui.div("High recall", class_="k"), ui.div(pc(T["report"]["High"]["recall"]), class_="v"), class_="mini"),
                       ui.div(ui.div("Low recall", class_="k"), ui.div(pc(T["report"]["Low"]["recall"]), class_="v"), class_="mini"),
                       class_="mini-stats"),
                ui.p("Recall is the share of real High (or Low) students the model correctly found. Low is harder for this model.",
                     class_="sub", style="margin-top:12px"),
            ),
            class_="grid-2"),
        class_="panel"),
    class_="page")

# ---------------------------------------------------------------- INSIGHTS
sel_choices = {f: L[f] for f in IMP_ORDER}
insights_page = ui.div(
    ui.div(ui.h1("Insights"), ui.p("What the data shows, what the model learned, and how far to trust it."), class_="page-head"),

    ui.div(ui.span("Key findings"), ui.span("start here", class_="pill"), class_="section-title"),
    ui.output_ui("findings"),

    ui.div(ui.span("A. Dataset overview"), class_="section-title"),
    ui.div(
        stat("Rows in file", f"{D['rows_supplied']:,}", "as supplied"),
        stat("Repeated rows", f"{D['exact_duplicate_copies']}", f"{pc(D['exact_duplicate_copies'] / D['rows_supplied'], 0)} of the file"),
        stat("Unique records", f"{D['unique_rows']:,}", "used for training and testing"),
        stat("High share (unique)", pc(D['class_counts_unique']['High'] / D['unique_rows'], 0), f"{D['class_counts_unique']['High']:,} High, {D['class_counts_unique']['Low']:,} Low"),
        class_="stat-strip"),
    ui.div(
        ui.div(ui.h2("Was the file really balanced?"),
               ui.p("The file shows 1,801 High and 1,801 Low. Most of the repeated rows are Low, so the balance comes largely from copying Low records.", class_="sub"),
               ui.output_ui("plot_balance"), class_="panel"),
        ui.div(ui.h2("Why this matters"),
               ui.HTML("<p>If a record and its exact copy land on opposite sides of the train/test split, the model is tested on something it already saw. That inflates accuracy.</p>"),
               ui.HTML(f"<p>There are no original-record IDs, so only exact copies can be detected. Near-copies (rows that were slightly altered) cannot be confirmed and may remain. "
                       f"{D['conflicting_label_groups']} sets of identical inputs also carry both labels, which caps the accuracy any model can reach.</p>"),
               ui.output_ui("desc_table"), class_="panel"),
        class_="grid-2"),

    ui.div(ui.span("B. Student activity analysis"), class_="section-title"),
    ui.div(
        ui.h2("How does each activity differ between High and Low?"),
        ui.p("Pick an activity. Counts use a log scale because a few students have very large values.", class_="sub"),
        ui.input_select("feat_pick", None, sel_choices, selected=IMP_ORDER[0]),
        ui.output_ui("plot_dist"),
        class_="panel"),
    ui.div(
        ui.div(ui.h2("Side-by-side comparison"),
               ui.p("The last column is the chance that a random High student has more of the activity than a random Low student. 0.50 means no difference.", class_="sub"),
               ui.output_ui("compare_table"),
               ui.p("With 2,720 records, even small differences get tiny p-values. Judge by the size of the gap, not by the p-value alone.", class_="sub", style="margin-top:10px"),
               class_="panel"),
        ui.div(ui.h2("Which activities move together?"),
               ui.p("Spearman correlation between activities.", class_="sub"),
               ui.output_ui("plot_heat"),
               ui.output_ui("heat_note"), class_="panel"),
        class_="grid-2"),

    ui.div(ui.span("C. Model performance"), class_="section-title"),
    ui.div(
        stat("Accuracy", pc(T["accuracy"]), f"95% range {pc(T['accuracy_ci95'][0])} to {pc(T['accuracy_ci95'][1])}"),
        stat("Balanced accuracy", pc(T["balanced_accuracy"]), "average of High and Low recall"),
        stat("Macro F1", pc(T["macro_f1"]), "average of High and Low F1"),
        stat("ROC AUC", f"{T['roc_auc']:.2f}", "ranking quality (1.0 is perfect)"),
        class_="stat-strip"),
    ui.div(
        ui.div(ui.h2("Confusion matrix"),
               ui.p(f"Out of {D['test_rows']} unseen students.", class_="sub"),
               ui.output_ui("plot_cm"),
               ui.output_ui("cm_note"), class_="panel"),
        ui.div(ui.h2("Per-class results"),
               ui.output_ui("report_table"),
               ui.h3("Models compared (5-fold cross-validation on training data)"),
               ui.output_ui("comparison_table"), class_="panel"),
        class_="grid-2"),
    ui.div(
        ui.h2("Is the 90% target reachable honestly?"),
        ui.output_ui("target_note"), class_="panel"),

    ui.div(ui.span("D. What the model relies on"), class_="section-title"),
    ui.div(
        ui.div(ui.h2("Permutation importance (held-out test set)"),
               ui.p("How much accuracy drops when one activity's values are shuffled. Longer bar means the model leans on it more.", class_="sub"),
               ui.output_ui("plot_imp"),
               ui.p("Importance shows what the model uses, not what causes a rating. Activities that overlap with others can look less important than they are.", class_="sub"),
               class_="panel"),
        ui.div(ui.h2("Importance, two ways"),
               ui.p("A second check: retrain without one activity and see how much cross-validated accuracy changes.", class_="sub"),
               ui.output_ui("imp_table"), class_="panel"),
        class_="grid-2"),
    ui.div(
        ui.h2("How does the prediction change as an activity changes?"),
        ui.p("Average estimated chance of High when one activity is set to each value and the others stay as in the training data.", class_="sub"),
        ui.input_select("pdp_pick", None, sel_choices, selected=IMP_ORDER[0]),
        ui.output_ui("plot_pdp"),
        ui.output_ui("pdp_note"), class_="panel"),
    ui.div(ui.span("E. How far to trust the probabilities"), class_="section-title"),
    ui.div(
        ui.div(ui.h2("Calibration"),
               ui.p("If the model says 70%, is it right about 70% of the time? Closer to the dashed line is better.", class_="sub"),
               ui.output_ui("plot_cal"), class_="panel"),
        ui.div(ui.h2("Accuracy by confidence"),
               ui.p("On the unseen test records, grouped by how confident the model was.", class_="sub"),
               ui.output_ui("conf_table"), ui.output_ui("conf_note"), class_="panel"),
        class_="grid-2"),
    class_="page")

# ---------------------------------------------------------------- ABOUT
VAR_DESC = {
    "Submission.Actions": "Number of submission-related actions",
    "Quiz.Actions": "Number of quiz-related actions",
    "Forum.Actions": "Number of forum-related actions",
    "Create.Actions": "Number of create-related actions",
    "Read.Actions": "Number of read-related actions",
    "Update.Actions": "Number of update-related actions",
    "Delete.Actions": "Number of delete-related actions",
    "Days.Logged.In": "Number of days the student logged into the LMS",
}
dev_rows = "".join(
    f'<div class="kv"><span class="k">Developer {i + 1}</span><span class="v">{n}</span></div>' for i, n in enumerate(DEVELOPERS))
about_page = ui.div(
    ui.div(ui.h1("About the project"), ui.p("Student rating prediction from LMS activity (High or Low)."), class_="page-head"),
    ui.div(
        ui.div(
            ui.div(ui.h2("Purpose"),
                   ui.p("Learning management systems record how students interact with a course: submissions, quizzes, forum posts, "
                        "reading and logins. This project uses those records to classify a student's rating as High or Low so that "
                        "instructors can notice who may need support while there is still time to act."),
                   ui.p("The model predicts the original dataset labels, High and Low. No official mapping to Pass or Fail was provided, "
                        "so none is assumed."), class_="panel"),
            ui.div(ui.h2("Dataset"),
                   ui.p(f"The file augmented_balanced_dataset_m.csv has {D['rows_supplied']:,} rows, eight numeric LMS predictors and one target, rating. "
                        f"It contains {D['exact_duplicate_copies']} exact repeated rows, leaving {D['unique_rows']:,} unique records "
                        f"({D['class_counts_unique']['High']:,} High, {D['class_counts_unique']['Low']:,} Low). The source of the data was not stated in the files provided."),
                   table(["Variable", "Description"], [[f"<code>{f}</code>", VAR_DESC[f]] for f in FEATURES] +
                         [["<code>rating</code>", "Target class: High or Low"]]), class_="panel"),
            ui.div(ui.h2("Method"),
                   ui.HTML(f"""<ul class="clean">
                   <li><b>Inspection:</b> no missing or negative values; 8 predictors are whole-number counts; Read.Actions has extreme outliers (maximum {int(DF['Read.Actions'].max()):,}).</li>
                   <li><b>Cleaning:</b> exact repeated rows were removed before any splitting, to prevent copies appearing in both training and test sets.</li>
                   <li><b>Split:</b> stratified 80/20 split with random_state=42 ({D['train_rows']:,} train, {D['test_rows']:,} test). The test set was used once, at the end.</li>
                   <li><b>Model selection:</b> seven classifiers were compared with 5-fold cross-validation on the training set only; the best were tuned with randomized search.</li>
                   <li><b>Final model:</b> {M['selected_model']} ({M['selected_params']['n_estimators']} trees), saved as a scikit-learn pipeline with Joblib and loaded by this app.</li>
                   <li><b>Interpretation:</b> permutation importance on the test set, drop-one-feature checks, partial dependence, and calibration.</li></ul>"""),
                   class_="panel"),
            ui.div(ui.h2("Model results"),
                   ui.p(f"On {D['test_rows']} unseen records the model reached {pc(T['accuracy'])} accuracy (95% range {pc(T['accuracy_ci95'][0])} to {pc(T['accuracy_ci95'][1])}). "
                        f"High: precision {pc(T['report']['High']['precision'])}, recall {pc(T['report']['High']['recall'])}. "
                        f"Low: precision {pc(T['report']['Low']['precision'])}, recall {pc(T['report']['Low']['recall'])}."),
                   ui.output_ui("about_target"), class_="panel"),
            ui.div(ui.h2("Interpreting a prediction"),
                   ui.HTML("""<ul class="clean">
                   <li>A <b>High</b> prediction means the activity pattern resembles students rated High. It does not guarantee success.</li>
                   <li>A <b>Low</b> prediction means it resembles students rated Low. It does not mean the student will fail.</li>
                   <li>The percentage is the model's confidence, and it is least reliable between 50% and 70%.</li>
                   <li>Use predictions to start a conversation or offer support, never as the only basis for a decision about a student.</li></ul>"""),
                   class_="panel"),
            ui.div(ui.h2("Limitations"),
                   ui.HTML(f"""<ul class="clean">
                   <li><b>Data:</b> the file was augmented and contained {D['exact_duplicate_copies']} repeated rows. Near-copies cannot be ruled out, so real-world accuracy may be lower than {pc(T['accuracy'], 0)}.</li>
                   <li><b>What the data measures:</b> LMS counts show clicks, not learning. Students who study offline, in groups, or on paper look inactive. The rating's definition and time period are not documented in the files provided.</li>
                   <li><b>Generalization:</b> the model reflects one course population. A different course, term, or LMS setup may behave differently. Very large values (above the 99th percentile) are rare in training.</li>
                   <li><b>Interpretation:</b> importance and partial dependence describe the model, not causes. For example, more quiz actions pointing toward Low does not mean quizzes harm ratings.</li>
                   <li><b>Errors have costs:</b> a wrongly predicted Low may label a student unfairly; a wrongly predicted High may miss someone who needs help. About {pc(1 - T['accuracy'], 0)} of test predictions were wrong.</li></ul>"""),
                   class_="panel"),
            class_="col-main"),
        ui.div(
            ui.div(ui.h2("At a glance"),
                   kv("Records (unique)", f"{D['unique_rows']:,}"), kv("Predictors", f"{D['n_predictors']}"),
                   kv("Algorithm", M["selected_model"]), kv("Trees", f"{M['selected_params']['n_estimators']}"),
                   kv("Test accuracy", pc(T["accuracy"])), kv("Balanced accuracy", pc(T["balanced_accuracy"])),
                   class_="panel"),
            ui.div(ui.h2("Built with"),
                   kv("Language", "Python"), kv("Framework", "Shiny for Python"),
                   kv("Model package", "scikit-learn"), kv("Data", "pandas, NumPy"),
                   kv("Charts", "Matplotlib"), kv("Model file", "Joblib"), class_="panel"),
            ui.div(ui.h2("Developers"), ui.HTML(dev_rows),
                   kv("Course / Section", COURSE_SECTION), kv("University", UNIVERSITY), kv("Academic year", ACADEMIC_YEAR),
                   class_="panel"),
            class_="col-side"),
        class_="grid-2-wide", style="align-items:start"),
    class_="page")

# ---------------------------------------------------------------- page shell
NAV_JS = """
Shiny.addCustomMessageHandler('nav_active', function(p){
  document.querySelectorAll('.nav-btn').forEach(function(b){b.classList.toggle('active', b.id==='nav_'+p);});
  window.scrollTo({top:0});
});
"""
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap"),
        ui.tags.link(rel="stylesheet", href="styles.css"),
    ),
    ui.div(
        sidebar,
        ui.div(
            ui.navset_hidden(
                ui.nav_panel(None, home, value="home"),
                ui.nav_panel(None, predict_page, value="predict"),
                ui.nav_panel(None, insights_page, value="insights"),
                ui.nav_panel(None, about_page, value="about"),
                id="page"),
            class_="main"),
        class_="shell"),
    ui.tags.script(NAV_JS),
    title="Student Performance Classifier",
)


# ---------------------------------------------------------------- server
def server(input, output, session):
    # ---- navigation
    async def go(p):
        ui.update_navset("page", selected=p)
        await session.send_custom_message("nav_active", p)

    def wire(trigger, page):
        @reactive.effect
        @reactive.event(trigger)
        async def _():
            await go(page)

    for k in ("home", "predict", "insights", "about"):
        wire(input[f"nav_{k}"], k)
    wire(input.cta_predict, "predict")
    wire(input.cta_insights, "insights")

    # ---- presets
    def fill(series):
        for f in FEATURES:
            ui.update_numeric(f"in_{IDS[f]}", value=float(round(series[f])))

    @reactive.effect
    @reactive.event(input.preset_high)
    def _():
        fill(MEDIAN_HIGH)

    @reactive.effect
    @reactive.event(input.preset_low)
    def _():
        fill(MEDIAN_LOW)

    @reactive.effect
    @reactive.event(input.preset_reset)
    def _():
        fill(MEDIAN_ALL)

    # ---- prediction
    def read_inputs():
        vals, errs = {}, []
        for f in FEATURES:
            v = input[f"in_{IDS[f]}"]()
            if v is None or (isinstance(v, float) and math.isnan(v)):
                errs.append(f"{L[f]}: enter a number.")
            elif v < 0:
                errs.append(f"{L[f]}: cannot be negative.")
            elif v != int(v):
                errs.append(f"{L[f]}: use a whole number.")
            elif v > 10_000_000:
                errs.append(f"{L[f]}: value is too large to be realistic.")
            else:
                vals[f] = int(v)
        return vals, errs

    @reactive.calc
    @reactive.event(input.predict)
    def result():
        vals, errs = read_inputs()
        if errs:
            return {"errors": errs}
        row = pd.DataFrame([vals], columns=FEATURES)
        classes = list(model.classes_)

        def p_high(r):
            return float(model.predict_proba(r)[0][classes.index("High")])

        p = p_high(row)
        contrib = {}
        for f in FEATURES:
            alt = row.copy()
            alt[f] = MEDIAN_ALL[f]
            contrib[f] = p - p_high(alt)
        label = "High" if p >= 0.5 else "Low"
        conf = max(p, 1 - p)
        k = min(int(round((conf - 0.5) / 0.1, 6)), 4)
        lo = 50 + 10 * k
        band = BANDS.get(f"{lo}-{min(lo + 10, 100)}%")
        high_vals = [L[f] for f in FEATURES if vals[f] > M["feature_p99"][f]]
        return {"vals": vals, "p": p, "label": label, "conf": conf, "band": band,
                "contrib": contrib, "outliers": high_vals}

    @render.ui
    def result_panel():
        if input.predict() == 0:
            return ui.div(ui.div(ui.tags.b("No prediction yet"),
                                 "Set the eight values and press Predict rating."), class_="empty panel")
        r = result()
        if "errors" in r:
            return ui.div(ui.h2("Please fix the inputs"),
                          callout("err", ui.HTML("<br>".join(r["errors"]))), class_="panel")
        hi = r["label"] == "High"
        pct_h, pct_l = 100 * r["p"], 100 * (1 - r["p"])
        meaning = ("The activity pattern looks more like students rated <b>High</b> in the training data."
                   if hi else "The activity pattern looks more like students rated <b>Low</b> in the training data.")
        bits = [ui.HTML(f"<p>{meaning} It does not mean the student will pass or fail.</p>")]
        if r["band"]:
            tone = "warn" if r["conf"] < 0.7 else "info"
            txt = (f"On unseen test records, predictions with this level of confidence ({r['band']['band']}) were correct "
                   f"{pc(r['band']['accuracy'], 0)} of the time ({r['band']['n']} records).")
            if r["conf"] < 0.7:
                txt = "<b>Treat this as uncertain.</b> " + txt
            bits.append(callout(tone, ui.HTML(txt)))
        if r["outliers"]:
            bits.append(callout("warn", ui.HTML(
                "<b>Unusually high input:</b> " + ", ".join(r["outliers"]) +
                " is above 99% of the training records, so the model has little to compare against.")))
        bits.append(ui.p("This is an estimate from LMS activity only, not a guaranteed academic outcome. "
                         "It should support, not replace, human judgment.", class_="sub"))
        return ui.div(
            ui.h2("Result"),
            ui.div(ui.div(f"{r['label']} rating", class_=f"badge-res {'high' if hi else 'low'}"),
                   ui.div(ui.HTML(f"Model confidence<br><b>{pct_h if hi else pct_l:.1f}%</b>"), class_="conf"),
                   class_="result-top"),
            ui.HTML(f'<div class="prob"><div class="h" style="width:{pct_h:.1f}%"></div><div class="l" style="width:{pct_l:.1f}%"></div></div>'
                    f'<div class="prob-legend"><span>High <b>{pct_h:.1f}%</b></span><span>Low <b>{pct_l:.1f}%</b></span></div>'),
            ui.div(*bits, style="margin-top:14px"), class_="panel")

    @render.ui
    def result_charts():
        if input.predict() == 0:
            return ui.div()
        r = result()
        if "errors" in r:
            return ui.div()
        return ui.div(
            ui.div(ui.h2("Where this student sits"),
                   ui.p("Each row compares the entered value with all unique training records.", class_="sub"),
                   ui.output_ui("plot_profile"), class_="panel"),
            ui.div(ui.h2("What pushed this prediction"),
                   ui.p("Blue: this value makes High more likely than a typical value would. Amber: less likely. It is a what-if check for this one student, so it can differ from the overall pattern because activities interact.", class_="sub"),
                   ui.output_ui("plot_local"), class_="panel"))

    @plot_ui("Where the entered student sits compared with typical High and Low students")
    def plot_profile():
        r = result()
        return charts.fig_profile({f: pctl(f, r["vals"][f]) for f in FEATURES},
                                  {f: pctl(f, MEDIAN_HIGH[f]) for f in FEATURES},
                                  {f: pctl(f, MEDIAN_LOW[f]) for f in FEATURES})

    @plot_ui("Bar chart of how each activity moved the predicted chance of High")
    def plot_local():
        return charts.fig_local(result()["contrib"])

    # ---- insights: charts
    @plot_ui("Class counts in the supplied file and after removing repeats")
    def plot_balance():
        return charts.fig_class_balance(D["class_counts_supplied"], D["class_counts_unique"])

    @plot_ui("Distribution of the chosen activity for High and Low")
    def plot_dist():
        return charts.fig_feature_dist(DF, input.feat_pick())

    @plot_ui("Correlation heatmap of LMS activities")
    def plot_heat():
        return charts.fig_heatmap(CORR)

    @plot_ui("Confusion matrix on the test set")
    def plot_cm():
        return charts.fig_confusion(T["confusion_matrix"])

    @plot_ui("Permutation importance of each activity")
    def plot_imp():
        return charts.fig_importance(IMP)

    @plot_ui("Partial dependence of predicted High probability")
    def plot_pdp():
        return charts.fig_pdp(M["pdp"], input.pdp_pick())

    @plot_ui("Calibration curve")
    def plot_cal():
        return charts.fig_calibration(M["calibration_curve"])

    # ---- insights: tables and text
    @render.ui
    def desc_table():
        rows = []
        for d in M["describe"]:
            rows.append([L[d["feature"]], f"{d['mean']:,.1f}", f"{d['50%']:,.0f}", f"{d['max']:,.0f}"])
        return ui.div(ui.h3("Activity counts (unique records)"), table(["Activity", "Mean", "Median", "Max"], rows, num={1, 2, 3}))

    @render.ui
    def compare_table():
        rows = []
        for f in IMP_ORDER:
            s = STATS[f]
            rows.append([L[f], f"{s['median_high']:g}", f"{s['median_low']:g}", f"{s['p_high_gt_low']:.2f}", pval(s["mw_p_value"])])
        return table(["Activity", "Median High", "Median Low", "P(High > Low)", "p-value"], rows, num={1, 2, 3, 4})

    @render.ui
    def heat_note():
        pairs = []
        for i, a in enumerate(FEATURES):
            for b in FEATURES[i + 1:]:
                pairs.append((abs(CORR.loc[a, b]), a, b, CORR.loc[a, b]))
        pairs.sort(reverse=True)
        top = pairs[:3]
        txt = "; ".join(f"{L[a]} and {L[b]} ({v:+.2f})" for _, a, b, v in top)
        return ui.p(f"Strongest links: {txt}. Closely related activities share information, so the model may rely on one and ignore its partner.", class_="sub")

    @render.ui
    def cm_note():
        (a, b), (c, d) = T["confusion_matrix"]
        return ui.HTML(f"<p>Correct: {a + d} of {a + b + c + d}. A High student predicted Low: <b>{b}</b>. A Low student predicted High: <b>{c}</b>.</p>")

    @render.ui
    def report_table():
        rep = T["report"]
        rows = [[c, pc(rep[c]["precision"]), pc(rep[c]["recall"]), pc(rep[c]["f1-score"]), f"{int(rep[c]['support'])}"] for c in ("High", "Low")]
        gap = rep["High"]["f1-score"] - rep["Low"]["f1-score"]
        note = (f"Low is harder to predict: its F1 is {abs(gap) * 100:.0f} points lower than High's." if gap > 0.03
                else "The model performs similarly for both classes.")
        return ui.div(table(["Class", "Precision", "Recall", "F1", "Records"], rows, num={1, 2, 3, 4}),
                      ui.p("Precision: when the model says this class, how often it is right. Recall: how many real members of the class it finds.", class_="sub", style="margin-top:10px"),
                      ui.p(note))

    @render.ui
    def comparison_table():
        rows = [[c["model"], pc(c["cv_accuracy"]), pc(c["cv_balanced_accuracy"])] for c in M["model_comparison"]]
        return table(["Model", "Accuracy", "Balanced acc."], rows, num={1, 2}, hl=0)

    def target_block(lead="Not in these experiments."):
        rs = M["raw_split_comparison"]
        return [
            callout("warn", ui.HTML(
                f"<b>{lead}</b> With repeats removed, the best model reached <b>{pc(T['accuracy'])}</b> on unseen records. "
                f"Reaching 90% only happens when repeated rows leak across the split: on the file as supplied the same model scored {pc(rs['test_accuracy'])}, "
                f"but {rs['test_rows_with_exact_copy_in_train']} of its {rs['test_rows']} test rows had an exact copy in training.")),
            ui.p(f"Seven algorithms and a hyperparameter search were tried. None approached 90% on unique records "
                 f"(best cross-validated accuracy {pc(M['model_comparison'][0]['cv_accuracy'])}). Both figures are reported openly so that readers can judge."),
        ]

    @render.ui
    def target_note():
        return ui.div(*target_block())

    @render.ui
    def about_target():
        return ui.div(*target_block("Why not 90%?"))

    @render.ui
    def imp_table():
        full = M["ablation"]["full_cv_accuracy"]
        rows = []
        for f in IMP_ORDER:
            rows.append([L[f], f"{100 * IMP_MAP.loc[f, 'perm_mean']:+.1f} pts",
                         f"{100 * (DROP[f] - full):+.1f} pts"])
        return ui.div(table(["Activity", "Shuffle drop", "Retrain without"], rows, num={1, 2}),
                      ui.p(f"Shuffle drop: accuracy lost on the test set. Retrain without: change in cross-validated accuracy ({pc(full)}) after removing the activity; negative means accuracy fell.", class_="sub", style="margin-top:10px"))

    @render.ui
    def pdp_note():
        f = input.pdp_pick()
        d = M["pdp"][f]
        x, p = d["x"], d["p_high"]
        spread = max(p) - min(p)
        if spread < 0.06:
            txt = f"Changing {L[f]} barely moves the average prediction (range {100 * spread:.0f} points), so the model gives it little weight."
        else:
            txt = (f"Average chance of High moves from {pc(p[0], 0)} at {x[0]:g} to {pc(p[-1], 0)} at {x[-1]:g} "
                   f"(a {'rise' if p[-1] > p[0] else 'fall'}), but the curve is {'not always steady' if any((p[i + 1] - p[i]) * (p[-1] - p[0]) < -0.02 for i in range(len(p) - 1)) else 'fairly steady'}. "
                   "This describes the model, not what would happen to a real student who changed their behavior.")
        return ui.p(txt)

    @render.ui
    def conf_table():
        rows = [[b["band"], f"{b['n']}", pc(b["accuracy"], 0)] for b in M["confidence_bands"]]
        return table(["Model confidence", "Test records", "Actually correct"], rows, num={1, 2})

    @render.ui
    def conf_note():
        cal = M["calibration_curve"]
        gap = max(abs(a - b) for a, b in zip(cal["predicted"], cal["observed"]))
        low = [b for b in M["confidence_bands"] if b["band"] in ("50-60%", "60-70%")]
        n_low = sum(b["n"] for b in low)
        return ui.div(
            ui.p(f"The largest gap between predicted and observed in the calibration chart is {100 * gap:.0f} points, so probabilities are "
                 f"{'reasonably' if gap < 0.08 else 'only roughly'} calibrated. "
                 f"{n_low} of {D['test_rows']} test predictions ({pc(n_low / D['test_rows'], 0)}) were below 70% confidence; treat those as close to a toss-up.", style="margin-top:10px"))

    @render.ui
    def findings():
        top3 = IMP_ORDER[:3]
        weak = [L[f] for f in FEATURES if IMP_MAP.loc[f, "perm_mean"] < 0.005]
        q, fo, dy = STATS["Quiz.Actions"], STATS["Forum.Actions"], STATS["Days.Logged.In"]
        dp = M["pdp"]["Days.Logged.In"]
        full = M["ablation"]["full_cv_accuracy"]
        worst_drop = min(DROP.items(), key=lambda kv: kv[1])
        rep = T["report"]
        lowb = [b for b in M["confidence_bands"] if b["band"] in ("50-60%", "60-70%")]
        n_low = sum(b["n"] for b in lowb)
        lo_acc, hi_acc = min(b["accuracy"] for b in lowb), max(b["accuracy"] for b in lowb)
        err_share = 1 - T["accuracy"]
        items = [
            ("Which activities separate High from Low?",
             f"The model leans most on {L[top3[0]]}, {L[top3[1]]} and {L[top3[2]]}. {', '.join(weak)} add almost nothing once the others are known."),
            ("Where are the biggest raw differences?",
             f"Quiz actions: median {q['median_high']:g} for High versus {q['median_low']:g} for Low. Forum actions: {fo['median_high']:g} versus {fo['median_low']:g}. "
             "Students rated Low tend to have more of both. This does not show that quizzes or forums cause a Low rating; a student who is struggling may simply attempt or ask more."),
            ("Does logging in more mean a High rating?",
             f"Only modestly. Median days logged in are {dy['median_high']:g} (High) and {dy['median_low']:g} (Low). Inside the model, the average chance of High rises from "
             f"{pc(dp['p_high'][0], 0)} at {dp['x'][0]:g} days to {pc(dp['p_high'][-1], 0)} at {dp['x'][-1]:g} days. It helps the model most in combination with other activities."),
            ("One activity or several?",
             f"Several. Removing any single activity changes cross-validated accuracy by at most {100 * abs(worst_drop[1] - full):.1f} points ({L[worst_drop[0]]}), "
             "because related activities carry overlapping information."),
            ("Which predictions deserve caution?",
             f"Anything under 70% confidence ({n_low} of {D['test_rows']} test cases, right only about {pc(lo_acc, 0)} to {pc(hi_acc, 0)} of the time), Low predictions (F1 {pc(rep['Low']['f1-score'], 0)} vs {pc(rep['High']['f1-score'], 0)} for High), "
             "and students with extreme counts, such as very high Read.Actions."),
            ("How could a university use this?",
             f"As an early screening cue: a Low estimate with high confidence can prompt a check-in or an offer of support. With about {pc(err_share, 0)} of predictions wrong ({pc(T['accuracy'])} accurate), "
             "it should never decide a grade, label or penalty. Rating definitions and timing should be confirmed with the data owner before any real use."),
        ]
        return ui.div(*[ui.div(ui.div(str(i + 1), class_="ic"), ui.div(ui.tags.b(t), ui.p(b)), class_="finding")
                        for i, (t, b) in enumerate(items)], class_="panel")


app = App(app_ui, server, static_assets=ROOT / "assets")
