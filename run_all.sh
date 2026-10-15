#!/usr/bin/env bash
# Runs every remaining step of the Udacity NYC Airbnb project from GitHub Codespaces:
#   GitHub releases 1.0.0 / 1.0.1, EDA notebook, full pipeline, hyperparameter sweep,
#   promotion of the best model to "prod", test-set evaluation, and the sample2 release runs.
# Usage (from the repository folder):   bash run_all.sh
set -u
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
export HYDRA_FULL_ERROR=1
REPO_URL="https://github.com/rand1999/build-ml-pipeline-for-short-term-rental-prices.git"

say()  { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31mSTOPPED: %s\nSend the lines above this message to Claude.\033[0m\n' "$*"; exit 1; }

# ---------- Python ----------
if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY=python3; fi
say "Using $($PY --version)"
$PY -m pip install -q --upgrade "wandb>=0.22.3" "mlflow==3.3.2" "hydra-core==1.3.2" \
    pandas matplotlib nbconvert ipykernel || fail "pip install failed"

# ---------- conda (MLflow builds one env per step with it) ----------
if ! command -v conda >/dev/null 2>&1; then
  if [ -x "$HOME/miniforge3/bin/conda" ]; then
    export PATH="$HOME/miniforge3/bin:$PATH"
  else
    say "Installing conda (Miniforge) - one time only, about 1 minute"
    curl -fsSL -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh" \
      || fail "could not download Miniforge"
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3" >/dev/null || fail "Miniforge install failed"
    export PATH="$HOME/miniforge3/bin:$PATH"
  fi
fi
say "conda: $(conda --version)"

# ---------- W&B login ----------
if ! $PY -c "import wandb,sys; sys.exit(0 if wandb.Api().api_key else 1)" 2>/dev/null; then
  say "Log in to W&B: paste your API key when asked (it stays hidden, do not share it)"
  $PY -m wandb login --relogin || fail "W&B login failed"
fi
ENTITY=$($PY -c "import wandb; print(wandb.Api().default_entity)")
say "W&B entity: $ENTITY  project: nyc_airbnb"
W="$ENTITY/nyc_airbnb"

# ---------- GitHub releases ----------
say "Creating GitHub releases 1.0.0 and 1.0.1"
git push origin main --tags || fail "git push failed"
for v in 1.0.0 1.0.1; do
  if gh release view "$v" >/dev/null 2>&1; then echo "release $v already exists"
  else
    notes="Pipeline release $v"; [ "$v" = "1.0.1" ] && notes="Fix: drop listings outside NYC boundaries so sample2.csv passes the data checks"
    gh release create "$v" --title "$v" --notes "$notes" \
      || echo "!! Could not create release $v automatically. Create it on GitHub: Releases > Draft a new release > tag $v"
  fi
done

# ---------- 1) download + clean ----------
say "Step: download + basic_cleaning (sample1.csv)"
$PY main.py "main.steps='download,basic_cleaning'" || fail "download/basic_cleaning failed"

say "Tagging clean_sample.csv:latest as 'reference'"
$PY - <<EOF || fail "could not add the reference tag"
import wandb
a = wandb.Api().artifact("$W/clean_sample.csv:latest")
if "reference" not in a.aliases:
    a.aliases.append("reference"); a.save()
print("reference ->", a.name)
EOF

# ---------- 2) EDA notebook ----------
say "Running EDA.ipynb top to bottom"
$PY -m jupyter nbconvert --to notebook --execute --inplace EDA.ipynb \
  || echo "!! The notebook did not run automatically; open it, Restart Kernel > Run All, then save."

# ---------- 3) checks, split, train ----------
say "Step: data_check + data_split + train_random_forest"
$PY main.py "main.steps='data_check,data_split,train_random_forest'" || fail "pipeline failed"

# ---------- 4) hyperparameter sweep ----------
say "Hyperparameter sweep (9 runs, a few minutes)"
$PY main.py -m "main.steps=train_random_forest" \
  modeling.max_tfidf_features=10,15,30 \
  modeling.random_forest.max_features=0.1,0.33,0.5 || fail "sweep failed"

# ---------- 5) promote best model ----------
say "Promoting the model with the lowest MAE to 'prod'"
$PY - <<EOF || fail "could not tag the best model"
import wandb
api = wandb.Api()
runs = [r for r in api.runs("$W") if r.job_type == "train_random_forest" and r.state == "finished"
        and r.summary.get("mae") is not None]
best = min(runs, key=lambda r: r.summary["mae"])
model = next(a for a in best.logged_artifacts() if a.type == "model_export")
if "prod" not in model.aliases:
    model.aliases.append("prod"); model.save()
print(f"best run {best.name}: mae={best.summary['mae']:.3f} r2={best.summary.get('r2'):.3f} -> {model.name}:prod")
EOF

# ---------- 6) test the prod model ----------
say "Step: test_regression_model"
$PY main.py "main.steps='test_regression_model'" || fail "test_regression_model failed"

# ---------- 7) releases on sample2 ----------
say "Release 1.0.0 on sample2.csv  (this one is EXPECTED to fail at data_check: listings outside NYC)"
if $PY -m mlflow run "$REPO_URL" -v 1.0.0 --env-manager conda \
     -P hydra_options="etl.sample='sample2.csv'"; then
  echo "!! 1.0.0 unexpectedly passed"
else
  echo ">> As expected, 1.0.0 failed on sample2 (test_proper_boundaries). Moving on."
fi

say "Release 1.0.1 on sample2.csv  (should succeed)"
$PY -m mlflow run "$REPO_URL" -v 1.0.1 --env-manager conda \
   -P hydra_options="etl.sample='sample2.csv'" || fail "release 1.0.1 failed"

# ---------- 8) save notebook outputs to GitHub ----------
say "Pushing the executed notebook to GitHub"
git add EDA.ipynb && git commit -q -m "Run EDA notebook" && git push origin main || true

say "ALL PIPELINE STEPS DONE"
APP=$($PY -c "import wandb,wandb.util as u; print(u.app_url(wandb.Api().settings['base_url']))" 2>/dev/null)
echo "W&B project link for the submission: $APP/$W"
echo "Last step by hand: open that link -> project Settings -> Visibility / Privacy: Public."
