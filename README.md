# NewsLens AI: Indian Multilingual News Classifier

Streamlit dashboard built from your experiment results (mBERT, XLM-RoBERTa, IndicBERT-v2; Hindi, Tamil, Telugu, Malayalam, Marathi).

## Folder layout
```
newslens_streamlit/
  app.py            dashboard
  inference.py      live classifier (separate module)
  requirements.txt
  data/             your 4 CSVs + best_model_info.json + label_mapping.json (already included)
  models/best_model/   <- you add this later for live predictions
```

## Run on Windows (PowerShell)
1. Install Python 3.10 or 3.11 from python.org (tick "Add Python to PATH").
2. Open PowerShell in the project folder (Shift + right-click inside it, "Open in Terminal").
3. Create and activate a virtual environment:
   ```
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If activation is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry.
4. Install packages:
   ```
   pip install -r requirements.txt
   ```
   To skip the large torch/transformers download for now, delete the last four lines of requirements.txt first; the dashboard works without them.
5. Start the app:
   ```
   streamlit run app.py
   ```
   It opens at http://localhost:8501.

## Dashboard data
The app reads `data/overall_metrics.csv`, `language_wise_metrics.csv`, `predictions.csv` and `error_analysis.csv`. Confusion matrices and per-category F1 are computed from `predictions.csv`. Nothing is hard-coded.

## Enabling the live classifier later
The page shows "not available" until `models/best_model/` contains:
- `config.json`
- `model.safetensors` (or `pytorch_model.bin`)
- tokenizer files (`tokenizer.json`/`tokenizer_config.json`, plus `spiece.model` or `vocab.txt` as applicable)

Save them from your training notebook (use the IndicBERT-v2 model, the selected best):
```python
model.save_pretrained("models/best_model")
tokenizer.save_pretrained("models/best_model")
```
Restart the app. Class names come from `data/label_mapping.json`; make sure the fine-tuned model uses the same label order (crime=0 ... technology=4).

## Upload CSV format
Required columns: `language`, `true_label`, `predicted_label`. Optional: `model`, `text`, `confidence`.
