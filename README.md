# Restaurant Sales & Operations Dashboard

Polished Streamlit dashboard for exploring restaurant sales, product performance, operational patterns, and category mix from a single combined CSV dataset.

## Tech Stack

- Streamlit
- Pandas
- Plotly

## Project Files

- `app.py` - Streamlit dashboard application
- `requirements.txt` - Minimal dependency list
- `restaurant_data.csv` - Source dataset used by the app

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The dashboard expects `restaurant_data.csv` to be in the project root.
- Dates are parsed explicitly in `DD-MM-YYYY` format.
- Exact duplicate rows are kept by default and can be removed from the sidebar.
- Excluded fields in the UI: `HSN`, `Covers`, `Variation`, `Server Name`.

## Verification

The dashboard was checked with:

- `python -m py_compile app.py`
- headless `streamlit run app.py --server.headless true`
- direct Python assertions for data shape, KPI math, filter behavior, and chart helpers