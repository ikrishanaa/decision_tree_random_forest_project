# Decision Trees & Random Forests Classification Project

## Objective
Implement **Decision Tree** and **Random Forest** classifiers, visualize results, compare performance, and analyze feature importances.

## Dataset
- Place your CSV dataset inside the `data/` folder.
- By default, if no dataset is provided, the script uses the **Iris dataset** from scikit-learn.

## Features
- Train/Test split
- Decision Tree training & visualization
- Random Forest training
- Feature importance interpretation
- Overfitting analysis via `max_depth`
- Cross-validation scoring

## Folder Structure
```
decision_tree_random_forest_project/
│
├── data/                 # Place dataset.csv here
├── outputs/              # Generated plots & metrics
├── src/                  # Python code
│   └── tree_models.py
├── README.md
└── requirements.txt
```

## How to Run
1. Create virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate   # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run script (default dataset):
```bash
python src/tree_models.py
```

4. Run script with your own dataset:
```bash
python src/tree_models.py --csv data/your_dataset.csv --target target_column_name
```

## Outputs
- `outputs/decision_tree.png`
- `outputs/feature_importances.png`
- `outputs/metrics.json`