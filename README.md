# Uncertainty-Aware Ensemble Learning for Loan Default Prediction

A full-stack web application that predicts loan default risk using an ensemble of machine learning models and quantifies **how confident** the prediction is. Instead of returning only a "default / no default" label, it reports default probability, an uncertainty score, a 90% bootstrap confidence interval, and a flag for cases that need human review.

> **Educational prototype.** Built as a B.Tech engineering project using synthetic demonstration data. It must not be used for real-world lending decisions.

---

## Features

- **Ensemble prediction**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, combined with a Soft Voting Ensemble
- **Uncertainty quantification**: uncertainty score, confidence score, and 90% bootstrap confidence interval
- **Uncertain-case flagging**: human-readable reasons when the models disagree
- **Portfolio dashboard**: KPIs, recent predictions, and dataset summary
- **Model comparison**: ROC-AUC, ROC curves, confusion matrices, and feature importance
- **Retraining**: retrain all models on `dataset/loan_data.csv` from the API
- **Prediction history**: stored in SQLite, with delete support
- **Reports**: single-application report (HTML / JSON) and CSV export of all predictions
- **Input validation**: sanity checks on income, loan amount, term, credit history, and obligations

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React, Vite, Tailwind CSS |
| Backend | Node.js, Express, TypeScript |
| ML Engine | Custom TypeScript ensemble (`server/mlEngine.ts`) |
| Database | SQLite (`server/database.ts`) |
| Package manager | Bun / npm |

## Project Structure

```
.
├── server.ts            # Express server + API routes (serves Vite in dev)
├── server/
│   ├── mlEngine.ts      # Ensemble models + uncertainty estimation
│   └── database.ts      # SQLite persistence layer
├── dataset/
│   └── loan_data.csv    # Training dataset
├── src/                 # React frontend
├── vite.config.ts
├── tsconfig.json
└── .env.example
```

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) 18+ (or [Bun](https://bun.sh/))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
npm install        # or: bun install

# 3. Set up environment variables
cp .env.example .env.local
# Edit .env.local and add any required keys (e.g. GEMINI_API_KEY)

# 4. Run the app
npm run dev        # or: bun run dev
```

The app runs at **http://localhost:3000**.

### Production

```bash
npm run build
NODE_ENV=production npm start
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Server status and model training state |
| GET | `/api/dashboard` | Portfolio KPIs, recent records, dataset summary |
| GET | `/api/models` | Model metrics, ROC curves, feature importance |
| POST | `/api/models/retrain` | Retrain all models on the dataset |
| POST | `/api/predict` | Predict default risk with uncertainty |
| GET | `/api/history` | All stored predictions |
| DELETE | `/api/history/:id` | Delete a prediction record |
| GET | `/api/reports/export-csv` | Export all predictions as CSV |
| GET | `/api/reports/:id?format=html\|json` | Download a single assessment report |

### Example: `POST /api/predict`

```json
{
  "applicant_name": "Jane Doe",
  "applicant_income": 60000,
  "loan_amount": 150000,
  "loan_term": 120,
  "credit_history": 1,
  "employment_status": "Salaried",
  "education": "Graduate",
  "marital_status": "Married",
  "property_area": "Urban",
  "existing_obligations": 500,
  "save": true
}
```

**Validation rules:** income ≥ 0, loan amount > 0, loan term between 6 and 360 months, credit history `1` (clean) or `0` (prior delinquency), obligations ≥ 0.

## How Uncertainty Is Estimated

1. Each base model outputs its own default probability.
2. The **soft voting ensemble** averages these probabilities.
3. Disagreement between models and bootstrap resampling produce an **uncertainty score** and a **90% confidence interval**.
4. Predictions with high uncertainty are flagged as **uncertain** and include reasons for manual review.

## Disclaimer

This project is for educational purposes only. The data is synthetic, and the model outputs must not be used for actual credit or lending decisions.
