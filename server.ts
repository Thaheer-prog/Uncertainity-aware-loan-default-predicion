import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { LoanDatabase } from './server/database.ts';
import { LoanInput, UncertaintyLoanMLEngine } from './server/mlEngine.ts';

const PORT = 3000;
const PROJECT_ROOT = process.cwd();

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  const mlEngine = new UncertaintyLoanMLEngine(PROJECT_ROOT);
  mlEngine.train();

  const db = new LoanDatabase(PROJECT_ROOT);
  await db.init(mlEngine);

  // Health endpoint
  app.get('/api/health', (_req, res) => {
    res.json({
      status: 'ok',
      models_trained: mlEngine.isTrained,
      dataset_records: mlEngine.datasetSummary.total_records,
    });
  });

  // Portfolio dashboard KPIs + recent rows
  app.get('/api/dashboard', (_req, res) => {
    const stats = db.getDashboardStats();
    const history = db.getAll();
    res.json({
      stats,
      recent: history.slice(0, 8),
      all_records: history,
      dataset_summary: mlEngine.datasetSummary,
      ensemble_auc: mlEngine.metrics.find((m) => m.key === 'soft_voting_ensemble')?.roc_auc ?? 0,
    });
  });

  // Model comparison benchmarks, ROC curves, confusion matrices & feature importance
  app.get('/api/models', (_req, res) => {
    res.json({
      models: mlEngine.metrics,
      feature_importance: mlEngine.featureImportance,
      dataset_summary: mlEngine.datasetSummary,
    });
  });

  // Retrain models on dataset/loan_data.csv
  app.post('/api/models/retrain', (_req, res) => {
    mlEngine.train();
    res.json({
      message: 'Ensemble models retrained and evaluated successfully.',
      models: mlEngine.metrics,
      dataset_summary: mlEngine.datasetSummary,
    });
  });

  // Predict loan default + uncertainty estimation + persist to SQLite
  app.post('/api/predict', (req, res) => {
    try {
      const body = req.body || {};

      const parseOptNum = (val: unknown): number | null => {
        if (val === undefined || val === null || val === '') return null;
        const n = Number(val);
        return Number.isFinite(n) ? n : null;
      };

      const applicantIncome = parseOptNum(body.applicant_income);
      const loanAmount = parseOptNum(body.loan_amount);
      const loanTerm = parseOptNum(body.loan_term);
      const creditHistory = parseOptNum(body.credit_history);
      const existingObligations = parseOptNum(body.existing_obligations);

      // Validate critical numeric ranges if supplied
      const errors: string[] = [];
      if (applicantIncome !== null && applicantIncome < 0) {
        errors.push('Applicant income cannot be negative.');
      }
      if (loanAmount !== null && loanAmount <= 0) {
        errors.push('Loan amount must be greater than zero.');
      }
      if (loanTerm !== null && (loanTerm < 6 || loanTerm > 360)) {
        errors.push('Loan term must be between 6 and 360 months.');
      }
      if (creditHistory !== null && creditHistory !== 0 && creditHistory !== 1) {
        errors.push('Credit history must be 1 (Clean) or 0 (Prior Delinquency).');
      }
      if (existingObligations !== null && existingObligations < 0) {
        errors.push('Existing obligations cannot be negative.');
      }

      if (errors.length > 0) {
        return res.status(400).json({ error: errors.join(' ') });
      }

      const input: LoanInput = {
        applicant_name: typeof body.applicant_name === 'string' ? body.applicant_name.slice(0, 80) : 'Applicant',
        applicant_income: applicantIncome,
        loan_amount: loanAmount,
        loan_term: loanTerm,
        credit_history: creditHistory,
        employment_status: typeof body.employment_status === 'string' ? body.employment_status : null,
        education: typeof body.education === 'string' ? body.education : null,
        marital_status: typeof body.marital_status === 'string' ? body.marital_status : null,
        property_area: typeof body.property_area === 'string' ? body.property_area : null,
        existing_obligations: existingObligations,
      };

      const result = mlEngine.predict(input);
      const saveToHistory = body.save !== false;
      const saved = saveToHistory ? db.insertPrediction(input, result) : null;

      return res.json({
        prediction_result: result,
        record: saved,
      });
    } catch (err) {
      console.error('Prediction error:', err);
      return res.status(500).json({ error: 'Failed to process loan prediction.' });
    }
  });

  // Get full prediction history
  app.get('/api/history', (_req, res) => {
    res.json({ records: db.getAll() });
  });

  // Delete prediction record
  app.delete('/api/history/:id', (req, res) => {
    const id = Number(req.params.id);
    if (!Number.isInteger(id)) {
      return res.status(400).json({ error: 'Invalid record ID.' });
    }
    const deleted = db.deleteById(id);
    if (!deleted) {
      return res.status(404).json({ error: 'Record not found.' });
    }
    return res.json({ deleted: true, id });
  });

  // Export CSV report of all stored predictions
  app.get('/api/reports/export-csv', (_req, res) => {
    const rows = db.getAll();
    const header = [
      'Application Ref',
      'Applicant Name',
      'Annual Income',
      'Loan Amount',
      'Loan Term (Months)',
      'Credit History',
      'Employment Status',
      'Education',
      'Marital Status',
      'Property Area',
      'Existing Obligations',
      'DTI Ratio',
      'Prediction',
      'Default Probability (%)',
      'Confidence Score (%)',
      'Uncertainty Score (%)',
      '90% CI Low (%)',
      '90% CI High (%)',
      'Uncertain Flag',
      'Risk Category',
      'Created At',
    ];
    const escapeCsv = (val: unknown) => {
      const str = String(val ?? '');
      return `"${str.replace(/"/g, '""')}"`;
    };

    const lines = [header.join(',')];
    for (const r of rows) {
      lines.push(
        [
          r.application_ref,
          r.applicant_name,
          r.applicant_income,
          r.loan_amount,
          r.loan_term,
          r.credit_history === 1 ? 'Clean (1)' : 'Delinquent (0)',
          r.employment_status,
          r.education,
          r.marital_status,
          r.property_area,
          r.existing_obligations,
          (r.dti_ratio * 100).toFixed(2) + '%',
          r.prediction,
          (r.default_probability * 100).toFixed(2),
          (r.confidence_score * 100).toFixed(2),
          (r.uncertainty_score * 100).toFixed(2),
          (r.ci_low * 100).toFixed(2),
          (r.ci_high * 100).toFixed(2),
          r.is_uncertain ? 'YES' : 'NO',
          r.risk_category,
          r.created_at,
        ]
          .map(escapeCsv)
          .join(',')
      );
    }

    res.setHeader('Content-Type', 'text/csv; charset=utf-8');
    res.setHeader('Content-Disposition', 'attachment; filename="loan_default_uncertainty_report.csv"');
    res.send(lines.join('\n'));
  });

  // Download single application assessment report (HTML printable / Markdown / JSON)
  app.get('/api/reports/:id', (req, res) => {
    const id = Number(req.params.id);
    const record = Number.isInteger(id) ? db.getById(id) : db.getByRef(req.params.id);
    if (!record) {
      return res.status(404).json({ error: 'Prediction record not found.' });
    }
    const format = req.query.format || 'html';
    if (format === 'json') {
      res.setHeader('Content-Disposition', `attachment; filename="${record.application_ref}_report.json"`);
      return res.json(record);
    }

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Credit Risk & Uncertainty Assessment — ${record.application_ref}</title>
  <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; color: #0f172a; max-width: 820px; margin: 40px auto; padding: 0 24px; line-height: 1.55; }
    header { border-bottom: 2px solid #1d4ed8; padding-bottom: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: flex-end; }
    h1 { margin: 0; font-size: 22px; color: #0f172a; }
    .meta { color: #475569; font-size: 13px; }
    .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px; }
    .box { border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; background: #f8fafc; }
    .label { font-size: 12px; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; }
    .value { font-size: 20px; font-weight: 700; margin-top: 4px; color: #0f172a; }
    table { width: 100%; border-collapse: collapse; margin: 16px 0 24px; font-size: 14px; }
    th, td { border: 1px solid #e2e8f0; padding: 9px 12px; text-align: left; }
    th { background: #f1f5f9; font-weight: 600; }
    .disclaimer { margin-top: 32px; padding: 12px 16px; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; font-size: 12px; color: #1e3a8a; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Uncertainty-Aware Loan Default Risk Report</h1>
      <div class="meta">Application Ref: ${record.application_ref} · Applicant: ${record.applicant_name}</div>
    </div>
    <div class="meta">Generated: ${new Date(record.created_at).toUTCString()}</div>
  </header>

  <div class="grid">
    <div class="box">
      <div class="label">Ensemble Prediction & Risk Tier</div>
      <div class="value">${record.prediction} (${record.risk_category})</div>
    </div>
    <div class="box">
      <div class="label">Default Probability (Soft Voting)</div>
      <div class="value">${(record.default_probability * 100).toFixed(2)}%</div>
    </div>
    <div class="box">
      <div class="label">Prediction Confidence (1 - Uncertainty)</div>
      <div class="value">${(record.confidence_score * 100).toFixed(2)}%</div>
    </div>
    <div class="box">
      <div class="label">Uncertainty Score & 90% Bootstrap CI</div>
      <div class="value">${(record.uncertainty_score * 100).toFixed(2)}% [${(record.ci_low * 100).toFixed(1)}% – ${(record.ci_high * 100).toFixed(1)}%]</div>
    </div>
  </div>

  <h3>Applicant Financial Profile</h3>
  <table>
    <tr><th>Annual Income</th><td>$${record.applicant_income.toLocaleString()}</td><th>Requested Loan Amount</th><td>$${record.loan_amount.toLocaleString()}</td></tr>
    <tr><th>Loan Term</th><td>${record.loan_term} Months</td><th>Credit History</th><td>${record.credit_history === 1 ? 'Clean (1)' : 'Prior Delinquency (0)'}</td></tr>
    <tr><th>Employment Status</th><td>${record.employment_status}</td><th>Education / Marital</th><td>${record.education} / ${record.marital_status}</td></tr>
    <tr><th>Property Area</th><td>${record.property_area}</td><th>Existing Monthly Obligations</th><td>$${record.existing_obligations.toLocaleString()} (DTI: ${(record.dti_ratio * 100).toFixed(1)}%)</td></tr>
  </table>

  <h3>Uncertainty Analysis (${record.is_uncertain ? 'Uncertain — Review Recommended' : 'High Consensus'})</h3>
  <ul>
    ${record.uncertainty_reasons.map((r) => `<li>${r}</li>`).join('')}
  </ul>

  <h3>Per-Model Default Probability Breakdown</h3>
  <table>
    <tr><th>Algorithm</th><th>Predicted Default Probability</th></tr>
    <tr><td>Logistic Regression</td><td>${((record.model_breakdown.logistic_regression ?? 0) * 100).toFixed(2)}%</td></tr>
    <tr><td>Decision Tree</td><td>${((record.model_breakdown.decision_tree ?? 0) * 100).toFixed(2)}%</td></tr>
    <tr><td>Random Forest</td><td>${((record.model_breakdown.random_forest ?? 0) * 100).toFixed(2)}%</td></tr>
    <tr><td>Gradient Boosting</td><td>${((record.model_breakdown.gradient_boosting ?? 0) * 100).toFixed(2)}%</td></tr>
    <tr><td>XGBoost</td><td>${((record.model_breakdown.xgboost ?? 0) * 100).toFixed(2)}%</td></tr>
    <tr><th>Soft Voting Ensemble</th><th>${((record.model_breakdown.soft_voting_ensemble ?? 0) * 100).toFixed(2)}%</th></tr>
  </table>

  <div class="disclaimer">
    <strong>Educational Prototype Notice:</strong> This report was generated by an individual B.Tech engineering prototype for "Uncertainty-Aware Ensemble Learning for Loan Default Prediction." It demonstrates ensemble uncertainty quantification on synthetic demonstration data and must not be used for real-world lending decisions.
  </div>
</body>
</html>`;

    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.setHeader('Content-Disposition', `attachment; filename="${record.application_ref}_report.html"`);
    return res.send(html);
  });

  // Vite middleware in development, static dist in production
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(PROJECT_ROOT, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
