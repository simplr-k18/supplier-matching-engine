# 🔍 Intelligent Supplier Matching Engine

> Open-source alternative to $500K Master Data Management tools

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![License](https://img.shields.io/badge/License-MIT-green)

## The Problem

Companies waste **100+ hours per week** manually matching duplicate supplier names across procurement systems. The same supplier appears as:

- "Apple Inc."
- "APPLE INC"
- "Apple Computer"
- "Aple Inc" (typo)

Existing solutions like SAP MDM cost **$500K/year** and still achieve only **60% accuracy**.

## The Solution

AI-powered matching engine that:

✅ **Combines 6 algorithms** (Levenshtein, Fuzzy Matching, Token Sort, Jaro-Winkler, Phonetic)  
✅ **Learns from corrections** using Machine Learning  
✅ **Achieves 80%+ accuracy** (proven in production at Fortune 500 bank)  
✅ **Saves 100+ hours/week** per organization  
✅ **Costs $0** (open source)  

---

## 🎯 Features

### 🤖 Multi-Algorithm Matching
- **Levenshtein Distance:** Character-level similarity
- **Fuzzy Matching:** Handles typos and variations
- **Token Sort/Set:** Word-order independent
- **Jaro-Winkler:** Optimized for names
- **Phonetic Matching:** Sounds-like matching

### 🧠 Machine Learning
- Learns from user accept/reject decisions
- Random Forest classifier with active learning
- Feature importance analysis
- Automatic model retraining

### 📊 Interactive UI
- Upload CSV/Excel files
- Review matches one-by-one
- Real-time analytics dashboard
- Export accepted matches

### 📈 Analytics
- Similarity score distribution
- Algorithm performance comparison
- High-confidence match identification
- Training data insights

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone repository
git clone https://github.com/simplr-k18/supplier-matching-engine.git
cd supplier-matching-engine

# Run with Docker
docker-compose up

# Open browser
http://localhost:8501
```

### Option 2: Local Installation
```bash
# Prerequisites: Python 3.11+

# Clone repository
git clone https://github.com/simplr-k18/supplier-matching-engine.git
cd supplier-matching-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python data/generate_sample_data.py

# Run application
streamlit run app.py
```

---

## 📖 Usage Guide

### 1. Upload Data

**Option A:** Load sample data (60+ suppliers with variations)  
**Option B:** Upload your own CSV/Excel with supplier names

**Expected format:**
```csv
vendor_name,country,spend_amount
Apple Inc.,USA,500000
APPLE INC,USA,300000
Microsoft Corp,USA,450000
```

### 2. Find Matches

- Adjust match threshold (50-100%)
- Enable ML predictions (if trained)
- Click "Run Matching Algorithm"
- View potential duplicates

### 3. Review Matches

- Review each match pair
- Accept ✅ or Reject ❌
- System learns from your decisions
- Track progress in real-time

### 4. Train ML Model

- After 10+ reviews, click "Train ML Model"
- Model learns which features matter most
- Future matches use ML predictions
- View feature importance in ML Insights tab

### 5. Export Results

- Download accepted matches as CSV
- Use in your ERP/procurement system
- Archive for compliance/audit

---

## 🏗️ Architecture
```
┌─────────────────┐
│  Upload CSV     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│  6 Algorithms   │ ───► │  Rule-Based      │
│  Calculate      │      │  Matching        │
│  Similarity     │      │  (Threshold)     │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         │                        ▼
         │               ┌──────────────────┐
         └──────────────►│  ML Classifier   │
                         │  (if trained)    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Present to User │
                         │  Accept/Reject   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Update Training │
                         │  Data & Retrain  │
                         └──────────────────┘
```

---

## 📊 Real-World Impact

### Case Study: Fortune 500 Bank

**Problem:**
- 10,000+ supplier records
- 2,000+ duplicates suspected
- Manual matching: 100+ hours/week
- Cost: $300K/year in labor

**Solution:**
- Implemented parent-entity-first matching logic
- Combined 5 algorithms with weighted scoring
- Added ML learning from corrections

**Results:**
- ✅ **80% match accuracy** (up from 60%)
- ✅ **100+ hours/week saved**
- ✅ **$300K/year cost savings**
- ✅ **60-day implementation**

**[Read full case study →](docs/case-study.md)**

---

## 🛠️ Tech Stack

**Backend:**
- Python 3.11
- pandas (data processing)
- scikit-learn (ML)
- FuzzyWuzzy, python-Levenshtein (string matching)
- RecordLinkage (deduplication)

**Frontend:**
- Streamlit (web framework)
- Plotly (visualizations)

**ML:**
- Random Forest Classifier
- Active Learning
- Feature Engineering

**Deployment:**
- Docker & Docker Compose
- GitHub Actions (CI/CD)
- Streamlit Cloud (free hosting)

---

## 📁 Project Structure
```
supplier-matching-engine/
├── data/
│   ├── raw/                    # Input data
│   ├── processed/              # Output matches
│   ├── training_data.csv       # ML training data
│   └── generate_sample_data.py # Sample data generator
├── models/
│   └── match_model.pkl         # Trained ML model
├── src/
│   ├── matcher.py              # Core matching algorithms
│   └── learner.py              # ML learning component
├── tests/
│   └── test_matcher.py         # Unit tests
├── docs/
│   └── case-study.md           # Real-world example
├── app.py                      # Streamlit UI
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧪 Testing
```bash
# Run unit tests
pytest tests/

# Test matching accuracy
python tests/test_matcher.py

# Load sample data and verify
python data/generate_sample_data.py
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file

---

## 👨‍💻 Author

**Rishanth Gautham Kalicheti**

- Portfolio: [your-portfolio-site.com]
- LinkedIn: [linkedin.com/in/rishanth-gautam](https://linkedin.com/in/rishanth-gautam)
- GitHub: [@simplr-k18](https://github.com/simplr-k18)
- Email: rishanthkalicheti@gmail.com

---

## 🙏 Acknowledgments

Built from real-world experience at:
- **First Citizens Bank** (formerly Silicon Valley Bank): Achieved 80% match improvement
- **Baxter International**: Unified procurement data across 4 regions

Inspired by the need for affordable, effective Master Data Management tools.

---

## 📈 Roadmap

- [ ] Add API endpoint for programmatic access
- [ ] Support for additional languages (non-English names)
- [ ] Hierarchical clustering for parent-subsidiary detection
- [ ] Integration with popular ERPs (SAP, Oracle, Coupa)
- [ ] Web scraping for company metadata enrichment
- [ ] Bulk processing mode (10K+ records)

---

## ⭐ Star History

If this project helped you, please star it on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=simplr-k18/supplier-matching-engine&type=Date)](https://star-history.com/#simplr-k18/supplier-matching-engine&Date)

---

**Built with first principles thinking and a problem-obsessed mindset.**