# 🚀 Quick Test Running Guide

## Organized Test Structure

Your tests are now organized into logical folders:

```
tests/
├── 📂 e2e/          - End-to-end tests (complete workflows)
├── 📂 integration/  - API integration tests
├── 📂 unit/         - Individual component tests
├── 📂 performance/  - Performance monitoring
├── 📂 analysis/     - Code analysis tools
├── 📂 scripts/      - Test runners
├── 📂 fixtures/     - Test data files
└── 📂 reports/      - Generated test reports
```

## 🎯 Quick Commands

### Run All Tests (Recommended)

```bash
cd tests
python scripts/run_all_tests.py
```

### Run End-to-End Tests Only

```bash
cd tests
./scripts/run_e2e_tests.sh
```

### Run Specific Category

```bash
# End-to-end tests
cd tests/e2e
python test_end_to_end.py

# Integration tests
cd tests/integration
python test_api.py
python test_session_api.py

# Unit tests
cd tests/unit
python test_resume_extractor.py
```

### Run Performance Analysis

```bash
cd tests/performance
python performance_monitor.py
```

### Run Code Analysis

```bash
cd tests/analysis
python detailed_code_analysis.py
```

## 📊 Test Reports

All test reports are saved to `tests/reports/` with timestamps:

- `consolidated_test_report_YYYYMMDD_HHMMSS.json`
- `e2e_test_report_YYYYMMDD_HHMMSS.json`
- `performance_report_YYYYMMDD_HHMMSS.json`

## ⚡ Prerequisites

1. **Start the server first:**

   ```bash
   python main.py
   ```

2. **Install test dependencies:**
   ```bash
   pip install aiohttp requests
   ```

## 🎯 Success Criteria

- **🏆 90-100%**: Production ready
- **✅ 75-89%**: Good, minor issues
- **⚠️ 50-74%**: Needs attention
- **❌ <50%**: Major issues

---

**For detailed documentation, see `README_TESTING.md`**
