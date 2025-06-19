# Branching Strategy for Smart Form Fill API

## Overview

This repository follows a three-branch strategy to maintain code quality and ensure smooth deployments across different environments.

## Branch Structure

### 1. **main** (Default Branch)

- **Purpose**: Development and integration branch
- **Environment**: Local development and testing
- **Stability**: Active development, may contain experimental features
- **Deployment**: Not deployed to production

### 2. **test** (Testing Branch)

- **Purpose**: Testing and quality assurance
- **Environment**: Testing/staging environment
- **Stability**: Features ready for testing
- **Deployment**: Deployed to testing environment for QA

### 3. **production** (Production Branch)

- **Purpose**: Production-ready code
- **Environment**: Production environment
- **Stability**: Stable, thoroughly tested code only
- **Deployment**: Deployed to production environment

## Workflow

### Development Flow

1. **Feature Development**

   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   # Develop your feature
   git add .
   git commit -m "feat: your feature description"
   git push origin feature/your-feature-name
   ```

2. **Merge to Main**
   - Create pull request from `feature/your-feature-name` to `main`
   - Code review and approval
   - Merge to `main`

### Testing Flow

3. **Deploy to Test**
   ```bash
   git checkout test
   git pull origin test
   git merge main
   git push origin test
   ```

### Production Flow

4. **Deploy to Production**
   ```bash
   git checkout production
   git pull origin production
   git merge test  # Only merge from test after thorough testing
   git push origin production
   ```

## Branch Protection Rules (Recommended)

### Main Branch

- Require pull request reviews before merging
- Require status checks to pass before merging
- Require branches to be up to date before merging

### Test Branch

- Require pull request reviews before merging
- Require status checks to pass before merging

### Production Branch

- Require pull request reviews before merging
- Require status checks to pass before merging
- Restrict pushes to specific users/teams

## Environment Variables

Each branch should have its own environment configuration:

### Main (.env.development)

```env
ENVIRONMENT=development
DEBUG=true
DATABASE_URL=postgresql://localhost:5432/smart_form_dev
JWT_SECRET_KEY=development-secret-key
```

### Test (.env.test)

```env
ENVIRONMENT=test
DEBUG=false
DATABASE_URL=postgresql://test-server:5432/smart_form_test
JWT_SECRET_KEY=test-secret-key
```

### Production (.env.production)

```env
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql://prod-server:5432/smart_form_prod
JWT_SECRET_KEY=production-secret-key
```

## Deployment Commands

### Local Development

```bash
# Start local development server
python -m uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

### Test Environment

```bash
# Deploy to test environment (example with Docker)
docker build -t smart-form-api:test .
docker run -p 8000:8000 --env-file .env.test smart-form-api:test
```

### Production Environment

```bash
# Deploy to production (example with AWS Lambda)
# Use your preferred deployment method (AWS Lambda, Docker, etc.)
```

## Quality Assurance

### Before Merging to Test

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Code review completed
- [ ] Documentation updated

### Before Merging to Production

- [ ] All tests pass in test environment
- [ ] Performance testing completed
- [ ] Security review completed
- [ ] Stakeholder approval obtained
- [ ] Rollback plan prepared

## Hotfix Process

For critical production issues:

1. **Create hotfix branch from production**

   ```bash
   git checkout production
   git pull origin production
   git checkout -b hotfix/critical-issue-name
   ```

2. **Fix the issue and test**

   ```bash
   # Make your fix
   git add .
   git commit -m "hotfix: critical issue description"
   git push origin hotfix/critical-issue-name
   ```

3. **Merge to production and backport**

   ```bash
   # Merge to production
   git checkout production
   git merge hotfix/critical-issue-name
   git push origin production

   # Backport to test and main
   git checkout test
   git merge production
   git push origin test

   git checkout main
   git merge test
   git push origin main
   ```

## Branch Status

- ✅ **main**: Active development
- ✅ **test**: Testing environment ready
- ✅ **production**: Production environment ready

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/MarselErlan/backend_ai_popup-.git
   cd backend_ai_popup
   ```

2. **Set up local environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Start development**
   ```bash
   git checkout main
   git pull origin main
   # Start coding!
   ```

## Support

For questions about the branching strategy or deployment process, please:

- Check existing documentation
- Create an issue in the repository
- Contact the development team

---

**Last Updated**: $(date)
**Version**: 1.0.0
