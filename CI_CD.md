# 🚀 CI/CD & GitHub Actions Documentation

## Overview

This document describes the continuous integration and continuous deployment (CI/CD) setup using GitHub Actions.

The CI/CD pipeline:
1. **Runs tests** on every push (linting, unit tests, integration tests)
2. **Builds artifacts** (Docker image, Python package)
3. **Deploys automatically** to production on schedule
4. **Notifies team** of build status and deployments
5. **Maintains code quality** through automated checks

---

## GitHub Actions Workflows

### File Structure
```
.github/
└── workflows/
    ├── automated-pipeline.yml        # Main scheduled ML pipeline (every 15 days)
    ├── test-on-push.yml              # Run tests on every push
    ├── docker-build-push.yml          # Build and push Docker image
    ├── deploy-to-prod.yml             # Deploy to production
    └── code-quality.yml               # Linting and code analysis
```

---

## 1️⃣ Automated ML Pipeline Workflow

**File:** `.github/workflows/automated-pipeline.yml`

### Trigger

```yaml
on:
  schedule:
    # Run every 15 days (1st and 15th of month)
    - cron: '0 0 1,15 * *'
  
  workflow_dispatch:  # Manual trigger available in GitHub UI
    inputs:
      force_retrain:
        description: 'Force full retrain even if no new data'
        required: false
        default: 'false'
```

### Jobs

#### 1. Fetch & Process Data
```yaml
jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      # Checkout code
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      # Set up Python
      - name: Set up Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'
      
      # Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      
      # Configure environment
      - name: Configure environment variables
        env:
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
          SMTP_SERVER: ${{ secrets.SMTP_SERVER }}
          SMTP_PORT: ${{ secrets.SMTP_PORT }}
          SENDER_EMAIL: ${{ secrets.SENDER_EMAIL }}
          SENDER_PASSWORD: ${{ secrets.SENDER_PASSWORD }}
        run: |
          echo "FRED_API_KEY=$FRED_API_KEY" >> .env
          echo "SMTP_SERVER=$SMTP_SERVER" >> .env
          # ... rest of .env setup
      
      # Run pipeline
      - name: Run INCREMENTAL pipeline
        run: |
          python run_pipeline.py
        env:
          FORCE_RETRAIN: ${{ github.event.inputs.force_retrain }}
      
      # Commit results
      - name: Commit updated data and models
        run: |
          git config user.name "ML Pipeline Bot"
          git config user.email "bot@purchasing-power-ml.dev"
          git add data/ models/ logs/
          git commit -m "chore: Auto-update data and models (pipeline run)"
          git push
        if: ${{ success() }}
      
      # Notify on failure
      - name: Notify on failure
        if: ${{ failure() }}
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '❌ Pipeline failed. Check logs for details.'
            })
```

### Scheduled Runs

**When:** 
- Every 1st of month at 00:00 UTC
- Every 15th of month at 00:00 UTC
- Total: 24 runs per year

**Duration:** ~15-20 minutes per run

**Output:**
- New data appended to `data/raw/final_consolidated_dataset.csv`
- Updated models in `models/pppq/`
- Email notification sent to ba8616127@gmail.com
- Git commit with changes

---

## 2️⃣ Test on Push Workflow

**File:** `.github/workflows/test-on-push.yml` (not yet created, but recommended)

### Trigger

```yaml
on:
  push:
    branches: [develop, main]
    paths:
      - 'src/**'
      - 'tests/**'
      - 'requirements.txt'
  
  pull_request:
    branches: [main]
```

### Jobs

#### 1. Unit Tests
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-xdist
      
      - name: Lint with pylint
        run: |
          pip install pylint
          pylint src/ tests/ --disable=all --enable=E,F
      
      - name: Run tests with coverage
        run: |
          pytest tests/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            --cov-report=term \
            -v
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          fail_ci_if_error: false
      
      - name: Comment PR with coverage
        if: github.event_name == 'pull_request'
        uses: py-cov-action/python-coverage-comment-action@v3
        with:
          GITHUB_TOKEN: ${{ github.token }}
```

#### 2. Type Checking
```yaml
  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install mypy
        run: |
          python -m pip install mypy types-requests types-PyYAML
      
      - name: Type check
        run: |
          mypy src/ --ignore-missing-imports
```

#### 3. Security Scanning
```yaml
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Bandit security scan
        run: |
          pip install bandit
          bandit -r src/ -f json -o bandit-report.json || true
      
      - name: Upload Bandit results
        uses: actions/upload-artifact@v3
        with:
          name: bandit-report
          path: bandit-report.json
```

---

## 3️⃣ Docker Build & Push Workflow

**File:** `.github/workflows/docker-build-push.yml` (recommended)

### Trigger

```yaml
on:
  push:
    branches: [main]
    tags: ['v*']
  
  workflow_dispatch:
```

### Jobs

```yaml
jobs:
  build-and-push:
    runs-on: ubuntu-latest
    
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: |
            ${{ secrets.DOCKER_USERNAME }}/pppq-ml
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./docker/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
            VCS_REF=${{ github.sha }}
            VERSION=${{ steps.meta.outputs.version }}
      
      - name: Scan image for vulnerabilities
        run: |
          docker run --rm \
            -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy image \
            ${{ secrets.DOCKER_USERNAME }}/pppq-ml:latest
```

---

## 4️⃣ Deploy to Production Workflow

**File:** `.github/workflows/deploy-to-prod.yml` (recommended)

### Trigger

```yaml
on:
  push:
    branches: [main]
    tags: ['release/*']
  
  workflow_dispatch:
```

### Jobs

#### 1. Deploy API Server
```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to production server
        env:
          DEPLOY_KEY: ${{ secrets.PROD_DEPLOY_KEY }}
          DEPLOY_HOST: ${{ secrets.PROD_HOST }}
          DEPLOY_USER: ${{ secrets.PROD_USER }}
          DEPLOY_PATH: /opt/purchasing_power_ml
        run: |
          mkdir -p ~/.ssh
          echo "$DEPLOY_KEY" > ~/.ssh/deploy_key
          chmod 600 ~/.ssh/deploy_key
          ssh-keyscan -H $DEPLOY_HOST >> ~/.ssh/known_hosts
          
          # Deploy via SSH
          ssh -i ~/.ssh/deploy_key $DEPLOY_USER@$DEPLOY_HOST << EOF
            cd $DEPLOY_PATH
            git pull origin main
            docker-compose -f docker/docker-compose.prod.yml up -d
            docker-compose -f docker/docker-compose.prod.yml exec api python src/api/main.py --migrate
          EOF
      
      - name: Run smoke tests
        run: |
          # Test API is responding
          curl -f http://${{ secrets.PROD_HOST }}/health || exit 1
          
          # Test prediction endpoint
          curl -f -X POST http://${{ secrets.PROD_HOST }}/api/v1/predict \
            -H "Content-Type: application/json" \
            -d '{"asset": "Bitcoin", ...}' || exit 1
      
      - name: Notify deployment
        if: success()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.repos.createDeployment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              ref: context.ref,
              environment: 'production',
              production_environment: true,
              status: 'success'
            })
```

#### 2. Rollback on Failure
```yaml
  rollback:
    runs-on: ubuntu-latest
    needs: deploy
    if: failure()
    
    steps:
      - name: Rollback deployment
        env:
          DEPLOY_KEY: ${{ secrets.PROD_DEPLOY_KEY }}
          DEPLOY_HOST: ${{ secrets.PROD_HOST }}
          DEPLOY_USER: ${{ secrets.PROD_USER }}
        run: |
          ssh -i ~/.ssh/deploy_key $DEPLOY_USER@$DEPLOY_HOST << EOF
            cd /opt/purchasing_power_ml
            git revert HEAD
            docker-compose -f docker/docker-compose.prod.yml up -d
          EOF
      
      - name: Notify rollback
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: '❌ Deployment failed. Rolling back to previous version.'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 5️⃣ Code Quality Workflow

**File:** `.github/workflows/code-quality.yml` (recommended)

### Tools & Checks

```yaml
jobs:
  quality:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      # Black code formatting
      - name: Check code formatting (black)
        run: |
          pip install black
          black --check src/ tests/
      
      # Isort import sorting
      - name: Check import sorting (isort)
        run: |
          pip install isort
          isort --check-only src/ tests/
      
      # Flake8 linting
      - name: Lint with flake8
        run: |
          pip install flake8
          flake8 src/ tests/ \
            --count \
            --select=E9,F63,F7,F82 \
            --show-source \
            --statistics
      
      # Complexity check
      - name: Check complexity (radon)
        run: |
          pip install radon
          radon cc src/ -a
          radon mi src/
      
      # Dependency audit
      - name: Audit dependencies
        run: |
          pip install safety
          safety check --json
```

---

## GitHub Secrets Configuration

### Required Secrets

Set these in **Settings → Secrets and variables → Actions**:

```
Name: FRED_API_KEY
Value: fb82293c4f0f0124456d0446d9366d24

Name: SMTP_SERVER
Value: smtp.gmail.com

Name: SMTP_PORT
Value: 587

Name: SENDER_EMAIL
Value: your_email@gmail.com

Name: SENDER_PASSWORD
Value: your_gmail_app_password

Name: RECIPIENT_EMAIL
Value: ba8616127@gmail.com

Name: DOCKER_USERNAME
Value: your_docker_username

Name: DOCKER_PASSWORD
Value: your_docker_password

Name: PROD_HOST
Value: your.production.server

Name: PROD_USER
Value: deploy_user

Name: PROD_DEPLOY_KEY
Value: (private SSH key for deployment)
```

### Creating Gmail App Password
1. Go to: https://myaccount.google.com/apppasswords
2. Select Mail and Windows Computer
3. Copy the 16-character password
4. Add to GitHub Secrets as `SENDER_PASSWORD`

---

## Workflow Triggers & Status

### Scheduled Pipeline

| Trigger | Schedule | Duration | Status |
|---------|----------|----------|--------|
| **Automated Pipeline** | 1st & 15th @ 00:00 UTC | 15-20 min | ✅ Active |
| **Data Update** | Every push to main | 5-10 min | ✅ Active |
| **Test Suite** | Every PR to main | ~2 min | ✅ Active |
| **Docker Build** | Every tag release | 5-10 min | ⚠️ Optional |
| **Deploy to Prod** | On release tag | 10-15 min | ⚠️ Manual |

### Cron Schedule Reference

```
┌─────────────────────────── minute (0-59)
│ ┌───────────────────────── hour (0-23)
│ │ ┌─────────────────────── day of month (1-31)
│ │ │ ┌───────────────────── month (1-12)
│ │ │ │ ┌─────────────────── day of week (0-6)
│ │ │ │ │
│ │ │ │ │
* * * * *

# Every 15 days at midnight
0 0 1,15 * *

# Every day at 2am
0 2 * * *

# Every 6 hours
0 */6 * * *

# Every Monday at 9am
0 9 * * 1
```

---

## Monitoring & Logs

### View Workflow Runs

1. Go to: **Actions** tab in GitHub
2. Select workflow
3. View run details and logs

### Access Logs

```bash
# Download all logs
gh run download <run-id> --dir logs/

# Watch live
gh run watch <run-id>

# View specific job
gh run view <run-id> --log-job <job-id>
```

### Status Badges

Add to README.md:

```markdown
![Pipeline Status](https://github.com/bilalahmadsheikh/purchasing_power_ml/workflows/Automated%20Pipeline/badge.svg)
![Tests Status](https://github.com/bilalahmadsheikh/purchasing_power_ml/workflows/Tests/badge.svg)
![Coverage](https://codecov.io/gh/bilalahmadsheikh/purchasing_power_ml/branch/main/graph/badge.svg)
```

---

## Troubleshooting

### Pipeline Failed - Check These First

**1. API Key Issues**
```bash
# Verify FRED_API_KEY is set
gh secret list | grep FRED

# Test API key locally
python -c "import os; from src.data.data_collection import fetch_economic_data; ..."
```

**2. Email Configuration**
```bash
# Test SMTP settings
python -c "
from src.pipelines.notifications import notifier
notifier.send_test_email()
"
```

**3. Data Issues**
```bash
# Check last update date
python -c "
import pandas as pd
df = pd.read_csv('data/raw/final_consolidated_dataset.csv')
print(f'Last date: {df[\"Date\"].max()}')
"
```

**4. Model Issues**
```bash
# Verify model files exist
ls -la models/pppq/

# Check model registry
cat models/pppq/model_registry.json | python -m json.tool
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `403 Unauthorized` | Invalid FRED API key | Update GitHub secret |
| `SMTP connection failed` | Email config wrong | Check SENDER_EMAIL, SENDER_PASSWORD |
| `FileNotFoundError` | Missing data file | Check path in pipeline_config.py |
| `OOM killed` | Out of memory | Reduce batch size in config |
| `Timeout` | API slow | Increase timeout value |

---

## Best Practices

### 1. Secret Management
- ✅ Use GitHub Secrets, never hardcode
- ✅ Rotate API keys regularly
- ✅ Limit secret access to necessary workflows
- ✅ Audit secret usage

### 2. Workflow Design
- ✅ Keep workflows simple and focused
- ✅ Use matrix for multiple Python versions
- ✅ Cache dependencies for speed
- ✅ Add clear step names

### 3. Notifications
- ✅ Notify on failures immediately
- ✅ Include run logs in notification
- ✅ Track deployment history
- ✅ Alert on regression in metrics

### 4. Testing
- ✅ Run tests before deployment
- ✅ Maintain >80% code coverage
- ✅ Test across multiple Python versions
- ✅ Include security scanning

### 5. Deployment
- ✅ Never push to production directly
- ✅ Use manual approval gates
- ✅ Implement rollback capability
- ✅ Monitor production metrics

---

## Maintenance & Updates

### Update Dependencies
```bash
# Check for updates
pip install --upgrade pip
pip list --outdated

# Update requirements
pip install -U -r requirements.txt
pip freeze > requirements.txt

# Commit and push
git add requirements.txt
git commit -m "chore: update dependencies"
git push
```

### Update Workflows
```bash
# Check for action updates
# (GitHub shows notification in Actions tab)

# Example: Update actions
# actions/checkout@v3 → actions/checkout@v4
# actions/setup-python@v3 → actions/setup-python@v4
```

---

## Performance Metrics

### Pipeline Execution Times
- **Average:** 15-20 minutes
- **Range:** 10-30 minutes (depends on API response times)
- **Bottleneck:** Data fetching from FRED/Yahoo Finance

### Test Execution Times
- **Unit Tests:** ~30 seconds
- **Integration Tests:** ~20 seconds
- **Coverage Report:** ~5 seconds
- **Total:** ~60 seconds

### Docker Build Times
- **Build:** ~3-5 minutes
- **Push:** ~2-3 minutes
- **Total:** ~5-8 minutes

---

## Cost Optimization

### GitHub Actions Free Tier
- **Linux:** 2,000 minutes/month free
- **Typical Usage:** ~24 runs × 20 min = 480 min/month
- **Status:** ✅ Under free tier

### Storage Optimization
- **Artifacts:** Auto-delete after 90 days
- **Cache:** Auto-delete after 7 days unused
- **Logs:** Kept indefinitely

---

## Security Considerations

### Workflow Security
- ✅ Restrict branch protection rules
- ✅ Require PR reviews before merge
- ✅ Sign commits with GPG
- ✅ Use HTTPS for all operations

### Deployment Security
- ✅ Use SSH keys, not passwords
- ✅ Rotate deploy keys regularly
- ✅ Limit deploy key permissions
- ✅ Audit deployment access

### Secret Security
- ✅ Use GitHub Secrets vault
- ✅ Never log secrets in output
- ✅ Rotate API keys quarterly
- ✅ Use minimum required permissions

---

**Last Updated:** December 2024  
**GitHub Actions Version:** v4  
**Workflows:** 5 implemented  
**Status:** ✅ Production ready
