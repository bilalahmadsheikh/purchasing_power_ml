# Deployment Guide

## GitHub Setup

### 1. Initialize and Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Add CI/CD pipeline with GitHub Actions"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/purchasing_power_ml.git
git push -u origin main
```

### 2. Create Development Branch

```bash
git checkout -b develop
git push -u origin develop
```

## Configure GitHub Secrets

In your GitHub repository, go to **Settings > Secrets and variables > Actions** and add:

### Required Secrets

#### Docker Registry (for pushing images to GHCR)
- `DOCKER_REGISTRY_TOKEN`: Your GitHub Token with `packages:write` permission
- Auto-uses `${{ secrets.GITHUB_TOKEN }}` by default

#### Optional Secrets (for external registries)
- `DOCKER_REGISTRY_USERNAME`: Docker Hub username
- `DOCKER_REGISTRY_TOKEN`: Docker Hub token

#### Model Registry (MLflow/DVC)
- `MODEL_REGISTRY_TOKEN`: Your MLflow or DVC credentials
- `MODEL_REGISTRY_URL`: MLflow tracking server URL

#### Notifications (optional)
- `SLACK_WEBHOOK`: Slack webhook for build notifications
- `DISCORD_WEBHOOK_URL`: Discord webhook for notifications

#### Deployment (optional)
- `DEPLOYMENT_HOST`: Production server hostname
- `DEPLOYMENT_USER`: SSH username for deployment server
- `DEPLOYMENT_KEY`: SSH private key (base64 encoded)

## Environment Configuration

### Create Environment Variables File
```bash
# Create .env file (add to .gitignore - never commit this)
echo "MODEL_DIR=./models/pppq/" >> .env
echo "DATA_DIR=./data/processed/pppq/" >> .env
echo "LOG_LEVEL=INFO" >> .env
```

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pre-commit

# Set up pre-commit hooks
pre-commit install

# Run tests locally
pytest tests/ --cov=src

# Run linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

## Workflow Triggers

### Automatic Triggers

| Workflow | Trigger | Branch |
|----------|---------|--------|
| **CI/CD** | Push or PR | main, develop |
| **Model Training** | Code changes or weekly | main, develop |
| **Integration Tests** | Push or PR | main, develop |
| **Data Validation** | Daily (2 AM UTC) | main, develop |
| **Release** | Git tag (v*.*.*)| N/A |

### Manual Triggers

1. **GitHub UI**: Go to **Actions** tab
2. **GitHub CLI**: `gh workflow run ci-cd.yml`
3. **Workflow Dispatch**: Add `workflow_dispatch` to trigger manually

## Deployment Strategy

### Staging Environment
- Triggered on push to `develop` branch
- After all tests pass
- URL: `https://staging-pppq.example.com`
- Automated deployment

### Production Environment
- Triggered on push to `main` branch
- After all tests pass
- URL: `https://pppq-api.example.com`
- **Requires manual approval**

## Monitoring Builds

1. **GitHub Actions Tab**
   - Go to your repository > Actions
   - View all workflow runs
   - Click on a workflow to see detailed logs

2. **Status Checks**
   - PR status checks must pass before merging
   - Required checks configured in branch protection rules

3. **Artifacts**
   - Download coverage reports
   - Download security scan results
   - Download model metrics

## Setting Up Branch Protection

In **Settings > Branches > Add rule**:

1. **Branch name pattern**: `main`
2. **Require status checks to pass before merging**:
   - ✓ code-quality
   - ✓ unit-tests
   - ✓ ml-tests
   - ✓ build-image
3. **Require code reviews before merging**: 1 approval
4. **Require branches to be up to date before merging**
5. **Include administrators**: ✓ (enforces rules on admins too)

## Troubleshooting

### Build Fails: "Module not found"
```bash
# Ensure all __init__.py files exist
touch src/__init__.py
touch src/models/__init__.py
touch src/api/__init__.py
touch src/data/__init__.py
git add src/**/__init__.py
git commit -m "Add missing __init__.py files"
git push
```

### Docker Build Fails
```bash
# Check Dockerfile and docker/.dockerignore
# Build locally first
docker build -t pppq-api:test -f docker/Dockerfile .
```

### Tests Timeout
- Increase sleep time in workflow
- Check for external API calls in tests
- Run tests locally: `pytest tests/ -v`

### Registry Authentication Issues
```bash
# Verify GitHub token has packages:write permission
# Test locally:
docker login ghcr.io -u USERNAME -p TOKEN
docker push ghcr.io/username/repo:latest
```

## Upgrading Dependencies

```bash
# Update requirements.txt
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt

# Commit and push
git add requirements.txt
git commit -m "Update dependencies"
git push
```

## Viewing Build Logs

1. GitHub Actions > Click workflow run
2. Click on failed job
3. Expand the step with error
4. View detailed logs

## Local Testing with `act`

```bash
# Install act
brew install act  # macOS
choco install act  # Windows
apt-get install act  # Linux

# Run workflow locally
act -j unit-tests
act -j build-image

# View available jobs
act --list
```

## Performance Optimization

- **Cache Strategy**: Uses pip cache for faster installs
- **Parallel Jobs**: Code quality, ML tests, data validation run in parallel
- **Docker Layer Caching**: Enabled for faster rebuilds
- **Artifact Size**: Keep artifacts under 500MB

## Next Steps

1. Push code to GitHub
2. Configure all required secrets
3. Create develop and main branches
4. Set up branch protection rules
5. Create first release tag: `git tag -a v1.0.0 -m "Initial release" && git push origin v1.0.0`
6. Monitor Actions tab for successful builds
7. Set up Slack/Discord notifications (optional)

For more help, see CI_CD_TROUBLESHOOTING.md
