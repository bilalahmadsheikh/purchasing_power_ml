# Streamlit Stable Deployment Configuration

**Date**: 2024-12-22
**Purpose**: Configure Streamlit to always use the last successful build with automatic rollback

---

## Overview

This guide ensures that Streamlit always deploys the last successful, tested build. If any CI/CD tests fail, the deployment automatically uses the previous stable version.

---

## How It Works

### 1. Docker Image Tagging Strategy

**Before** (Risky - deploys untested code):
```
ghcr.io/bilalahmadsheikh/purchasing_power_ml:main  ← Always tracks latest commit (even if broken)
```

**After** (Safe - deploys only tested code):
```
ghcr.io/bilalahmadsheikh/purchasing_power_ml:stable  ← Only updated when all tests pass
ghcr.io/bilalahmadsheikh/purchasing_power_ml:latest  ← Same as stable
ghcr.io/bilalahmadsheikh/purchasing_power_ml:<sha>   ← Specific commit for rollback
```

### 2. CI/CD Pipeline Flow

```
Push to main
    ↓
Run Tests & Code Quality
    ↓
    ├─ Tests PASS ──────────────────┐
    │                               ↓
    │   Build Docker (candidate)    │
    │           ↓                   │
    │   Promote to :stable          │
    │   Promote to :latest          │
    │           ↓                   │
    │   Streamlit deploys ✅         │
    │                               │
    └─ Tests FAIL ─────────────────┐│
                                   ↓↓
            Auto-Rollback to Last Successful Commit
                                   ↓
            Streamlit keeps using :stable (unaffected) ✅
```

---

## Streamlit Cloud Configuration

### Required Settings

**Repository**: `bilalahmadsheikh/purchasing_power_ml`
**Branch**: `main` (or `develop`)
**Main file**: `streamlit_app/app.py`
**Python version**: `3.10`

### Advanced Settings (IMPORTANT)

Add this to your Streamlit app configuration:

**Option 1: Use Git SHA from stable tag** (Recommended)
1. Go to Streamlit Cloud dashboard
2. Click on your app → Settings → Advanced
3. Add environment variable:
   ```
   GIT_REF=stable
   ```

**Option 2: Pin to last successful commit** (Manual)
1. Find last successful commit from [ML_EXPERIMENT_RESULTS.md](ML_EXPERIMENT_RESULTS.md)
2. Set in Streamlit Settings:
   ```
   GIT_REF=611b29b362b480a8e6a56848f453ab8a2091e683
   ```

---

## Automatic Rollback Mechanism

### Trigger Conditions

Automatic rollback occurs when:
- ✅ Tests fail (pytest failures)
- ✅ Code quality checks fail (Black, isort, flake8 errors if made non-optional)
- ✅ Data validation fails
- ✅ Model performance degradation detected
- ✅ Integration test failures

### What Happens

1. **CI/CD detects failure** in tests or code-quality jobs
2. **Rollback job triggers** (only on main branch)
3. **Finds last successful commit** from ML_EXPERIMENT_RESULTS.md
4. **Reverts HEAD** to that commit
5. **Pushes revert** with `[skip ci]` to prevent loop
6. **:stable tag unchanged** - Streamlit continues working
7. **GitHub notification** sent with rollback details

### Example Rollback Commit Message

```
⚠️ Auto-rollback: Tests failed, reverting to last successful build 611b29b [skip ci]

Reason: Integration tests failed on commit dad549a
Action: Automatically reverted to last known good state
Status: Streamlit deployment unaffected (using :stable tag)
```

---

## Manual Intervention (If Needed)

### Check Current Stable Version

```bash
# View current stable Docker image
docker pull ghcr.io/bilalahmadsheikh/purchasing_power_ml:stable
docker inspect ghcr.io/bilalahmadsheikh/purchasing_power_ml:stable | grep "Created"

# Or check via GitHub API
curl -H "Authorization: token YOUR_TOKEN" \
  https://api.github.com/repos/bilalahmadsheikh/purchasing_power_ml/commits/$(git rev-parse stable)
```

### Manually Rollback Streamlit

If Streamlit is showing issues:

1. **Option A: Reboot app**
   - Go to Streamlit Cloud dashboard
   - Click "Reboot app" (uses current :stable tag)

2. **Option B: Pin to specific commit**
   - Find last working commit from ML_EXPERIMENT_RESULTS.md
   - Streamlit Settings → Advanced → Set `GIT_REF=<commit-sha>`

3. **Option C: Redeploy from scratch**
   - Delete app from Streamlit Cloud
   - Recreate with repository + stable configuration

---

## Monitoring & Verification

### Check if Rollback Occurred

```bash
# View recent commits
git log --oneline -10

# Look for rollback commits
git log --grep="Auto-rollback" --oneline

# View ML_EXPERIMENT_RESULTS.md for successful builds
tail -50 docs/ML_EXPERIMENT_RESULTS.md
```

### Verify Stable Tag Status

**GitHub Actions**:
- Visit: https://github.com/bilalahmadsheikh/purchasing_power_ml/actions
- Check "Promote to Stable" job status
- Verify no rollback jobs have run

**Docker Registry**:
- Visit: https://github.com/bilalahmadsheikh/purchasing_power_ml/pkgs/container/purchasing_power_ml
- Verify `:stable` tag timestamp
- Check tag matches expected commit SHA

---

## Benefits

### ✅ Automatic Protection
- Tests fail → Code automatically reverts
- Streamlit always uses last good build
- No manual intervention needed
- Zero downtime

### ✅ Audit Trail
- All rollbacks logged in git history
- ML_EXPERIMENT_RESULTS.md tracks successful builds
- GitHub Actions logs show failure reasons
- Easy to trace what went wrong

### ✅ Developer Experience
- Push code without fear
- Broken commits auto-revert
- Stable tag always points to working code
- Can still debug failed builds from SHA tags

---

## Testing the Rollback System

### Intentional Failure Test

**WARNING**: Only do this on a test branch, NOT main!

```bash
# Create test branch
git checkout -b test-rollback

# Introduce failing test
echo "def test_fail(): assert False" >> tests/test_integration.py

# Commit and push
git add tests/test_integration.py
git commit -m "test: Intentional failure to test rollback"
git push origin test-rollback

# Create PR to main and merge
# Watch CI/CD automatically rollback
```

### Verify Rollback Worked

1. Check GitHub Actions for "Rollback on Failure" job ✅
2. Verify latest commit message starts with "⚠️ Auto-rollback"
3. Confirm Streamlit still works (using :stable tag)
4. Check ML_EXPERIMENT_RESULTS.md - no new entry for failed commit

---

## Troubleshooting

### Rollback Job Didn't Run

**Possible reasons**:
1. Tests didn't actually fail (check logs)
2. Not on main branch (rollback only on main)
3. GitHub Actions permissions issue

**Fix**:
```bash
# Manually revert
git revert HEAD
git push origin main
```

### Streamlit Deployed Broken Code

**If using branch tracking** (not recommended):
- Streamlit deploys every commit on branch
- Solution: Switch to `:stable` tag or `GIT_REF` pinning

**If using :stable tag**:
- Check if "Promote to Stable" job ran
- Verify tests actually passed
- May need to manually reboot Streamlit app

### Multiple Rollbacks in a Row

**Symptom**: Rollback → Rollback → Rollback (loop)

**Cause**: Rollback commit itself is failing tests

**Fix**:
```bash
# Find last truly successful commit (2-3 commits back)
LAST_GOOD=$(grep -B 3 "Status: ✅ Tests Passed" docs/ML_EXPERIMENT_RESULTS.md | grep "Commit:" | tail -2 | head -1 | awk '{print $2}')

# Hard reset to that commit
git reset --hard $LAST_GOOD
git push --force origin main
```

---

## Configuration Files

### `.github/workflows/ci-cd.yml`

Key sections:
- `build-docker`: Tags images as `:sha` and `:candidate`
- `promote-to-stable`: Only runs if tests pass, tags as `:stable`
- `rollback-on-failure`: Reverts code if tests fail

### `docs/ML_EXPERIMENT_RESULTS.md`

Used by rollback mechanism to find last successful commit:
```bash
LAST_SUCCESS=$(grep -B 3 "Status: ✅ Tests Passed" docs/ML_EXPERIMENT_RESULTS.md | grep "Commit:" | tail -1 | awk '{print $2}')
```

---

## FAQ

**Q: What if I want to deploy a specific version?**
A: Set `GIT_REF=<commit-sha>` in Streamlit settings

**Q: How do I disable automatic rollback?**
A: Remove the `rollback-on-failure` job from `.github/workflows/ci-cd.yml`

**Q: Will this slow down deployments?**
A: No, :stable tag is only updated after tests pass (same timing as before)

**Q: What if tests pass but Streamlit still breaks?**
A: Add Streamlit smoke tests to the CI/CD pipeline or manually pin `GIT_REF` to last working commit

**Q: Can I rollback rollbacks?**
A: Yes, use `git revert <rollback-commit-sha>` or `git reset --hard <good-commit>`

---

## Status

✅ **Active**: Automatic rollback enabled on main branch
✅ **Tested**: System verified with intentional test failures
✅ **Monitored**: ML_EXPERIMENT_RESULTS.md tracks all successful builds

**Last Updated**: 2024-12-22
**Status**: Production Ready
