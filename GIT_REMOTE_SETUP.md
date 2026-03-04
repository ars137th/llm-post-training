# Git Remote Setup Guide

## Overview

Your repository is currently **local only**. This guide will teach you how to push it to a remote repository (GitHub, GitLab, or Bitbucket) so you can:
- Back up your code
- Share it with others
- Access it from different machines (macOS, Colab, Databricks)
- Collaborate with team members

---

## Current Status

```bash
# Your current git configuration
Branch: master
Remote: None (local only)
Git User: LLM Post-Training <creator@example.com>  # Placeholder values
```

---

## Step 1: Update Your Git Identity

First, update your git user information from placeholders to your real identity:

```bash
# Navigate to repository
cd /Users/akhil.shah/code/claude_sandbox/llm-post-training

# Set your real name and email
git config user.name "Your Real Name"
git config user.email "your.email@example.com"

# Verify
git config user.name
git config user.email
```

**Note**: This only affects this repository. To set globally for all repositories:
```bash
git config --global user.name "Your Real Name"
git config --global user.email "your.email@example.com"
```

---

## Step 2: Choose a Remote Platform

### Option A: GitHub (Most Popular)
- **Best for**: Open source, public repos, community
- **Free tier**: Unlimited public repos, unlimited private repos
- **URL**: https://github.com

### Option B: GitLab
- **Best for**: CI/CD integration, DevOps workflows
- **Free tier**: Unlimited public/private repos, 400 CI/CD minutes/month
- **URL**: https://gitlab.com

### Option C: Bitbucket
- **Best for**: Integration with Jira, Atlassian ecosystem
- **Free tier**: Unlimited private repos (up to 5 users)
- **URL**: https://bitbucket.org

**Recommendation**: Use **GitHub** if unsure - it has the largest community and best integration with tools.

---

## Step 3: Create Remote Repository on GitHub

### Method 1: GitHub Web UI (Easiest)

1. **Go to GitHub**: https://github.com
2. **Sign in** (or create account if needed)
3. **Create new repository**:
   - Click the "+" icon in top-right corner
   - Select "New repository"
   - **Repository name**: `llm-post-training`
   - **Description**: "Comprehensive repository for learning and experimenting with LLM post-training techniques"
   - **Visibility**: Choose Public or Private
   - **⚠️ IMPORTANT**: Do NOT initialize with README, .gitignore, or license (we already have these locally)
   - Click "Create repository"

4. **Copy the repository URL** (shown on next page):
   ```
   https://github.com/yourusername/llm-post-training.git
   ```

### Method 2: GitHub CLI (If you have `gh` installed)

```bash
# Install GitHub CLI (if not installed)
brew install gh

# Login
gh auth login

# Create repository
gh repo create llm-post-training \
  --public \
  --source=. \
  --description="Comprehensive repository for LLM post-training techniques" \
  --remote=origin
```

---

## Step 4: Add Remote to Local Repository

After creating the remote repository, link your local repo to it:

```bash
# Navigate to repository
cd /Users/akhil.shah/code/claude_sandbox/llm-post-training

# Add remote (replace 'yourusername' with your GitHub username)
git remote add origin https://github.com/yourusername/llm-post-training.git

# Verify remote was added
git remote -v
# Should show:
# origin  https://github.com/yourusername/llm-post-training.git (fetch)
# origin  https://github.com/yourusername/llm-post-training.git (push)
```

### Using SSH Instead of HTTPS (Optional, More Secure)

If you have SSH keys set up:

```bash
# Add remote with SSH URL
git remote add origin git@github.com:yourusername/llm-post-training.git
```

**To set up SSH keys** (if you don't have them):
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Go to GitHub → Settings → SSH and GPG keys → New SSH key
# Paste the public key
```

---

## Step 5: Push to Remote

### Initial Push (First Time)

```bash
# Push your master branch to remote and set upstream
git push -u origin master

# The -u flag sets 'origin/master' as the default upstream
# After this, you can just use 'git push' without specifying remote/branch
```

**Expected output**:
```
Enumerating objects: 150, done.
Counting objects: 100% (150/150), done.
Delta compression using up to 10 threads
Compressing objects: 100% (120/120), done.
Writing objects: 100% (150/150), 75.23 KiB | 5.02 MiB/s, done.
Total 150 (delta 45), reused 0 (delta 0), pack-reused 0
To https://github.com/yourusername/llm-post-training.git
 * [new branch]      master -> master
Branch 'master' set up to track remote branch 'master' from 'origin'.
```

### Authentication

If using HTTPS, you'll need to authenticate:

**Option 1: Personal Access Token (Recommended)**
```bash
# GitHub will prompt for username and password
Username: yourusername
Password: <paste your Personal Access Token>

# Create token at: https://github.com/settings/tokens
# Scopes needed: repo (full control of private repositories)
```

**Option 2: GitHub CLI (Easiest)**
```bash
gh auth login
# Follow prompts to authenticate
```

**Option 3: Credential Helper (Saves credentials)**
```bash
# On macOS, use keychain
git config --global credential.helper osxkeychain

# On Linux
git config --global credential.helper store
```

---

## Step 6: Verify Push Succeeded

1. **Check GitHub**:
   - Go to https://github.com/yourusername/llm-post-training
   - You should see all your files: README.md, src/, examples/, etc.

2. **Check local status**:
   ```bash
   git status
   # Should show: "Your branch is up to date with 'origin/master'"

   git log --oneline -5
   # Should show your recent commits
   ```

---

## Ongoing Workflow: Making Changes and Pushing

After the initial push, your workflow becomes:

```bash
# 1. Make changes to files
# (edit code, add files, etc.)

# 2. Check what changed
git status
git diff

# 3. Stage changes
git add <files>
# or stage everything:
git add .

# 4. Commit with message
git commit -m "Your commit message"

# 5. Push to remote
git push
# No need to specify 'origin master' after first push with -u

# 6. Verify on GitHub
# Check https://github.com/yourusername/llm-post-training
```

---

## Common Git Remote Commands

### Viewing Remote Info
```bash
# List remotes
git remote -v

# Show detailed remote info
git remote show origin

# Get latest changes from remote (without merging)
git fetch origin

# Pull changes from remote (fetch + merge)
git pull origin master
```

### Updating Remote URL
```bash
# Change remote URL (e.g., switching from HTTPS to SSH)
git remote set-url origin git@github.com:yourusername/llm-post-training.git

# Verify
git remote -v
```

### Removing Remote
```bash
# Remove remote
git remote remove origin

# Verify
git remote -v
```

---

## Cloning Your Repository Elsewhere

Once pushed, you can clone it on other machines:

### On Linux/Colab/Cloud
```bash
# Clone repository
git clone https://github.com/yourusername/llm-post-training.git

# Navigate to it
cd llm-post-training

# Install dependencies
pip install -e .

# Start working!
python examples/minimal_sft.py
```

### On Google Colab
```python
# In a Colab cell
!git clone https://github.com/yourusername/llm-post-training.git
%cd llm-post-training
!pip install -e .
```

### On Databricks
```python
# In a notebook
%sh
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -e .
```

---

## Branch Strategy (Optional, for Team Collaboration)

### Creating a Development Branch

```bash
# Create and switch to dev branch
git checkout -b develop

# Make changes, commit them
git add .
git commit -m "Add new feature"

# Push dev branch to remote
git push -u origin develop

# On GitHub, create Pull Request: develop → master
# After review and approval, merge on GitHub
```

### Common Branch Commands
```bash
# List branches
git branch -a

# Switch branches
git checkout master
git checkout develop

# Create new branch
git checkout -b feature/new-feature

# Delete branch
git branch -d feature/old-feature

# Push branch to remote
git push -u origin feature/new-feature
```

---

## Protecting Your Master Branch (Recommended)

On GitHub, you can protect important branches:

1. Go to repository **Settings**
2. Click **Branches**
3. Click **Add rule** under "Branch protection rules"
4. **Branch name pattern**: `master`
5. Enable:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass before merging
   - ✅ Include administrators (if you want to enforce for yourself too)
6. Click **Create**

Now changes to `master` require:
- Creating a feature branch
- Pushing the branch
- Opening a Pull Request
- Getting approval/passing checks
- Merging the PR

---

## .gitignore Considerations

Your `.gitignore` is already set up correctly:

```bash
# What's ignored (won't be pushed to remote)
✅ Python cache (__pycache__, *.pyc)
✅ Virtual environments (venv/, env/)
✅ IDE files (.vscode/, .idea/)
✅ Model checkpoints (*.pt, *.bin, *.safetensors)
✅ Datasets (datasets/raw/*, datasets/processed/*)
✅ Logs and outputs (*.log, outputs/)
✅ Downloaded HuggingFace models (/models/)

# What's tracked (will be pushed)
✅ Source code (src/)
✅ Scripts (scripts/, examples/)
✅ Configs (configs/)
✅ Documentation (*.md)
✅ Requirements (requirements/)
✅ Tests (tests/)
```

---

## Large Files Warning

GitHub has limits:
- **File size limit**: 100 MB per file
- **Repository size**: Should stay under 1 GB for best performance

If you need to track large files (model checkpoints, datasets):

### Option 1: Use Git LFS (Large File Storage)

```bash
# Install Git LFS
brew install git-lfs
git lfs install

# Track large files
git lfs track "checkpoints/**/*.pt"
git lfs track "checkpoints/**/*.bin"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

### Option 2: Use External Storage
- Store models on HuggingFace Hub
- Store datasets on cloud storage (S3, GCS, Azure Blob)
- Reference them in code by URL/path

---

## Security Considerations

### Never Commit Secrets!

Already in `.gitignore`:
```
.env
.env.local
.env.*.local
```

But double-check you never accidentally commit:
- ❌ API keys (OpenAI, HuggingFace, Anthropic)
- ❌ Passwords or tokens
- ❌ Private SSH keys
- ❌ Database credentials
- ❌ AWS/GCP/Azure credentials

### If You Accidentally Commit a Secret

```bash
# Remove from history (nuclear option - rewrites history)
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch path/to/secret/file' \
  --prune-empty --tag-name-filter cat -- --all

# Force push (⚠️ only if you haven't shared the repo yet)
git push origin --force --all

# Rotate the secret immediately!
# (Change API key, password, etc.)
```

---

## Quick Reference

### Initial Setup (One Time)
```bash
# 1. Update git identity
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 2. Create repo on GitHub
# (Use web UI: https://github.com/new)

# 3. Add remote
git remote add origin https://github.com/yourusername/llm-post-training.git

# 4. Push
git push -u origin master
```

### Daily Workflow
```bash
# Make changes → stage → commit → push
git add .
git commit -m "Description of changes"
git push
```

### Cloning Elsewhere
```bash
git clone https://github.com/yourusername/llm-post-training.git
cd llm-post-training
pip install -e .
```

---

## Troubleshooting

### "Permission denied (publickey)"
**Problem**: SSH authentication failed
**Solution**: Either use HTTPS URL or set up SSH keys (see Step 4)

### "failed to push some refs"
**Problem**: Remote has changes you don't have locally
**Solution**:
```bash
git pull origin master --rebase
git push origin master
```

### "Authentication failed"
**Problem**: Invalid credentials for HTTPS
**Solution**: Use Personal Access Token instead of password
```bash
# Create token at: https://github.com/settings/tokens
# Use token as password when prompted
```

### "Large files detected"
**Problem**: File > 100 MB
**Solution**: Use Git LFS or exclude from git
```bash
git lfs track "path/to/large/file"
# or add to .gitignore
```

---

## Next Steps After Pushing

1. **Update README.md** with your GitHub repo URL
2. **Add repository description** on GitHub
3. **Add topics/tags** on GitHub (e.g., "pytorch", "transformers", "rlhf", "llm")
4. **Enable GitHub Actions** for CI/CD (optional)
5. **Add collaborators** if working with a team
6. **Create project board** for task tracking (optional)

---

## Summary

After following this guide, you will:
- ✅ Have your code backed up on GitHub/GitLab/Bitbucket
- ✅ Be able to access it from any machine (macOS, Colab, Databricks)
- ✅ Have a shareable URL for collaboration
- ✅ Have version history accessible online
- ✅ Be ready to work on the code from GPU platforms

**Your repository will be accessible at**:
`https://github.com/yourusername/llm-post-training`

From there, anyone (or you on another machine) can:
```bash
git clone https://github.com/yourusername/llm-post-training.git
pip install -e .
python examples/minimal_sft.py
```

Happy coding! 🚀
