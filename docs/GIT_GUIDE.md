# Git Commands Guide: Best Practices and Workflows

A comprehensive guide to all Git commands used in this repository, with explanations, examples, and best practices.

## Table of Contents
- [Repository Setup](#repository-setup)
- [Basic Git Workflow](#basic-git-workflow)
- [Checking Status and History](#checking-status-and-history)
- [Staging and Committing](#staging-and-committing)
- [Pushing and Pulling](#pushing-and-pulling)
- [Branching and Merging](#branching-and-merging)
- [Undoing Changes](#undoing-changes)
- [Commit Message Best Practices](#commit-message-best-practices)
- [Advanced Workflows](#advanced-workflows)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Git Configuration](#git-configuration)

---

## Repository Setup

### Initialize a New Repository

```bash
# Create new repository
git init

# Initialize with main branch (modern convention)
git init --initial-branch=main
```

**When to use:**
- Starting a new project from scratch
- Converting an existing directory to a Git repository

**Best practices:**
- ✅ Use `main` as default branch name (not `master`)
- ✅ Create `.gitignore` immediately
- ✅ Make first commit with README

### Clone an Existing Repository

```bash
# Clone from GitHub (what we did)
git clone https://github.com/ars137th/llm-post-training.git

# Clone to specific directory
git clone https://github.com/ars137th/llm-post-training.git my-folder

# Clone specific branch
git clone -b develop https://github.com/ars137th/llm-post-training.git
```

**What happens:**
1. Creates a new directory with repository name
2. Downloads all files and commit history
3. Sets up remote connection named `origin`
4. Checks out the default branch

**Best practices:**
- ✅ Clone via HTTPS for public repos
- ✅ Clone via SSH for private repos (faster, no password prompts)
- ✅ Verify repository URL before cloning

### Set Up Remote Connection

```bash
# View current remotes
git remote -v

# Add a remote
git remote add origin https://github.com/username/repo.git

# Change remote URL
git remote set-url origin https://github.com/username/new-repo.git

# Remove a remote
git remote remove origin
```

**Example from our work:**
```bash
# Check current remote
$ git remote -v
origin  https://github.com/ars137th/llm-post-training.git (fetch)
origin  https://github.com/ars137th/llm-post-training.git (push)
```

---

## Basic Git Workflow

### The Standard Workflow (What We Did Throughout)

```bash
# 1. Check status (see what's changed)
git status

# 2. Add files to staging area
git add <files>

# 3. Commit with message
git commit -m "Description of changes"

# 4. Push to remote
git push origin <branch>

# 5. Pull latest changes (when resuming work)
git pull origin <branch>
```

**Visual representation:**

```
Working Directory → Staging Area → Local Repository → Remote Repository
     (edit)           (git add)      (git commit)        (git push)
```

### Our Actual Workflow During DPO Implementation

```bash
# Created DPO loss functions
git add src/core/dpo/loss.py
git commit -m "Add DPO loss function implementation"
git push origin master

# Created DPO trainer
git add src/core/dpo/trainer.py
git commit -m "Add DPO trainer implementation"
git push origin master

# Created training script
git add scripts/train/train_dpo.py
git commit -m "Add DPO training script"
git push origin master

# Created configs
git add configs/
git commit -m "Add DPO configuration files"
git push origin master
```

**Pattern:**
- Small, focused commits
- Clear commit messages
- Push after each logical unit of work

---

## Checking Status and History

### git status - See What's Changed

```bash
# Full status
git status

# Short format (cleaner)
git status -s

# Show branch and tracking info
git status -sb
```

**Example output:**
```
On branch master
Your branch is up to date with 'origin/master'.

Changes not staged for commit:
  modified:   src/core/dpo/trainer.py

Untracked files:
  docs/DPO_CONFIGURATION.md

no changes added to commit
```

**What each section means:**
- **Untracked files**: New files Git doesn't know about (red in terminal)
- **Changes not staged**: Modified files not yet added (red)
- **Changes to be committed**: Files in staging area (green)

**Best practices:**
- ✅ Run before committing to verify what you're including
- ✅ Check for unintended files (secrets, large files, temp files)
- ✅ Use `.gitignore` for files that should never be tracked

### git log - View Commit History

```bash
# Standard log
git log

# One line per commit (cleaner)
git log --oneline

# Last 5 commits
git log -n 5

# Pretty format with graph
git log --oneline --graph --all --decorate

# Search commits by message
git log --grep="DPO"

# See what changed in each commit
git log -p
```

**Example from our work:**
```bash
$ git log --oneline -n 5
f587066 Add implementation philosophy to DPO docs and Python practices guide
9bb7df8 Pass inner models to DPOTrainer, not LanguageModel wrappers
c677adb Fix print_model_info to access LanguageModel attributes correctly
08e63df Fix device handling in DPO training script
4c21fab Add DPO configuration files
```

**Best practices:**
- ✅ Use `--oneline` for quick overview
- ✅ Use `--graph` to visualize branches
- ✅ Use `--grep` to find specific commits

### git diff - See Specific Changes

```bash
# See unstaged changes
git diff

# See staged changes
git diff --staged

# Compare two commits
git diff commit1 commit2

# See changes in specific file
git diff path/to/file

# Word-level diff (not line-level)
git diff --word-diff
```

**Example:**
```bash
# See what changed before committing
$ git diff src/core/dpo/trainer.py
```

---

## Staging and Committing

### git add - Stage Files for Commit

```bash
# Add specific file
git add src/core/dpo/loss.py

# Add multiple files
git add file1.py file2.py file3.py

# Add all files in directory
git add src/core/dpo/

# Add all changed files (use carefully!)
git add .

# Add all Python files
git add "*.py"

# Add interactively (choose hunks)
git add -p
```

**What we did:**
```bash
# Add single file
git add src/core/dpo/trainer.py

# Add entire directory
git add configs/

# Add multiple specific files
git add docs/DPO_THEORY.md docs/PYTHON_PRACTICES.md
```

**Best practices:**
- ✅ Add files selectively (not `git add .` blindly)
- ✅ Review with `git status` before committing
- ✅ Use `git add -p` for partial file commits
- ❌ Never add secrets, API keys, or credentials
- ❌ Don't add large binary files (use Git LFS)
- ❌ Don't add generated files (build artifacts, `__pycache__`, etc.)

### git commit - Save Changes

```bash
# Commit with inline message
git commit -m "Add DPO loss function"

# Commit with detailed message (opens editor)
git commit

# Commit and add all tracked files
git commit -am "Quick fix"

# Amend previous commit (change message or add files)
git commit --amend

# Amend without changing message
git commit --amend --no-edit
```

**Our commit examples:**
```bash
# Simple, clear message
git commit -m "Add DPO trainer implementation"

# Multi-line message with details
git commit -m "Fix device handling in DPO training script

- Pass device parameter to LanguageModel.from_pretrained()
- Remove invalid .cpu() calls (LanguageModel doesn't have that method)
- Handle 'auto' device option properly
- Fixes AttributeError: 'LanguageModel' object has no attribute 'cpu'"

# Amend to add forgotten file
git add forgotten_file.py
git commit --amend --no-edit
```

**Best practices:**
- ✅ Write clear, descriptive commit messages
- ✅ Use present tense ("Add" not "Added")
- ✅ Start with capital letter
- ✅ Keep first line under 50 characters
- ✅ Add detailed description after blank line
- ❌ Don't commit broken code
- ❌ Don't commit with "WIP" or "temp" messages to main branch

---

## Pushing and Pulling

### git push - Upload Commits to Remote

```bash
# Push to default remote and branch
git push

# Push to specific remote and branch
git push origin master

# Push and set upstream (first push on new branch)
git push -u origin feature-branch

# Force push (dangerous!)
git push --force

# Safer force push (fails if remote has new commits)
git push --force-with-lease
```

**What we did:**
```bash
# Standard push to master
git push origin master

# Output:
# To https://github.com/ars137th/llm-post-training
#    08e63df..ba1985a  master -> master
```

**Best practices:**
- ✅ Pull before pushing to avoid conflicts
- ✅ Use `-u` flag first time pushing new branch
- ✅ Push frequently (don't let local/remote diverge)
- ❌ Never force push to main/master
- ❌ Never force push to shared branches
- ⚠️ Only force push to your own feature branches (with `--force-with-lease`)

### git pull - Download Changes from Remote

```bash
# Pull from current branch's remote
git pull

# Pull from specific remote and branch
git pull origin master

# Pull with rebase (cleaner history)
git pull --rebase

# Fetch without merging
git fetch origin
```

**What happens:**
1. Downloads commits from remote
2. Merges them into your local branch
3. If conflicts, you must resolve them

**Best practices:**
- ✅ Pull before starting new work
- ✅ Pull before pushing (avoid conflicts)
- ✅ Use `git pull --rebase` for cleaner history
- ✅ Commit or stash local changes before pulling

### git fetch vs git pull

```bash
# Fetch only (download but don't merge)
git fetch origin

# See what was fetched
git log origin/master

# Manually merge after reviewing
git merge origin/master

# Pull = fetch + merge
git pull origin master
```

**When to use:**
- **fetch**: When you want to review changes before merging
- **pull**: When you trust the remote changes

---

## Branching and Merging

### git branch - Manage Branches

```bash
# List all branches
git branch

# List all branches (including remote)
git branch -a

# Create new branch
git branch feature-dpo

# Create and switch to new branch
git checkout -b feature-dpo

# Modern syntax (Git 2.23+)
git switch -c feature-dpo

# Delete branch (safe, won't delete unmerged)
git branch -d feature-dpo

# Force delete branch
git branch -D feature-dpo

# Rename current branch
git branch -m new-name
```

**What we encountered:**
```bash
# Check which branch we're on
$ git branch -a
* master
  remotes/origin/master

# Note: We worked directly on master (small team, educational repo)
```

**Best practices:**
- ✅ Create feature branches for new work
- ✅ Use descriptive branch names: `feature/dpo-implementation`, `fix/trainer-bug`
- ✅ Keep main/master clean (deployable)
- ✅ Delete branches after merging
- ❌ Don't work directly on main/master (except for small repos)

### git merge - Combine Branches

```bash
# Merge branch into current branch
git merge feature-branch

# Merge with commit message
git merge feature-branch -m "Merge DPO feature"

# Abort merge if conflicts
git merge --abort

# Fast-forward only (fails if diverged)
git merge --ff-only feature-branch

# No fast-forward (always create merge commit)
git merge --no-ff feature-branch
```

**Example workflow:**
```bash
# On feature branch: implement DPO
git checkout -b feature/dpo
git add src/core/dpo/
git commit -m "Implement DPO"

# Switch to main and merge
git checkout main
git merge feature/dpo

# Delete feature branch
git branch -d feature/dpo
```

### git rebase - Alternative to Merge

```bash
# Rebase current branch onto master
git rebase master

# Interactive rebase (rewrite history)
git rebase -i HEAD~3

# Continue after resolving conflicts
git rebase --continue

# Abort rebase
git rebase --abort
```

**Rebase vs Merge:**

**Merge** (what we used):
```
A---B---C feature
         \
D---E-----F---G master (merge commit F)
```

**Rebase**:
```
A---B---C (original)
         
D---E---A'---B'---C' (rebased, linear history)
```

**Best practices:**
- ✅ Use merge for feature branches (preserves history)
- ✅ Use rebase for cleanup before pushing
- ✅ Use interactive rebase to squash commits
- ❌ Never rebase public/shared branches
- ❌ Never rebase commits already pushed to main

---

## Undoing Changes

### Unstage Files

```bash
# Unstage specific file
git reset HEAD file.py

# Unstage all files
git reset HEAD

# Modern syntax
git restore --staged file.py
```

### Discard Local Changes

```bash
# Discard changes in working directory
git checkout -- file.py

# Modern syntax
git restore file.py

# Discard all changes (dangerous!)
git checkout .
```

### Undo Last Commit (Keep Changes)

```bash
# Undo commit, keep changes staged
git reset --soft HEAD~1

# Undo commit, keep changes unstaged
git reset HEAD~1

# Undo commit, discard changes (dangerous!)
git reset --hard HEAD~1
```

**What we did when we made mistakes:**
```bash
# Example: Committed wrong file
git add wrong_file.py
git commit -m "Add feature"

# Undo commit, keep changes
git reset HEAD~1

# Now fix and commit correctly
git add correct_file.py
git commit -m "Add feature"
```

### Revert a Commit (Safe)

```bash
# Create new commit that undoes previous commit
git revert abc123

# Revert without committing
git revert --no-commit abc123
```

**Revert vs Reset:**
- **Reset**: Rewrites history (use only on local commits)
- **Revert**: Creates new commit (safe for pushed commits)

### Recover Deleted Files

```bash
# Restore deleted file from last commit
git checkout HEAD -- deleted_file.py

# Find commit where file was deleted
git log --all --full-history -- deleted_file.py

# Restore from specific commit
git checkout abc123 -- deleted_file.py
```

---

## Commit Message Best Practices

### Our Commit Message Format

We followed these conventions throughout:

```bash
# Short, descriptive summary (50 chars or less)
git commit -m "Add DPO loss function implementation"

# Detailed message with bullet points
git commit -m "Fix device handling in DPO training script

- Pass device parameter to LanguageModel.from_pretrained()
- Remove invalid .cpu() calls (LanguageModel doesn't have that method)
- Handle 'auto' device option properly
- Fixes AttributeError: 'LanguageModel' object has no attribute 'cpu'"
```

### Commit Message Template

```
<type>: <subject>

<body>

<footer>
```

**Example:**
```
feat: Add DPO trainer implementation

- Custom DPOTrainer extending HuggingFace Trainer
- Manages policy (trainable) and reference (frozen) models
- Implements compute_loss with DPO/IPO objective
- Tracks implicit rewards, KL divergence, accuracy

Closes #42
```

### Conventional Commits

```bash
# Feature
git commit -m "feat: add DPO loss function"

# Bug fix
git commit -m "fix: resolve device handling in trainer"

# Documentation
git commit -m "docs: add DPO configuration guide"

# Refactoring
git commit -m "refactor: simplify log probability computation"

# Tests
git commit -m "test: add unit tests for DPO loss"

# Chore (maintenance)
git commit -m "chore: update dependencies"
```

**Types we used:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

### Good vs Bad Commit Messages

**❌ Bad:**
```bash
git commit -m "fix"
git commit -m "WIP"
git commit -m "updates"
git commit -m "Fixed bug"
git commit -m "Changed some files"
```

**✅ Good (what we did):**
```bash
git commit -m "Add DPO trainer implementation"
git commit -m "Fix device handling in DPO training script"
git commit -m "Add comprehensive DPO configuration documentation"
git commit -m "Pass inner models to DPOTrainer, not LanguageModel wrappers"
```

**Rules:**
1. ✅ Use imperative mood: "Add feature" not "Added feature"
2. ✅ Capitalize first letter
3. ✅ No period at end of subject line
4. ✅ Limit subject to 50 characters
5. ✅ Separate subject from body with blank line
6. ✅ Wrap body at 72 characters
7. ✅ Explain *what* and *why*, not *how*

---

## Advanced Workflows

### Stashing Changes

```bash
# Save work in progress
git stash

# Stash with message
git stash save "Work on DPO trainer"

# List stashes
git stash list

# Apply most recent stash
git stash apply

# Apply and remove stash
git stash pop

# Apply specific stash
git stash apply stash@{2}

# Drop stash
git stash drop stash@{0}

# Clear all stashes
git stash clear
```

**When to use:**
- Need to switch branches but have uncommitted work
- Want to pull latest changes but have local modifications
- Testing something temporarily

### Cherry-Picking Commits

```bash
# Apply specific commit to current branch
git cherry-pick abc123

# Cherry-pick multiple commits
git cherry-pick abc123 def456

# Cherry-pick without committing
git cherry-pick --no-commit abc123
```

**When to use:**
- Need specific fix from another branch
- Want to backport commit to older version

### Interactive Rebase (Rewriting History)

```bash
# Rewrite last 3 commits
git rebase -i HEAD~3

# In editor:
# pick abc123 First commit
# squash def456 Second commit (combine with first)
# reword ghi789 Third commit (change message)
```

**Commands in interactive rebase:**
- `pick`: Keep commit as-is
- `reword`: Keep commit, change message
- `squash`: Combine with previous commit
- `fixup`: Like squash, discard commit message
- `drop`: Remove commit

**When to use:**
- Clean up messy commit history before pushing
- Combine related commits
- Fix commit messages

### Submodules

```bash
# Add submodule
git submodule add https://github.com/user/repo.git path/to/submodule

# Clone repo with submodules
git clone --recurse-submodules https://github.com/user/repo.git

# Initialize submodules in existing clone
git submodule init
git submodule update

# Update submodules
git submodule update --remote
```

---

## Common Issues and Solutions

### Issue 1: Merge Conflicts

**Symptoms:**
```bash
$ git merge feature-branch
Auto-merging src/core/dpo/trainer.py
CONFLICT (content): Merge conflict in src/core/dpo/trainer.py
Automatic merge failed; fix conflicts and then commit the result.
```

**Solution:**
```bash
# 1. Open conflicted file
# Look for:
<<<<<<< HEAD
# Your changes
=======
# Their changes
>>>>>>> feature-branch

# 2. Edit file to resolve conflict
# 3. Remove conflict markers
# 4. Stage resolved file
git add src/core/dpo/trainer.py

# 5. Complete merge
git commit
```

**Best practices:**
- ✅ Communicate with team about conflicting changes
- ✅ Pull frequently to minimize conflicts
- ✅ Use tools: VSCode, Git GUI, merge tools

### Issue 2: Accidentally Committed to Wrong Branch

**Solution:**
```bash
# On wrong branch
git log  # Note commit hash

# Undo commit
git reset --hard HEAD~1

# Switch to correct branch
git checkout correct-branch

# Apply the commit
git cherry-pick <commit-hash>
```

### Issue 3: Pushed Sensitive Data

**⚠️ IMPORTANT: If you push secrets/keys to GitHub:**

1. **Immediately rotate credentials** (new API key, password, etc.)
2. Remove from history:
```bash
# Use BFG Repo-Cleaner or git-filter-branch
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive/file" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (notify team!)
git push origin --force --all
```

3. **Better:** Use secrets management (environment variables, vault)

### Issue 4: Diverged Branches

**Symptoms:**
```bash
$ git push
! [rejected]        master -> master (non-fast-forward)
```

**Solution:**
```bash
# Option 1: Pull and merge
git pull origin master
# Resolve conflicts if any
git push origin master

# Option 2: Pull with rebase (cleaner)
git pull --rebase origin master
git push origin master

# Option 3: Force push (only if you're sure!)
git push --force-with-lease origin master
```

### Issue 5: Large Files

**GitHub rejects files > 100MB:**

```bash
# Error
remote: error: GH001: Large files detected.
```

**Solution:**
```bash
# Use Git LFS for large files
git lfs install
git lfs track "*.pth"  # PyTorch models
git lfs track "*.bin"  # Model weights
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

---

## Git Configuration

### User Configuration

```bash
# Set name and email (required)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# View config
git config --list

# Edit config file
git config --global --edit
```

### Useful Aliases

```bash
# Create shortcuts
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'

# Pretty log
git config --global alias.lg "log --oneline --graph --all --decorate"

# Usage
git st  # Instead of git status
git lg  # Pretty log
```

### Default Editor

```bash
# Use VSCode
git config --global core.editor "code --wait"

# Use vim
git config --global core.editor "vim"

# Use nano
git config --global core.editor "nano"
```

### Default Branch Name

```bash
# Use 'main' instead of 'master'
git config --global init.defaultBranch main
```

### Line Endings (Important for Cross-Platform)

```bash
# macOS/Linux
git config --global core.autocrlf input

# Windows
git config --global core.autocrlf true
```

---

## Our Git Workflow Summary

Throughout the DPO implementation, we followed this workflow:

```bash
# 1. Check status before starting
git status

# 2. Create/modify files
# ... write code ...

# 3. Stage specific files
git add src/core/dpo/loss.py

# 4. Commit with clear message
git commit -m "Add DPO loss function implementation

- Implements dpo_loss() with log-sigmoid
- Implements ipo_loss() with squared loss
- Includes compute_sequence_log_probs() helper
- Adds detailed metrics computation
- Fully documented with examples"

# 5. Push to remote
git push origin master

# 6. Fix issues (if needed)
git add fixed_file.py
git commit -m "Fix device handling bug"
git push origin master
```

**Key principles we followed:**
- ✅ Small, focused commits (each does one thing)
- ✅ Clear, descriptive commit messages
- ✅ Push after each logical unit of work
- ✅ Fix bugs immediately with new commits (not amend)
- ✅ Document everything

---

## Quick Reference Card

### Essential Commands

| Command | Purpose |
|---------|---------|
| `git status` | Check what's changed |
| `git add <file>` | Stage file for commit |
| `git commit -m "msg"` | Commit staged changes |
| `git push origin <branch>` | Upload commits |
| `git pull origin <branch>` | Download changes |
| `git log --oneline` | View commit history |
| `git diff` | See unstaged changes |
| `git branch` | List branches |
| `git checkout -b <name>` | Create and switch branch |
| `git merge <branch>` | Merge branch |

### Emergency Commands

| Problem | Solution |
|---------|----------|
| Uncommit last commit | `git reset HEAD~1` |
| Discard local changes | `git checkout -- <file>` |
| Undo pushed commit | `git revert <commit>` |
| Abort merge | `git merge --abort` |
| Stash current work | `git stash` |
| Unstage file | `git restore --staged <file>` |

---

## Resources

**Official Documentation:**
- Git Book: https://git-scm.com/book/en/v2
- Git Reference: https://git-scm.com/docs

**Interactive Learning:**
- Learn Git Branching: https://learngitbranching.js.org/
- Git Immersion: https://gitimmersion.com/

**Cheat Sheets:**
- GitHub Git Cheat Sheet: https://education.github.com/git-cheat-sheet-education.pdf
- Atlassian Git Tutorials: https://www.atlassian.com/git/tutorials

**Tools:**
- GitHub Desktop: GUI for Git
- GitKraken: Visual Git client
- VSCode: Built-in Git integration

---

## Summary

**Our Git workflow for this project:**

1. ✅ **Work directly on master** (small team, educational repo)
2. ✅ **Commit frequently** (after each feature/fix)
3. ✅ **Write clear messages** (what changed and why)
4. ✅ **Push immediately** (keep remote in sync)
5. ✅ **Fix bugs with new commits** (transparent history)

**For larger projects:**
- Use feature branches
- Create pull requests
- Code review before merge
- CI/CD testing
- Protected main branch

**Remember:**
- Git is your safety net (commits are snapshots)
- Don't be afraid to commit (you can always undo)
- Push frequently (backup and collaboration)
- Read error messages (Git is usually helpful)
- Use `.gitignore` properly (never commit secrets!)

Happy Git-ing! 🚀
