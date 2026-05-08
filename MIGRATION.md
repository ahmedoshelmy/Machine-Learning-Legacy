# Repository Rename Guide: Machine-Learning-Legacy → ai-projects

This guide walks you through renaming your repository both locally and on GitHub.

## Step 1: Close VS Code Workspace

1. In VS Code, go to **File** → **Close Folder**
2. This will release the lock on the folder

## Step 2: Rename Local Repository

Once the folder is unlocked, open PowerShell and run:

```powershell
cd c:\code\Personal
ren "Machine-Learning-Legacy" "ai-projects"
cd ai-projects
```

## Step 3: Rename on GitHub

### Option A: Using GitHub Web Interface (Recommended)
1. Go to https://github.com/ahmedoshelmy/Machine-Learning-Legacy
2. Click **Settings** (top right)
3. Under "Repository name", change to: `ai-projects`
4. Click **Rename**

### Option B: Using GitHub CLI
```bash
gh repo rename ai-projects
```

## Step 4: Update Local Git Remote

After renaming on GitHub, update your local repository:

```powershell
cd c:\code\Personal\ai-projects
git remote set-url origin https://github.com/ahmedoshelmy/ai-projects.git
```

Verify the change:
```powershell
git remote -v
```

You should see:
```
origin  https://github.com/ahmedoshelmy/ai-projects.git (fetch)
origin  https://github.com/ahmedoshelmy/ai-projects.git (push)
```

## Step 5: Commit and Push Changes

```powershell
cd c:\code\Personal\ai-projects
git add .
git commit -m "chore: organize repository with courses, projects, and utilities structure"
git push origin main
```

## Step 6: Reopen Workspace

1. Open VS Code
2. File → Open Folder
3. Navigate to `c:\code\Personal\ai-projects`
4. Click "Select Folder"

## Verification Checklist

- [ ] Folder renamed locally to `ai-projects`
- [ ] GitHub repository renamed to `ai-projects`
- [ ] Git remote URL updated
- [ ] All files and structure visible in VS Code
- [ ] New folders appear:
  - [ ] `courses/`
  - [ ] `projects/` (with 10 sub-projects)
  - [ ] `tools-and-utilities/`
- [ ] README.md updated with new structure
- [ ] All changes pushed to GitHub

## Next Steps

1. **Move existing course materials** to `courses/` subdirectories
2. **Move existing projects** to `projects/` 
3. **Organize utilities** in `tools-and-utilities/`
4. **Update project READMEs** as you work on each one

---

If you encounter any issues, check:
- VS Code is completely closed before renaming
- GitHub shows the updated repository name
- Git remote is pointing to the new URL
