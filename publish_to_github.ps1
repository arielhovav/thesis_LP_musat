# One-time: put the repo on GitHub. Run from the repo root in PowerShell.
# Prerequisites: git installed (https://git-scm.com), a GitHub account, and either
#   (a) the GitHub CLI `gh` logged in (winget install GitHub.cli ; gh auth login), or
#   (b) an empty repository created in the browser at https://github.com/new (no README, no .gitignore).
param([string]$RepoName = "thesis_LP_musat", [string]$GitHubUser = "", [switch]$Private)

Copy-Item new_chat_kit\.gitignore .gitignore -Force
if (-not (Test-Path README.md)) { Copy-Item new_chat_kit\README_for_github.md README.md }

git init
git add .gitignore README.md *.py *.md *.html analysis lmexp golden*.pt
git add -f model_polished.pt 2>$null
git status --short | Measure-Object -Line | ForEach-Object { "files staged: $($_.Lines)" }
git commit -m "a^n b^n under fixed-precision MDL: code, golden networks, summaries (6 Nov 2026)"

if (Get-Command gh -ErrorAction SilentlyContinue) {
  $vis = if ($Private) { "--private" } else { "--public" }
  gh repo create $RepoName $vis --source . --push
} else {
  if (-not $GitHubUser) { Write-Host "Create an empty repo at https://github.com/new, then rerun with -GitHubUser <name>"; exit }
  git branch -M main
  git remote add origin "https://github.com/$GitHubUser/$RepoName.git"
  git push -u origin main
}
Write-Host "done. runs/ is ignored (regenerable); if you also want the sweep outputs online, zip runs/ and attach it to a GitHub Release."
