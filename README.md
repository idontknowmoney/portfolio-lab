# portfolio-lab

## Branching & contributing

```
feature/*, fix/*  --PR--> develop --PR (release or single feature)--> main
hotfix/*  (from main) --PR--> main --back-merge PR--> develop
```

- `main` is production, `develop` is the dev environment. Both are protected: no direct pushes, no force pushes, no deletion.
- Every change goes through a pull request with a passing `lint-test` check and 1 approval (repo admins can bypass the approval).
- Branch names: `feature/<x>`, `fix/<x>`, `hotfix/<x>`, `release/<x>`.
- PRs into `main` must come from `develop`, `hotfix/*` or `release/*` (enforced by the `source-branch` check).
- Use a **merge commit** for `develop -> main` PRs so the two branches don't drift. Squash or rebase is fine for `feature/* -> develop`.
- After a hotfix lands on `main`, open a back-merge PR `main -> develop`.
