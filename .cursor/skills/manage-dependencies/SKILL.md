---
name: manage-dependencies
description: Add, update, or remove individual Python packages in requirements.txt for the SIOP 2026 Master Tutorial repository, check version compatibility, and sync the dependencies table in REPO_ONBOARDING.md.
---

# Manage Dependencies

This skill handles targeted dependency changes in requirements.txt — adding a new package with appropriate version constraints, updating an existing package version, or removing an unused one. It maintains the clean, curated requirements.txt approach (not a `pip freeze` dump) and keeps REPO_ONBOARDING.md in sync.

## When to Use

- Use this skill when adding a single new package to the codebase
- This skill is helpful for updating an existing package version and checking compatibility
- Use when syncing the dependencies table in REPO_ONBOARDING.md after any requirements change
- This skill is helpful for resolving version conflicts between packages
- Use when a package version is causing issues in the workshop or Colab environment
- This skill is helpful for removing a package that is no longer used

## Instructions

### Step 1: Understand the Change Context

Before modifying requirements.txt, determine:
- **What package is changing?** (new addition, version bump, removal)
- **Why is this change needed?** (new feature, bug fix, compatibility issue, Colab failure)
- **Is this a major version change?** (e.g., 0.14.x → 0.15.x or 1.x → 2.x)

Use the ask questions tool if:
- The reason for the change is unclear
- A major version bump is proposed (may have breaking changes)
- Multiple packages need coordinated updates

### Step 2: Check Domain-Specific Compatibility

The causal inference codebase has critical interdependencies. Before making changes, verify compatibility:

**Core Causal Inference Stack:**
- `statsmodels` (GEE, propensity score models, VIF)
- `econml` (Double Machine Learning)
- `lifelines` (Cox PH, Kaplan-Meier survival analysis)
- `scikit-learn` (Random Forest for diagnostics and DML nuisance models)

**Known Interdependencies:**
- `econml >= 0.15.0` requires `scikit-learn >= 1.0.0`
- `statsmodels 0.14.x` works with `scipy 1.11.x - 1.13.x`
- `lifelines >= 0.27.0` requires `pandas >= 1.0.0`

Check the package changelog for breaking changes before bumping a version. If updating `econml`, `statsmodels`, or `scikit-learn`, treat it as high-risk and validate thoroughly.

### Step 3: Version Constraint Strategy

This repo uses a **clean, curated requirements.txt** — not a `pip freeze` dump. Only direct dependencies are listed. Keep it that way.

**Use minimum version constraints** (e.g., `pandas>=2.2.2`) to allow Colab and other environments to resolve compatible versions without being over-pinned:

```
# Good — allows compatible resolution
pandas>=2.2.2

# Avoid unless there is a specific known breakage above a version
pandas==2.2.2
```

Use exact pinning (`==`) only when a specific version is known to be required for compatibility or when a higher version is known to break something.

### Step 4: Make the Change

**Adding a New Package:**
1. Determine the minimum required version that provides the needed functionality
2. Add to requirements.txt under the appropriate group with a comment explaining its purpose if non-obvious:
   ```
   # Excel export with formatting
   openpyxl>=3.1.5
   ```
3. Maintain the existing grouping structure:
   - Core data / numerical
   - Statistical modelling
   - Visualization
   - Excel export

**Updating an Existing Package:**
1. Check the changelog for breaking changes
2. Update the version constraint in requirements.txt
3. Add an inline comment if it is a significant change

**Removing a Package:**
1. Verify the package is truly unused — search the codebase for imports before removing
2. Remove from requirements.txt
3. Remove from the REPO_ONBOARDING.md dependencies table

### Step 5: Validate the Specific Change

Test that the targeted change does not break functionality. Focus validation on what the changed package actually touches:

**For statsmodels changes:** Verify GEE models fit, propensity score estimation runs, VIF checks work

**For econml changes:** Verify Linear DML and Causal Forest DML complete without errors

**For lifelines changes:** Verify Cox PH fits, Kaplan-Meier curves render

**For scikit-learn changes:** Verify Random Forest diagnostics and DML nuisance models run

**For matplotlib/seaborn changes:** Verify propensity overlap plots, weight distribution plots, and KM curves render

**For openpyxl changes:** Verify Excel export produces correctly formatted output

**Minimum validation for any change:**
```bash
# Install from updated requirements.txt in a clean environment
pip install -r requirements.txt

# Verify the affected imports load without error
python -c "import <changed_package>"

# Run the relevant notebook section
```

### Step 6: Sync REPO_ONBOARDING.md

After any change to requirements.txt, update the **Key Dependencies** table in REPO_ONBOARDING.md (Section 3):

- **Add** new packages with version and purpose
- **Update** version numbers for changed packages
- **Remove** deleted packages from the table
- Maintain the existing table format and column order:

```markdown
| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | >=2.2.2 | Data manipulation |
```

### Step 7: When to Ask for Clarification

Use the ask questions tool if:

- A major version bump is proposed: *"Package X has a major version update. Should I proceed? This may have breaking changes."*
- There is a dependency conflict: *"Package X requires scipy >= 1.14, but statsmodels requires scipy <= 1.13. How should I resolve this?"*
- The scope is unclear: *"Should I update all causal inference packages together or just the one requested?"*
- A Colab-specific failure is reported: *"Can you share the exact error message from Colab so I can identify the conflicting version?"*

### Best Practices

- **Keep requirements.txt curated** — only direct dependencies, not transitive ones. Never replace it with `pip freeze` output.
- **Prefer minimum version constraints** (`>=`) over exact pins (`==`) unless there is a specific known breakage
- **One change at a time** — update one package, validate, then move to the next
- **Stability over cutting edge** — this is teaching material; avoid updates close to workshop dates unless critical
- **Always sync REPO_ONBOARDING.md** — the dependencies table should always reflect the current requirements.txt