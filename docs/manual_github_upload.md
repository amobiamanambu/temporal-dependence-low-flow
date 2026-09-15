# Uploading without terminal commands

## Recommended: GitHub Desktop

GitHub Desktop is the cleanest manual method because this repository contains
more than 100 files, the current limit for one browser upload.

1. Install and open [GitHub Desktop](https://desktop.github.com/), then sign in.
2. Select **File → Add Local Repository**.
3. Choose the folder named `temporal-dependence-low-flow`.
4. When GitHub Desktop says that the folder is not yet a Git repository, select
   **create a repository here**.
5. Keep the repository name `temporal-dependence-low-flow`. Do not add another
   README, license, or `.gitignore`; they are already present.
6. Review the file list in the **Changes** pane. Confirm that no raw NetCDF,
   shapefile, SQLite database, manuscript backup, or API key is listed.
7. Enter the summary `Initial reproducible research release` and select
   **Commit to main**.
8. Select **Publish repository**.
9. Choose the correct GitHub account, add the description “Code and example
   data for probabilistic low-flow forecasts with temporally coherent ensemble
   trajectories,” and uncheck **Keep this code private** when ready to publish.
10. Select **Publish repository** again.
11. Choose **View on GitHub** and confirm that the README, example figures,
    license, citation file, and directory links render correctly.
12. Open the **Actions** tab and confirm that the automated test run passes.

## Browser-only alternative

1. On GitHub, use the upper-right **+** menu and select **New repository**.
2. Name it `temporal-dependence-low-flow` and choose its visibility.
3. Do not initialize it with a README, `.gitignore`, or license.
4. Select **Create repository**, then **uploading an existing file**.
5. Upload the repository contents in groups of at most 100 files. On macOS,
   press **Command–Shift–.** in Finder to reveal `.github`, `.gitignore`, and
   `.gitattributes` before selecting files.
6. Preserve the directory structure, add a clear commit message, and select
   **Commit changes** after each group.

The browser accepts files only up to 25 MiB each. Every file in this prepared
repository is below that limit. Large continental forecast tables should be
placed in a DOI-issuing research-data repository, not uploaded to GitHub.
