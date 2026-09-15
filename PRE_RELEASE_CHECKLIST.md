# Pre-release checklist

Complete these items immediately before making the repository public:

- [ ] Confirm the author name and email in `CITATION.cff`.
- [ ] Confirm acceptance of the MIT code license and CC BY 4.0 example-data license.
- [ ] Upload with GitHub Desktop and confirm that ignored files are absent.
- [ ] Confirm the GitHub Actions test passes.
- [x] Add the final GitHub repository URL to the README and manuscript.
- [ ] Create a numbered GitHub release, beginning with `v0.1.0` for a
  pre-submission release or `v1.0.0` for the frozen publication release.
- [ ] Archive the numbered release in Zenodo, Mendeley Data, or another
  DOI-issuing repository.
- [x] Add the archival DOI to `CITATION.cff`, the README, and the manuscript's
  data-availability statement.
- [ ] Deposit the validated full derived-data package separately; do not upload
  the 200+ MB forecast Parquet files through GitHub's browser interface.
- [x] Include 105/120-day supporting outputs or a clear derivation path because
  those windows appear in the paper but are not in forecast-data release v1.0.0.
