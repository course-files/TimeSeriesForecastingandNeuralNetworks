# Cleanup Instructions

You can either:

- Wait until the end of the semester after the marks have been officially released to implement the cleanup in all the labs.
- Implement the cleanup once you are done with the lab **and you have pushed your commits to GitHub for grading**

The cleanup involves deleting the following from your development machine to save on space:

## Python Virtual Environment

- `.venv` (TensorFlow alone takes more than a gigabyte)

## Files Created by the Notebooks

- Everything in the `outputs` folder (except `.gitkeep`)
- `data/forecast_basic.csv` and `data/forecast_backtest.csv`, after you have submitted them

## Caches

- The `__pycache__` folders inside the project
- The `.pytest_cache` folder
- The pip cache, if you need the space: `pip cache purge`
