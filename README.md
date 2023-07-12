# Penalty-Weighted MME
This is a repository for all the data and code I used for my final project in my Climate Prediction, Modeling, and Scenarios course (GEOG 652). The aim of this project was to provide a proof-of-concept for a Multi-Modal Ensemble (MME) whose weights are found by optimizing for a loss function that accounts for both individual model performance and model interdependence. Below I will summarize the motivation, results and interpretation of this project, alongside the steps needed to install the project if you are interested in running the optimizer itself.

## Installation

If you wish to run your own version of the penalty-weighted MME, please follow the steps below. Ensure that Anaconda is installed, and that you're in your desired directory!

```
git clone https://github.com/svladusic/penalty-weighted-MME.git
conda env create
conda activate pen_weighted_env
python src/main.py
```

## Project Explanation

I will soon write a blog post providing the background, motivation, results and analysis of this project. The blogpost will be based around a written report for this project, but with accessibility and outreach in mind.

## Notes

- All data required to replicate project results can be found in the data folder of this project.
- I am largely happy with the state of this project. While I will spend a good amount of effort documenting the scripts used in this project, I will not refactor or expand the functionality of the project unless my priorities suddenly change. However, I do think there are many interesting ways the project could be expanded. If for whatever reason you are interested in contributing to the project, please let me know!
