# Swim_Standard
An interactive dashboard allowing swimming instructors to see how their students' timings compare to Indian national records, keep track of gaps, and predict pace curves. Includes suggestions for training, analysis of records, and a basic ML-based model for predicting pace.  

Features  
* Compare: Input a student's time and see how it stacks up against the Indian record.
* Exercises: View stroke-specific training suggestions
* Records: Explore national records by gender and category.
* Analytics: Visualize record speed vs. distance across strokes.
* Pace Predictions: Employed a power-law model to predict student times at multiple distances.

Dependencies
Built with Conda (Python 3.12). Required packages:
* numpy – numeric calculations + pace curve fit
* pandas – CSV parsing and cleaning
* panel – dashboard framework
* plotly – interactive charts
* (optional) tqdm, pytest, pytest-cov, pillow

Installation
Using Conda (recommended):
“conda create -n swim-standard python=3.12 numpy pandas panel plotly tqdm pytest pytest-cov pillow -c conda-forge -y”

Data Files
Keep these in the same folder as Swim_Standard.py:
* sfi_records.csv (required) – Indian records
* swim_exercises.csv (optional) – Training map

Supported time formats: mm:ss.xx, mm.ss.cs, ss.xx


