# BNP2

## Quick start

1. Install Julia
2. Open Julia REPL via `julia` in the directory of this repo
   1. Install DrWatson.jl via `] add DrWatson`
   2. ESC or backspace and do
      1. `using DrWatson`
      2. `@quickactivate`
   3. Instantiate all dependencies via `] instantiate`
   4. ESC or backspace and do
      1. `ENV["PYTHON"] = "$YOUR_PYTHON_PATH"`
         - Here `YOUR_PYTHON_PATH` is the Python environment that you installed W&B
      2. `] build PyCall`
3. Play with notebooks as usual

### How to find your Python path

1. Activate your virtual environment
2. Do `which python`