# BNP2

## Quick start

1. Install Julia
2. Open Julia REPL via `julia` in the directory of this repo
3. Install a few global packages by entering the package mode via `]` and do
   1. `add IJulia`
   2. `add DrWatson`
4. Install dependencies
   1. ESC or backspace and do
      1. `using DrWatson`
      2. `@quickactivate`
   2. Instantiate all dependencies via `] instantiate`
5. Link PyCall to Python virtual environment
   1. ESC or backspace and do
      1. `ENV["PYTHON"] = "$YOUR_PYTHON_PATH"`
         - Here `YOUR_PYTHON_PATH` is the Python environment that you installed W&B
      2. `] build PyCall`
6. Play with notebooks as usual

### How to find your Python path

1. Activate your virtual environment
2. Do `which python`
