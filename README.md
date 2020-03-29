# The Tiger Programming Language
Tiger is a simple, statically typed programming language.
See the 4-page specification [here](https://cs.nyu.edu/courses/fall13/CSCI-GA.2130-001/tiger-spec.pdf)

# Setup
First be sure you've created the virtual environment:
```sh
virtualenv --python=python3 venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

# Usage
Thus far the project only builds an intermediate representation of the abstract syntax tree in Python.
TODO:
- Type checking
- Embedded Python interpreter

Given a file named `hello_world.ti`:
```
/* Hello-world with function */

let
    function hello() = print("Hello, World!\n")
in
    hello()
end
```

## As a python module
```python
from tiger_pl import Tiger

prog = Tiger('hello_world.ti')
print(prog.execute())
```

## As a script
```sh
python -m tiger_pl hello_world.ti
```

# Running Tests
```sh
source venv/bin/activate
pytest tests/
```
