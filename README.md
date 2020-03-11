# Tiger Language
Tiger is a simple, statically typed programming language.
See the 4-page specification [here](https://cs.nyu.edu/courses/fall13/CSCI-GA.2130-001/tiger-spec.pdf)

# Usage
Thus far the project only builds an intermediate representation of the abstract syntax tree in Python.
```
/*  your_tiger.program */

/* Hello-world with function */

let
    function hello() = print("Hello, World!\n")
in
    hello()
end
```
```python
from tiger_pl import Tiger

prog = Tiger('your_tiger.program')
print(prog.execute())  # returns the python representation
```

# TODO
- Type checking
- Embedded Python interpreter

# Running Tests
First be sure you've created the virtual environment:
```sh
virtualenv --python=python3 venv
source venv/bin/activate
pip install -r requirements.txt

```
Then run the test suite:
```sh
pytest tests/
```
