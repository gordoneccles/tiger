# Tiger Language
Tiger is a simple, statically typed programming language.
See the 4-page specification [here](https://cs.nyu.edu/courses/fall13/CSCI-GA.2130-001/tiger-spec.pdf)

# Usage
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
from tiger_interpreter import Tiger

prog = Tiger('your_tiger.program')
prog.execute()
```
