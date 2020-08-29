# backprop
a failed attempt at a general backpropagation library, operating on arbitrary node functions.

This library attempts to mirror the discussion of  these notes on backpropagation examples https://www.redwrasse.io/supplementals/backpropexamples.html .

Lesson: Lack of static types makes Python a poor choice.
Nodes are general maps `R^n -> R^m`, they tend to operate on batches, with some performing reductions over batches, and nodes can have multiple child nodes. This complexity needs static types.

`example1.py` does work, while `linear_reg.py` is a massive headache try to match up Jacobian matrix multiplications generally for per-batch nodes and batch reduction nodes.
