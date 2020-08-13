# backprop
a failed attempt at a general backpropagation library.

Lesson: Lack of static types makes Python a poor choice.
Nodes are general maps `R^n -> R^m`, they tend to operate on batches, with some performing reductions over batches, and nodes can have multiple child nodes. This complexity needs static types.

This library attempts to mirror the discussion of (ref. supplemental material link).
