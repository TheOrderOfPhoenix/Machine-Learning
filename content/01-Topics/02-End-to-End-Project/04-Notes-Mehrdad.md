
# End-To-End
## What the objective is
## Pipelines
A sequence of data processing components is called a data pipeline. Pipelines are very
common in Machine Learning systems, since there is a lot of data to manipulate and
many data transformations to apply.
Components typically run asynchronously. Each component pulls in a large amount
of data, processes it, and spits out the result in another data store. Then, some time
later, the next component in the pipeline pulls this data and spits out its own output.
Each component is fairly self-contained: the interface between components is simply
the data store. This makes the system simple to grasp (with the help of a data flow
graph), and different teams can focus on different components. Moreover, if a com‐
ponent breaks down, the downstream components can often continue to run nor‐
mally (at least for a while) by just using the last output from the broken component.
This makes the architecture quite robust.
On the other hand, a broken component can go unnoticed for some time if proper
monitoring is not implemented. The data gets stale and the overall system’s perfor‐
mance drops.
## Looking at current solutions 

First,
you need to frame the problem: is it supervised, unsupervised, or Reinforcement
Learning? Is it a classification task, a regression task, or something else? Should you
use batch learning or online learning techniques?

## Select a Performance Measure

![[Pasted image 20250723211935.png]]

Did not understand the symbols...
![[Pasted image 20250723212219.png]]

## 5.  Check the Assumptions
