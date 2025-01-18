# Diff2Tune

This step follow [Inverting Gradients - How easy is it to break privacy in federated learning?](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html)

Main idea is: assume we have $M_{benign}$ and $M_{backdoor}$, then:
$$
\Delta \Theta = M_{benign} - M_{backdoor} \\
\theta^\prime = \theta - \eta \nabla_{\theta} L(x,y) \rightarrow \theta - \theta^\prime = \eta \nabla_{\theta} L(x,y)
$$
We want find $D$ such that $M_{benign}$ fine tune on it get $M_{ft}$ which be similar to $M_{backdoor}$

Hence is to solve:
$$
\argmin_{x\in\left[ 0, 1 \right]} \{ 1 - \frac{< \eta \nabla_{\theta} L(x,y), \Delta \Theta>}{||\eta \nabla_{\theta} L(x,y)||||\Delta \Theta||}\} + \alpha TV(x)
$$

## Discussion

### 1 

[DLG](https://arxiv.org/abs/1906.08935) has idea that optimize:
$$
\argmin || \nabla_\theta L(x, y) - \nabla_\theta L(x^\prime, y) ||^2
$$

What we do is $\argmin || M_{ft} - M_{backdoor} ||^2$, but note that:
$$
\begin{align}
& \argmin || M_{ft} - M_{backdoor} ||^2 \notag \\
=& \argmin ||(M_{benign} - M_{ft}) - (M_{benign} - M_{backdoor}) ||^2 \notag \\
=& \argmin || \eta \nabla_\theta L(x, y) - \eta^\prime \nabla_\theta L(x^\prime, y) ||^2 \notag
\end{align}
$$
and if we further **assume $\eta = \eta^\prime$ (fine tune and trigger injection has same learning rate)**, this two are the same.

### 2

Decide to do DLG first. Change code:
```
tp = transforms.ToTensor()
```
to
```
tp = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])
```
Solve the problem that loss suddenly comes from normal to 100+ and stuck at that graph.
