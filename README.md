Louvain4ppm
===========================
 
We make use of the Louvain heuritic[1] for maximising modularity[2] to find the maximum a posteriori (MAP) soluton of the ***uniform planted partition model*** (PPM)[3].

The modularity maximisation approach suffers from the **overfitting problem**. Specifically, it returns spurious communities in networks that are known to have no community structure. In comparsion, the uniform PPM is a more robust alternative as it takes statistical significance of its results into account. 

For example, the following code implements the Louvain algorithm to obtain the community structure in a Erdos-Renyi (ER) graph. In ER graphs, edges are randomly placed between nodes with the same probability, therefore it is accepted that there is no community structures in ER graphs. In our example, modularity maximisation claims there exist 10 groups with a modularity value Q = 0.461, while the uniform PPM correctly assigns all of the nodes into a single group with Q = 0[^1]. 

```python
from community import community_louvain
import networkx as nx

# generate an Erdos-Reny graph with 200 nodes
nr = 100; 
B = 2; 
N = nr * B; 

ep = 0.0                 # strength of assortative structute takes values in [0,1]
k = 5                    # average degree
p_in = (1 + ep)*k/N      # prob. of an within-group edge
p_out= (1 - ep)*k/N      # prob. of an between-group edge

G = nx.planted_partition_graph(B, nr, p_in, p_out)

# modularity maximisation
b_modularity = community_louvain.best_partition(G)
B_modularity = max(b_modularity.values())+1
Q_modularity = community_louvain.modularity(b_modularity, G)

# uniform PPM
b_ppm = community_louvain.best_partition_ppm(G)
B_ppm = max(b_ppm.values())+1
Q_ppm = community_louvain.modularity(b_ppm, G)

print(f"Modularity maximisation finds {B_modularity} communities with modularity value Q = {Q_modularity}", "\n")
print(f"Uniofrm PPM finds {B_ppm} communities with modularity value Q = {Q_ppm}")

```

| Modularity maximisation | uniform PPM |
:-------------------------:|:-------------------------:
<img src="/pics/synthetic_random_modularity_B_10_Q0.461-1.png" width=300><br> | <img src="/pics/synthetic_random_ppm__B_1_Q0.0.png" width=300><br>
B = 10, Q = 0.461 | B = 1, Q = 0

<br>
With the overfitting behaviour of modularity maximisaion in mind, we invite users of Louvain or any other modularity-based algorithms to re-examine their results as follows:
<br><br>

```python
# modularity maximisation
b_modularity = community_louvain.best_partition(G)      # G is the network to be analysed

b_ppm = community_louvain.best_partition_ppm(partition = b_modularity, G)
```
<br>

What the last two lines of code do is to run the Louvain algorithm for the uniform PPM with the starting point being the partition given by the modularity maximisation[^2]. We do such comparison in a network with two assortative groups[^3]. It is clear that modularity maximisation exaggerates the true structure, while running the Louvain for PPM leads to a partition that is almost identical to the underlying truth.<br><br>

| Modularity maximisation | uniform PPM |
:-------------------------:|:-------------------------:
<img src="/pics/synthetic_modularity_B_9_Q0.512-1.png" width=300><br> | <img src="/pics/synthetic_ppm_B_2_Q0.428_overlap0.355-1.png" width=300><br>
B = 9, Q = 0.512, accuracy = 0.355 | B = 2, Q = 0.428, accuracy = 0.965

<br><br>
Besides overfitting, modularity maximisation paradoxically has the problem of **underfitting**. The underfitting problem of modularity maximisaion leads to a **resolution limit** [4] on the number of detectable communities. 

The resolution limit of modularity maximisation is often demonstrated in the ring-of-cliques example. A clique is a fully connected subgraph. Consider a network consisting 24 cliques of size 5. These cliques are ordered and an edge is placed between every pair of neighbouring cliques to form a ring of cliques. Modularity maximisaion counter-intuitively favors a partition that merges every two neighbouring cliques together. In comparison, the uniform PPM has no resolution limit [3] and is able to identify all 24 cliques. 

```python

B = 24.                        # number of cliques
nr = 5                         # size of each clique
G = nx.ring_of_cliques(B,nr)

# modularity maximisation

b_modularity = community_louvain.best_partition(G)
B_modularity = max(b_modularity.values())+1
Q_modularity = community_louvain.modularity(b_modularity, G)

# uniform PPM

b_ppm = community_louvain.best_partition_ppm(G)
B_ppm = max(b_ppm.values())+1
Q_ppm = community_louvain.modularity(b_ppm, G)

print(f"Modularity maximisation finds {B_modularity} communities with modularity value Q = {Q_modularity}", "\n")
print(f"Uniofrm PPM finds {B_ppm} communities with modularity value Q = {Q_ppm}")

```
| Modularity maximisation | uniform PPM |
:-------------------------:|:-------------------------:
<img src="/pics/ring_of_clique_modularity-1.png" width=300><br> | <img src="/pics/ring_of_clique_ppm-1.png" width=300><br>
B = 12, Q = 0.8712| B = 24, Q = 0.8674


Further Explanations
----------

#### References:
<p><a>[1] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, <em>Fast unfolding of communities in large networks</em>, J. Stat. Mech.: Theory Exp. (2008) P10008. </a>
<p><a>[2] M. E. J. Newman, <em>Modularity and community structure in networks</em>, Proc. Natl. Acad. Sci. USA 103, 8577 (2006). </a>
<p><a>[3] L. Zhang, T.P. Peixoto, <em>Statistical inference of assortative community structures</em>, Phys. Rev. Res. 2 (2020) 043271.</a>
<p><a>[4] S. Fortunato and M. Barthélemy, <em>Resolution limit in community detection</em>, Proc. Natl. Acad. Sci. USA 104, 36 (2007).</a>
    
[^1]: Visuialisations are done with the [graph-tool library](https://graph-tool.skewed.de/static/doc/draw.html).
[^2]: Louvrain for modularity maximisaion usually starts with a partition that every node is put in its own group. This partition is a basin in the space of the posterior probability of the uniform PPM, therefore is not a good starting point if we want to apply Louvain for the uniform PPM. The defaul initial partition for the function `best_partition_ppm` is a partition that randomly assign each node to one of {1,2,..,N} possible groups, where N is the total number of nodes. In comparison.
[^3]: The model used to generate the syntehtic network is the uniform planted parittion model. The likelihood function of this model turns out to be equivalent to the modularity measure under certain choice of model parameters. The example used in the text was generated with the assortativity strength parameter `ep` being set to 0.85.
