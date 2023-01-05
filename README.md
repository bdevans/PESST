# PESST

Protein Evolution Simulator with Stability Tracking

Tenets of the Survivor Bias Hypothesis
--------------------------------------
1. Stability effects of mutations across a protein are normally distributed with a negative (destabilizing) mean.
2. The majority of proteins are marginally stable.
3. Contemporary proteins contain fewer significantly destabilizing amino acids than ancestral proteins.
4. The sequence space of contemporary proteins is positively biased for stabilizing mutations, despite its mutational landscape.

*PESST evolutionary algorithm pseudocode*

![algorithm](images/algorithm.png)

*Default parameters in PESST*

![parameters](images/parameters.png)

## Longer description

Axiom 1 refers only to the nature of the distribution of possible stabilizing effects throughout the protein. Importantly, Axiom 1 then drives Axiom 2, which in turn together drive Axioms 3 and 4. We took this same cascadic approach to the design of a model simulation framework to test these axioms, called Protein Evolution Simulation with Stability Tracking (PESST). By implementing the well evidenced properties defined by Axioms 1 and 2, the validity of subsequent Axioms can be explored. For the exploration of Axiom 1, PESST produces reconstructable, simulated phylogenetic information that has defined stability properties traceable over evolutionary time.

# PESST assumptions

- Proteins evolve in a fixed population size.
- The population derives from a single clone.
- The population evolves according to a uniform clock.
- Time is arbitrary - each generation leads to a fixed number of mutations across the population.
- Protein length does not change over the course of evolution.
- Changes in amino acid sequence only occur by substitution.
- Every amino acid position has an impact on the T_m of its protein.
- Each possible amino acid at every position has a fixed and unique ∆T_m.
- At a given position in a protein, each possible amino acid has a fixed ∆T_m value that remains equivalent for all proteins in the population.
- The distribution of ∆T_m values held by each possible amino acid at each position in the protein sequence is Gaussian.
- ∆T_m values are not epistatic.
- Protein fitness is binary (fit|unfit).
- A protein’s fitness is only derived from its stability in relation to a stability threshold
- A protein is fit until its stability is below the threshold, then it is unfit.
- The penalty for an unfit protein is always death.
- Proteins can duplicate by asexual reproduction, which occurs immediately when there is space in the population to fill.
- As all sequences are equally fit, the protein that undergoes asexual reproduction to fill the space in the population is random.
- Sequences can die randomly.
- A population can bifurcate into even sized sub-populations only.
- Bifurcation occurs at even time periods across the course of evolution.


# A simplified version is as follows:

## Initiate a starting sequence
- A protein Initiate a starting sequence (η_initial) of user definable length R is formed, containing a start methionine followed by randomly generated amino acids.
- Each position (r) can contain one of 20 amino acids (a). From a user definable normal distribution N of mean μ and shape σ^2, the model randomly generates a 2D matrix "∆" , where ∆_(r,a) describes a ∆T value of a given amino acid at a given position. The stability (T) of η is given by ∑_(r=1)^R▒∆_(r,a_r ).
- A user defined stability threshold is imposed on the dataset
- Rate variation is defined at every site based on a gamma distribution (Supplementary Figure 2).
- A user defined proportion of sites are fixed as invariant.
- Amino acids are modified until the T_m of the protein satisfies a user defined starting stability – this is Generation 0 for evolution.

## Evolving sequences
- The starting protein is cloned to a user defined population size.
- The population evolves under a uniform clock according the LG model of amino acid substitution (Le and Gascuel 2008; Supplementary Figure 3), with each site changing at a rate defined by the gamma distribution and invariant sites.
- During evolution the model continually tracks changes in stability at both the amino acid (∆T_m) and protein (T_m) level. If a protein drops below the stability threshold it is killed and immediately replaced by another clone.
- During evolution the population bifurcates into independent sub-populations at set time periods. Sub-populations undergo sequence replacement in populous (supplementary figure 4)
- If the user desires, evolution can progress where clones have a user defined probability of dying at each generation to ensure duplication within the population.



## Initiating a starting sequence
- A protein ($\eta_{initial}$) of user definable length $R$ is formed, containing a  start methionine followed by randomly selected amino acids.
- Each position ($r$) can contain one of 20 amino acids ($a$). From a user definable normal distribution $N$ of mean $\mu$ and variance $\sigma^2$, the model randomly generates a 2D matrix $\bm{\Delta}$, where $\Delta_{r,a}$ describes a $\Delta T$ value of a given amino acid at a given position. The stability ($T$) of $\eta$ is given by $T = \sum_{r=1}^{R} \Delta_{r,a_r}$, where $\Delta_{r,a}$ is the change in thermodynamic stability conferred by amino acid $a$ at location $r$.
- In nature, sites become fixed in a population if they are essential for function despite possible detrimental $\Delta T$ values. Therefore, to account for this behaviour, the model defines invariant sites to a proportion of the amino acids in the protein ($p_{invariant}$).
- The user sets a stability threshold ($\Omega$), such that $-\infty \lt \Omega \lt T_0$.
- Natural sequences exhibit rate variation across sites. Rate variation can be modelled to a gamma distribution ($\Gamma$) with four independent rate categories (Yang 1994).
- Independent rate categories are generated each run by taking the median value of four quartiles of 10,000 samples from a gamma distribution of a user defined shape ($\kappa$) and scale. Typically a scale of $\frac{1}{\kappa}$ is used. Each variant position is randomly assigned to one of four rate categories, defining a matrix of site-wise mutation probabilities $m$ (where $m_r\sim \Gamma(\kappa, \theta); \sum_{r=1}^R m_r = 1$), which remains constant throughout the simulation.
- The user sets one of five possible initial $T$ values (low, medium, marginal, [bounded range], high) that modifies $\eta$ into the sequence that will be used for evolution.
  - $T_0^{low}$ and $T_0^{high}$ are treated in the following manner: every site where residues are not fixed is swapped for another amino acid chosen randomly from a pool of the three largest or smallest values of $\Delta_{r,a}$.
  - $T_0^{medium}$ requires the user to input a $T$ range where $T_{min} \gt \Omega$ and $T_{min}$ and $T_{max}$ are between the minimum and maximum bounds of $T = \sum_{r=1}^{R} \Delta_{r,a_r}$. The model then modifies non-fixed residues until the first protein sequence is discovered that satisfies a value in the range by hill-climbing.

## Evolving a sequence
- Once a starting sequence $\eta$ of length $R$, with site-wise mutation probability $\bm{m}$, and a global fitness of $T_{max} \ge T \gt \Omega$ has been generated by the model, the sequence is cloned to generate a starting population ($\Phi$) of a user-defined size ($N$).
- The population evolves according to a uniform clock over a user-defined number of generations ($G$). At every generation, each amino acid undergoes mutation with a constant probability $p_m$, where $p_m\cdot R\cdot N$ defines the total number of mutations per generation. $p_m\cdot R\cdot N$ sites are selected to mutate at rates according to $\bm{m}$. A site with amino acid a transition to a new amino acid $a'$ based on the Le and Gascuel (LG) amino acid replacement matrix, $\bm{L}$ (Le and Gascuel 2008) that is modified so $a \ne a'$.
- A protein’s fitness is considered binary (fit|unfit). Proteins are considered unfit when $T \le \Omega$. Before each generation, the model checks for unfit sequences in $\Phi$. If this is satisfied, $\eta_{unfit}$ is deleted and replaced with another sequence in the population that satisfies $T \ge \Omega$.
- Evolution is simulated with population isolation to mimic bifurcations. In this instance, the model divides the global population into even sub-populations $\Phi_{roots}$ and $\Phi_{branches}$ where $\Phi_{branches}$ split at a bifurcation interval $g_B$ where $g_B = \lfloor \frac{G}{\lfloor \log_2(N-n_{roots})-\log_2(3)+1) \rfloor} \rfloor$. Isolation events occur at equal time-points such that every final population at the end of the run contains 3, 4 or 5 individuals. When an individual in a sub-population dies, it can then only be replaced *in populous*, generating independent lineages. An edge case in this factor required a feature in the model that diverges significantly from nature. If every sequence in a subpopulation of $\Phi$ satisfies $T \lt \Omega$ the entire subpopulation goes extinct. Therefore, the simulation reverts to the prior generation to re-attempt mutating sequences to avoid complete branch extinction.
- If the user desires, evolution can be run assuming death happens naturally in the population, aside from being outcompeted due to fitness. At every generation, each member of $\Phi$ has a user defined probability of dying ($p_{death}$). As before, dead individuals are immediately replaced by other individuals *in populous*. This allows for evolution that occurs without replacement caused by fitness to produce a phylogeny that is not a star-phylogeny.

## Outputs
The model is able to track and output a variety of useful data about the population’s evolution:
- At a user defined generation rate, the model can output FASTA files describing the sequences of $\Phi$.
- A scatter plot describing the change in $T$ of every sequence in $\Phi$ over time.
- At a user defined generation rate, the model will output data, graphs and animations describing every $\Delta_{r,a}$ of each amino acid within $\Phi$ at a given generation compared to $\Delta_{r,a}$ values stored in $\bm{\Delta}$. 
- At a user defined generation rate, the model will output data on the distribution of $\Delta T$ values within $\Phi$ ($\Delta_\Phi$), including the Anderson-Darling, Skewness-Kurtosis all, and 2-sided Kolmogrov-Smirnoff statistical tests for normality of the data.
