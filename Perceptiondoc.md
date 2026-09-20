# Neural perception and probabilistic coding

**Comprehensive bibliography: 80 unique references — 78 papers/reviews/preprints, 1 book chapter, and 1 correction.** Updated 19 September 2026.

This master list consolidates every distinct paper and chapter named anywhere in this document: the original merged bibliography, the SDT/attention/perceptual-learning discussion, the Pouget–Harper–McAlpine discussion, and the latest discussion of attention models and the unknown decoder. Each reference has a link and brief summary. Data and code links are retained with their associated papers.

**Organization:** papers are grouped by their main teaching purpose. Reference numbers are stable identifiers: entries 1–45 retain their earlier numbers so the preserved discussions remain usable. Numbers therefore do not run consecutively within every topic. Each paper appears once in the master bibliography, even when it is discussed in several places.

The full questions and answers are preserved after the bibliography. Their references to an earlier “40” or “45” describe the collection at that stage; the current total is 80. Topic placement indicates conceptual relevance, not a claim that every paper directly extends Jazayeri and Movshon or demonstrates an attentional effect.

**Choosing two papers for the next class:** [Five groups of five recommendations and suggested pairings](#recommended-reading-selection).

## Topic guide

| Group | References | Main question |
|---|---:|---|
| [1. Likelihoods, population codes, and empirical uncertainty](#likelihoods) | 8 | How can population activity carry likelihoods and uncertainty? |
| [2. Neural sampling and interpretation of probability codes](#probability-frameworks) | 5 | What does representing a probability distribution mean? |
| [3. SDT, neural sensitivity, and components of attention](#sdt) | 6 | Does attention change sensitivity, criterion, or both? |
| [4. Tuning width, informative slopes, and tuning placement](#tuning-theory) | 6 | When do gain, sharpening, or displaced tuning improve discrimination? |
| [5. Attention measured through gain, selectivity, and receptive fields](#attention-tuning) | 13 | Which tuning parameters does attention actually change? |
| [6. Computational mechanisms of attention and task performance](#attention-models) | 4 | Can a specified attentional mechanism improve complete task performance? |
| [7. Attention, covariance, cortical state, and causal perturbation](#attention-covariance) | 6 | Which changes in shared variability matter for behavior? |
| [8. Population readout, choice signals, and the unknown decoder](#readout) | 9 | How does available information relate to the readout the animal uses? |
| [9. Perceptual learning: tuning, maps, correlations, and readout](#learning) | 7 | What changes with discrimination training? |
| [10. Adaptation and redistribution of coding precision](#adaptation) | 2 | How do statistics and decoder assumptions alter perception? |
| [11. Motion context, surround effects, and causal inference](#motion-context) | 3 | How do context and inferred causes influence motion coding? |
| [12. Broad reviews connecting attention, coding, and behavior](#reviews) | 6 | Which reviews provide the broader framework? |
| [13. Choice, reward, action, and reusable neural datasets](#datasets) | 5 | Which related datasets support neural–behavioral analyses? |

## Suggested reading routes

These routes are teaching suggestions. Numbers link to the full entries below.

| Goal | Suggested sequence |
|---|---|
| Connect the two Jazayeri–Movshon papers | Nature Neuroscience 2006 ([1](#ref-1)) → Nature 2007 ([80](#ref-80)) |
| Move from likelihoods to empirical uncertainty | Jazayeri–Movshon ([1](#ref-1)) → Ma ([2](#ref-2)) → Beck ([3](#ref-3)) → Walker ([6](#ref-6)) → Geurts ([7](#ref-7)) |
| Introduce SDT and attention | Britten 1992 ([17](#ref-17)) → Luo–Maunsell 2015 ([47](#ref-47)) → Chandrasekaran 2024 ([55](#ref-55)); Maunsell review ([63](#ref-63)) for context |
| Teach gain, sharpening, and informative slopes | Pouget 1999 ([41](#ref-41)) → Butts–Goldman ([48](#ref-48)) → Harper–McAlpine Figure 2 ([43](#ref-43)) → Seriès 2004 ([42](#ref-42)) |
| Connect coding theory to measured attentional gain | McAdams–Maunsell ([25](#ref-25)) → Ling ([46](#ref-46)) → Scolari 2012 ([49](#ref-49)) → Kozyrev ([74](#ref-74)) |
| Ask whether tuning changes cause better performance | Lindsay–Miller ([72](#ref-72)) → Fox ([54](#ref-54)) → Abdirashid ([77](#ref-77)) and Tünçok ([78](#ref-78)); Salehi ([79](#ref-79)) for recurrent models |
| Make decoder uncertainty explicit | Ruff–Cohen 2019 ([75](#ref-75)) → Ni 2022 ([76](#ref-76)) → Nienborg–Cumming ([19](#ref-19)); Lindsay–Miller ([72](#ref-72)) supplies a model with a known computation |
| Study covariance and causal evidence | Cohen–Maunsell ([28](#ref-28), [29](#ref-29)) → Ruff–Cohen 2014 ([71](#ref-71)) → Nandy ([30](#ref-30)) with correction ([31](#ref-31)) → Shi ([32](#ref-32)) |
| Compare learning mechanisms | Schoups ([56](#ref-56)) and Raiguel ([57](#ref-57)) → Recanzone ([59](#ref-59)) and Polley ([60](#ref-60)) → Law–Gold ([61](#ref-61)) → Gu ([62](#ref-62)) → Liu 2026 ([24](#ref-24)) |
| Explore adaptation and decoder mismatch | Dean–Harper–McAlpine ([44](#ref-44)) → Seriès–Stocker–Simoncelli ([35](#ref-35)) |
| Compare probability-code frameworks | Haefner 2016 ([8](#ref-8)) → Echeveste ([9](#ref-9)) → Lange ([10](#ref-10)) → Walker review ([11](#ref-11)) → Haefner synthesis ([12](#ref-12)) |
| Explore generative and causal-inference models | Shivkumar ([14](#ref-14)) → Lengyel ([15](#ref-15)) → Liu ([24](#ref-24)) |
| Connect spikes to choices | Britten ([17](#ref-17), [18](#ref-18)) → Nienborg–Cumming ([19](#ref-19)) → Pitkow ([20](#ref-20)) → Chicharro ([23](#ref-23)) |
| Use reward and eye-movement datasets | Larry ([38](#ref-38)) and London ([40](#ref-40)), with Stoll–Rudebeck ([39](#ref-39)) for interpretation |

## Master bibliography

<a id="likelihoods"></a>

## 1. Likelihoods, population codes, and empirical uncertainty

Start with the two Jazayeri–Movshon papers, then follow population likelihoods into Bayesian computation and empirical uncertainty decoding.

<a id="ref-1"></a>

**1. Jazayeri, M., & Movshon, J. A. (2006). [Optimal representation of sensory information by neural populations](https://www.nature.com/articles/nn1691). Nature Neuroscience.**

The starting point: under specified assumptions about neural response statistics, weighted population activity can represent a stimulus log likelihood. The framework connects tuning curves and readout weights to perceptual judgments, rather than reducing the population to a single best stimulus estimate.

<a id="ref-80"></a>

**80. Jazayeri, M., & Movshon, J. A. (2007). [A new perceptual illusion reveals mechanisms of sensory decoding](https://www.nature.com/articles/nature05739). Nature, 446, 912–915.**

Reports a direction-estimation bias following fine motion discrimination and explains it with a decoder that emphasizes neurons tuned away from the discrimination boundary. The same stimulus is not similarly misperceived in a coarse task. Connects task-dependent readout, informative tuning slopes, and perceptual bias; this is the Nature companion to entry 1 mentioned in the teaching question.

<a id="ref-2"></a>

**2. Ma, W. J., Beck, J. M., Latham, P. E., & Pouget, A. (2006). [Bayesian inference with probabilistic population codes](https://pubmed.ncbi.nlm.nih.gov/17057707/). Nature Neuroscience.**

Develops probabilistic population codes in which population responses carry both a stimulus estimate and its uncertainty. Shows how operations such as combining independent cues can implement Bayesian inference through relatively simple neural computations.

<a id="ref-3"></a>

**3. Beck, J. M., et al. (2008). [Probabilistic population codes for Bayesian decision making](https://pmc.ncbi.nlm.nih.gov/articles/PMC2742921/). Neuron.**

Extends population-code theory to decisions made from accumulating noisy evidence. Provides a bridge between probabilistic inference, neural integration, and activity associated with motion decisions in MT and LIP.

<a id="ref-4"></a>

**4. van Bergen, R. S., Ma, W. J., Pratte, M. S., & Jehee, J. F. M. (2015). [Sensory uncertainty decoded from visual cortex predicts behavior](https://pmc.ncbi.nlm.nih.gov/articles/PMC4670781/). Nature Neuroscience.**

Decodes probability distributions over orientation from human visual-cortex fMRI. Trial-to-trial differences in decoded uncertainty predict aspects of perceptual behavior, connecting population-level uncertainty estimates to what observers report.

<a id="ref-5"></a>

**5. van Bergen, R. S., & Jehee, J. F. M. (2019). [Probabilistic representation in human visual cortex reflects uncertainty in serial decisions](https://pmc.ncbi.nlm.nih.gov/articles/PMC6786811/). Journal of Neuroscience.**

Relates decoded uncertainty to serial dependence: how a recent percept influences the next judgment. Supports an account in which current and previous sensory estimates are combined according to their reliability and assumptions about temporal continuity.

<a id="ref-6"></a>

**6. Walker, E. Y., Cotton, R. J., Ma, W. J., & Tolias, A. S. (2020). [A neural basis of probabilistic computation in visual cortex](https://www.nature.com/articles/s41593-019-0554-5). Nature Neuroscience.**

A particularly close empirical continuation of the likelihood question. Likelihood functions decoded from simultaneous macaque V1 recordings predict orientation-category choices beyond what a point estimate explains, including variability at a fixed stimulus orientation.

Data/code: [likelihood code](https://github.com/eywalker/v1_likelihood) and [project code](https://github.com/eywalker/v1_project). The paper describes data availability on reasonable request; a public code repository should not be read as evidence that all raw trials are openly downloadable.

<a id="ref-7"></a>

**7. Geurts, L. S., Cooke, J. R. H., van Bergen, R. S., & Jehee, J. F. M. (2022). [Subjective confidence reflects representation of Bayesian probability in cortex](https://www.nature.com/articles/s41562-021-01247-w). Nature Human Behaviour.**

Connects sensory uncertainty to subjective confidence. Precision decoded from visual-cortex activity covaries with confidence, while additional regions carry signals related to the readout of that uncertainty.

Data/code: [preprocessed behavioral and fMRI data](https://doi.org/10.34973/983b-a047) and [Jehee lab code](https://github.com/jeheelab/). Raw imaging data have separate access conditions.

<a id="probability-frameworks"></a>

## 2. Neural sampling and interpretation of probability codes

Alternative computational frameworks and the evidence needed to distinguish them.

<a id="ref-8"></a>

**8. Haefner, R. M., Berkes, P., & Fiser, J. (2016). [Perceptual decision-making as probabilistic inference by neural sampling](https://pubmed.ncbi.nlm.nih.gov/27146267/). Neuron.**

Models neural activity as samples from an internal posterior distribution. Feedback during inference explains task-dependent correlations and relationships between sensory responses and choices, including different time courses for choice probability and stimulus influence on behavior.

<a id="ref-9"></a>

**9. Echeveste, R., Aitchison, L., Hennequin, G., & Lengyel, M. (2020). [Cortical-like dynamics in recurrent circuits optimized for sampling-based probabilistic inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC7610392/). Nature Neuroscience.**

Optimizes recurrent excitatory–inhibitory circuits to perform sampling-based inference. The resulting networks reproduce several features of cortical dynamics, including structured variability, transient responses, and oscillatory activity, linking a computational objective to circuit behavior.

<a id="ref-10"></a>

**10. Lange, R. D., Shivkumar, S., Chattoraj, A., & Haefner, R. M. (2023). [Bayesian encoding and decoding as distinct perspectives on neural coding](https://pmc.ncbi.nlm.nih.gov/articles/PMC11003438/). Nature Neuroscience.**

Distinguishes an experimenter's inference about an external stimulus from the brain's representation of beliefs about latent causes. This distinction clarifies how sampling and population-code descriptions can sometimes coexist, and why decoding a distribution does not uniquely identify the underlying neural computation.

<a id="ref-11"></a>

**11. Walker, E. Y., et al. (2023). [Studying the neural representations of uncertainty](https://www.nature.com/articles/s41593-023-01444-y). Nature Neuroscience.**

A methodological review of what would count as evidence that neurons represent uncertainty. Particularly useful for distinguishing a correlation with uncertainty, an experimenter's successful probabilistic decoder, and evidence that the brain itself uses the proposed representation.

<a id="ref-12"></a>

**12. Haefner, R. M., Beck, J., Savin, C., Salmasi, M., & Pitkow, X. (2024). [How does the brain compute with probabilities?](https://arxiv.org/abs/2409.02709). arXiv preprint.**

Compares major proposals for neural probabilistic computation, including probabilistic population codes, distributed distributional codes, and neural sampling. A useful synthesis for identifying which experiments could distinguish coding schemes rather than merely show uncertainty-related activity.

<a id="sdt"></a>

## 3. SDT, neural sensitivity, and components of attention

Response distributions, neurometric performance, and the distinction between sensitivity and criterion.

<a id="ref-16"></a>

**16. Werner, G., & Mountcastle, V. B. (1965). [Neural activity in mechanoreceptive cutaneous afferents: stimulus-response relations, Weber functions, and information transmission](https://doi.org/10.1152/jn.1965.28.2.359). Journal of Neurophysiology.**

A foundational quantitative treatment of sensory response variability and information. Useful for connecting response distributions, stimulus discrimination, and Weber-like sensitivity before moving to population likelihoods.

<a id="ref-17"></a>

**17. Britten, K. H., Shadlen, M. N., Newsome, W. T., & Movshon, J. A. (1992). [The analysis of visual motion: a comparison of neuronal and psychophysical performance](https://pubmed.ncbi.nlm.nih.gov/1464765/). Journal of Neuroscience.**

Uses signal-detection analysis to compare the sensitivity of individual MT neurons with the monkey's motion-discrimination performance. An accessible starting point for ROC analysis, neurometric functions, and the distinction between neural sensitivity and the animal's actual readout.

Data note: a currently working public release of the exact 1992 trial data was not established.

<a id="ref-67"></a>

**67. Verghese, P. (2001). [Visual search and attention: a signal detection theory approach](https://www.sciencedirect.com/science/article/pii/S0896627301003920). Neuron. Review.**

Develops an SDT account of attention and visual search, emphasizing distractors, uncertainty, and decisions based on multiple sensory signals. A direct theoretical bridge from response distributions to selection and search performance.

<a id="ref-47"></a>

**47. Luo, T. Z., & Maunsell, J. H. R. (2015). [Neuronal modulations in visual cortex are associated with only one of multiple components of attention](https://doi.org/10.1016/j.neuron.2015.05.007). Neuron.**

Separates behavioral sensitivity from criterion and relates them to V4 modulation. In the tested task, the neural changes track sensitivity rather than criterion. A strong introduction to the distinction between improving sensory discriminability and changing response policy; compare with Chandrasekaran et al. (55).

<a id="ref-66"></a>

**66. Luo, T. Z., & Maunsell, J. H. R. (2019). [Attention can be subdivided into neurobiological components corresponding to distinct behavioral effects](https://pmc.ncbi.nlm.nih.gov/articles/PMC6936701/). Proceedings of the National Academy of Sciences. Review/perspective.**

Organizes attention into components with distinct behavioral effects and potentially distinct neural mechanisms. Especially useful for sensitivity versus criterion and for interpreting apparently inconsistent mappings between neuronal modulation and attention.

<a id="ref-55"></a>

**55. Chandrasekaran, A. N., et al. (2024). [Dissociable components of attention exhibit distinct neuronal signatures in primate visual cortex](https://pmc.ncbi.nlm.nih.gov/articles/PMC10836731/). Science Advances.**

Dissociates covert attention from planned eye movements. In this task, V4 rate modulation tracks choice bias at the covertly attended location, while reduced correlated variability tracks sensitivity at the planned saccade location. A counterpoint to Luo & Maunsell (47): the mapping from neural modulation to behavioral components depends on the task.

<a id="tuning-theory"></a>

## 4. Tuning width, informative slopes, and tuning placement

General coding theory for asking what changing a tuning-curve parameter accomplishes. Attention, adaptation, and learning can then be evaluated as candidate biological mechanisms.

<a id="ref-41"></a>

**41. Pouget, A., Deneve, S., Ducom, J.-C., & Latham, P. E. (1999). [Narrow versus wide tuning curves: What's best for a population code?](https://physiology.med.cornell.edu/faculty/nirenberg/lab/pel/papers/nc-width.pdf). Neural Computation, 11, 85–90.**

A short theoretical analysis showing why tuning width or gain alone cannot determine coding quality: noise correlations and the decoder matter. Explicitly discusses attention, sharpening, and gain, and distinguishes information available in a representation from performance with a particular readout. A useful introductory reading for the SDT lecture.

<a id="ref-45"></a>

**45. Zhang, K., & Sejnowski, T. J. (1999). [Neuronal tuning: To sharpen or broaden?](https://direct.mit.edu/neco/article/11/1/75/6232/Neuronal-Tuning-To-Sharpen-or-Broaden). Neural Computation, 11, 75–84.**

Analyzes how tuning width affects Fisher information and shows that the relationship depends on the dimensionality of the encoded variable under the model assumptions. Complements the Pouget papers by explaining another reason sharpening has no universal benefit.

<a id="ref-70"></a>

**70. Pouget, A., Deneve, S., & Latham, P. E. (2001). [The relevance of Fisher information for theories of cortical computation and attention](https://direct.mit.edu/books/edited-volume/2997/Visual-Attention-and-Cortical-Circuits). In Visual Attention and Cortical Circuits. Book chapter.**

Quantitative background connecting neural tuning, noise, and discrimination precision through Fisher information. Supports introducing local discriminability before Bayesian inference and comparing attentional changes in gain, tuning, and variability. The link is to the volume and chapter listing.

<a id="ref-48"></a>

**48. Butts, D. A., & Goldman, M. S. (2006). [Tuning curves, neuronal variability, and sensory coding](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0040092). PLOS Biology.**

Explains how stimulus discrimination and response noise determine which parts of a tuning curve are informative. Steep flanks can dominate fine discrimination, while other regimes favor responses near the peak. A conceptual foundation for evaluating gain, sharpening, and tuning placement without equating high firing with useful information.

<a id="ref-43"></a>

**43. Harper, N. S., & McAlpine, D. (2004). [Optimal neural population coding of an auditory spatial cue](https://www.nature.com/articles/nature02768). Nature, 430, 682–686.**

Optimizes the distribution of preferred interaural time differences for the stimulus range an animal encounters. Depending on head size and sound frequency, optimal tuning peaks can lie outside that range, placing informative slopes inside it. Particularly useful for teaching tuning placement, local discriminability, and why concentrating tuning peaks at a target is not automatically optimal.

Teaching figure: Figure 2. [Full paper PDF](https://science.umd.edu/biology/soareslab/NACS641web/hearing.pdf).

<a id="ref-42"></a>

**42. Seriès, P., Latham, P. E., & Pouget, A. (2004). [Tuning curve sharpening for orientation selectivity: coding efficiency and the impact of correlations](https://www.nature.com/articles/nn1321). Nature Neuroscience, 7, 1129–1135.**

Examines sharpening in spiking-network models of orientation selectivity. In the studied models, sharpening through cortical lateral interactions produces correlations and substantial information loss compared with the alternative architecture. Shows why changing tuning width in an idealized encoding model and implementing sharpening through a circuit can have different coding consequences.

<a id="attention-tuning"></a>

## 5. Attention measured through gain, selectivity, and receptive fields

Physiological, psychophysical, and fMRI evidence for selective gain, altered tuning, and changes in spatial sampling. The entries distinguish direct neuronal measurements from population estimates and model-based inferences.

<a id="ref-25"></a>

**25. McAdams, C. J., & Maunsell, J. H. R. (1999). [Effects of attention on orientation-tuning functions of single neurons in macaque cortical area V4](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782389/). Journal of Neuroscience.**

Characterizes how attention changes orientation-tuned V4 responses, with effects resembling response gain more than wholesale sharpening of tuning. Useful for asking how multiplicative response changes alter a neuron's discriminability or a population decoder.

<a id="ref-26"></a>

**26. Treue, S., & Martínez-Trujillo, J. C. (1999). [Feature-based attention influences motion processing gain in macaque visual cortex](https://www.nature.com/articles/21176). Nature.**

Shows that attending to a motion feature modulates visual responses according to the relationship between the attended feature and neuronal preference. A foundation for feature-based gain models and their consequences for population coding.

Link correction: this paper's DOI is **10.1038/21176**. The `nature07821` URL belongs to Nienborg & Cumming (2009), not this paper.

<a id="ref-51"></a>

**51. Martínez-Trujillo, J. C., & Treue, S. (2004). [Feature-based attention increases the selectivity of population responses in primate visual cortex](https://pubmed.ncbi.nlm.nih.gov/15120065/). Current Biology.**

Shows how differential gain across neurons can sharpen the population response to an attended feature. The teaching distinction is between increased population selectivity and narrowing every individual neuron's tuning curve. A direct continuation of Treue & Martínez-Trujillo (26).

<a id="ref-46"></a>

**46. Ling, S., Liu, T., & Carrasco, M. (2009). [How spatial and feature-based attention affect the gain and tuning of population responses](https://sites.bu.edu/vision/files/2014/01/LingLiuCarrasco_2009.pdf). Vision Research.**

Combines motion-discrimination thresholds across external-noise levels with a population model. In that framework, spatial attention acts through gain, while feature attention involves gain and sharpening. Particularly close to the proposed lecture; Figure 1 illustrates the alternatives. The tuning changes are inferred from psychophysics and modeling, rather than directly measured in individual neurons.

<a id="ref-50"></a>

**50. Scolari, M., & Serences, J. T. (2009). [Adaptive allocation of attentional gain](https://www.jneurosci.org/content/29/38/11933). Journal of Neuroscience.**

Behavioral evidence that attentional gain can be allocated to features useful for the current discrimination, including features displaced from the target. Complements the later fMRI study (49) and helps distinguish target-centered enhancement from enhancement optimized for the task.

<a id="ref-49"></a>

**49. Scolari, M., Byers, A., & Serences, J. T. (2012). [Optimal deployment of attentional gain during fine discriminations](https://pmc.ncbi.nlm.nih.gov/articles/PMC3384562/). Journal of Neuroscience.**

Uses fMRI and an encoding model to examine attentional enhancement of populations tuned away from the target. Such populations can provide useful slopes for distinguishing nearby alternatives. Supports task-dependent allocation of gain; Figure 1 contrasts fine and coarse discrimination.

<a id="ref-74"></a>

**74. Kozyrev, V., Daliri, M. R., Schwedhelm, P., & Treue, S. (2019). [Strategic deployment of feature-based attentional gain in primate visual cortex](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000387). PLOS Biology.**

In macaque MT, attentional modulation peaks along the flanks of the population activity profile rather than exactly at the target feature. Connects the theoretical value of informative tuning slopes to measured allocation of attentional gain during target–distractor discrimination.

<a id="ref-27"></a>

**27. David, S. V., Hayden, B. Y., Mazer, J. A., & Gallant, J. L. (2008). [Attention to stimulus features shifts spectral tuning of V4 neurons during natural vision](https://pmc.ncbi.nlm.nih.gov/articles/PMC2948549/). Neuron.**

Studies feature attention with naturalistic stimuli and shows changes in V4 tuning toward behaviorally relevant features. Extends attention models beyond a uniform increase in firing rate to changes in the features emphasized by the neural representation.

<a id="ref-58"></a>

**58. Womelsdorf, T., et al. (2006). [Dynamic shifts of visual receptive fields in cortical area MT by spatial attention](https://doi.org/10.1038/nn1748). Nature Neuroscience.**

Direct physiological evidence that spatial attention shifts MT receptive fields toward an attended location. Establishes a change in spatial sampling during attention; the behavioral contribution of this change still depends on the population response and its readout.

<a id="ref-77"></a>

**77. Abdirashid, S. S., Knapen, T., & Dumoulin, S. O. (2025). [The precision of attention controls attraction of population receptive fields](https://doi.org/10.1167/jov.25.11.3). Journal of Vision, 25(11):3.**

Uses focused versus distributed attention and 7T fMRI to measure population receptive fields. Focused attention produces stronger attraction toward the attended location, explained by an attention-field model. Connects a specific attentional parameter—spatial precision—to changes in measured population tuning.

<a id="ref-78"></a>

**78. Tünçok, E., Carrasco, M., & Winawer, J. (2025). [Spatial attention selectively alters visual cortical representation during target anticipation](https://www.nature.com/articles/s41467-025-63795-3). Nature Communications, 16:8746.**

Combines psychophysics and fMRI to study attention before target onset. Spatially selective baseline changes and shifts in population receptive-field centers accompany behavioral benefits at cued locations and costs elsewhere. Demonstrates anticipatory preparation, without isolating receptive-field shifts as the sole cause of improvement.

<a id="ref-33"></a>

**33. Westerberg, J. A., Schall, J. D., Woodman, G. F., & Maier, A. (2023). [Feedforward attentional selection in sensory cortex](https://www.nature.com/articles/s41467-023-41745-1). Nature Communications.**

Uses laminar recordings during visual search to examine where and when attentional selection emerges in sensory cortex. Early V4 activity and its relationship to reaction time make this useful for connecting neural selection signals with the timing of behavioral reports.

Data: [Zenodo record](https://zenodo.org/records/8173495), retained from the earlier conversation; current archive contents were not independently inspected for this merge.

<a id="ref-34"></a>

**34. Mendoza-Halliday, D., et al. (2024). [Dissociable neuronal substrates of feature attention and working memory](https://doi.org/10.1016/j.neuron.2023.12.007). Neuron.**

Separates neural signals related to attending to a feature from those related to holding that feature in working memory. Useful for testing whether superficially similar population signals support distinct cognitive operations.

Data/code: [Zenodo release](https://zenodo.org/records/10573510). The earlier review identified per-trial firing-rate data; matching choice and reaction-time fields were not established, so a ready-made neural–behavioral trial analysis should not be assumed.

<a id="attention-models"></a>

## 6. Computational mechanisms of attention and task performance

Models with specified mechanisms and downstream computations, useful for testing whether a proposed modulation is sufficient to improve performance.

<a id="ref-52"></a>

**52. Reynolds, J. H., & Heeger, D. J. (2009). [The normalization model of attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC2752446/). Neuron.**

A mechanistic model in which attentional modulation interacts with divisive normalization. Stimulus and attention-field conditions determine whether responses resemble response gain, contrast gain, or other modulations. Useful for connecting a specific circuit computation to changes in measured response functions.

<a id="ref-72"></a>

**72. Lindsay, G. W., & Miller, K. D. (2018). [How biological attention mechanisms improve task performance in a large-scale visual system model](https://elifesciences.org/articles/38105). eLife.**

Compares attentional modulation based on unit preference with modulation based on each unit's influence on task output in a visual network. Selectivity and task usefulness can diverge, and gradient-based modulation can perform better. Explicit SDT analyses separate sensitivity and criterion; Figures 2, 6, and 7 are especially relevant to the lecture.

<a id="ref-54"></a>

**54. Fox, K. J., Birman, D., & Gardner, J. L. (2023). [Gain, not concomitant changes in spatial receptive field properties, improves task performance in a neural network attention model](https://elifesciences.org/articles/78392). eLife.**

Separates the consequences of spatial gain from accompanying changes in receptive-field position, size, and structure. Gain explains the performance benefit in their tested network and tasks; the accompanying receptive-field changes are neither necessary nor sufficient. Especially useful for asking whether an observed tuning change causes better performance.

<a id="ref-79"></a>

**79. Salehi, S., Lei, J., Benjamin, A. S., Müller, K.-R., & Kording, K. P. (2026). [Modeling attention and binding in the brain through bidirectional recurrent gating](https://www.nature.com/articles/s41467-026-72146-9). Nature Communications, 17:4072.**

A recurrent visual architecture uses top-down and lateral multiplicative modulation to perform orienting, filtering, and visual search and reproduce several attention and binding phenomena. Links specified mechanisms to complete task behavior; establishes computational sufficiency within the model rather than identifying the brain's implementation. Published May 5, 2026.

Code: [bio-attention repository](https://github.com/ssnio/bio-attention).

<a id="attention-covariance"></a>

## 7. Attention, covariance, cortical state, and causal perturbation

How shared variability and state relate to behavior, including causal perturbation and its accompanying correction.

<a id="ref-28"></a>

**28. Cohen, M. R., & Maunsell, J. H. R. (2009). [Attention improves performance primarily by reducing interneuronal correlations](https://doi.org/10.1038/nn.2439). Nature Neuroscience.**

Connects attentional improvements in behavior to changes in shared V4 variability. In this experiment, reduced correlations account for a large fraction of the improvement in population sensitivity, making it a useful case study in why population covariance matters.

<a id="ref-29"></a>

**29. Cohen, M. R., & Maunsell, J. H. R. (2010). [A neuronal population measure of attention predicts behavioral performance on individual trials](https://pmc.ncbi.nlm.nih.gov/articles/PMC3045704/). Journal of Neuroscience.**

Constructs a population measure of attentional state that predicts trial-by-trial performance. Particularly useful for a compact analysis linking a projection of population activity to hits, misses, and fluctuations in internal state.

<a id="ref-71"></a>

**71. Ruff, D. A., & Cohen, M. R. (2014). [Attention can either increase or decrease spike count correlations in visual cortex](https://www.nature.com/articles/nn.3835). Nature Neuroscience.**

Shows that attention does not uniformly reduce pairwise correlations. A useful counterweight to treating decorrelation as a universal mechanism: what matters is how shared variability is structured relative to stimulus signals and the relevant readout.

<a id="ref-30"></a>

**30. Nandy, A. S., Nassi, J. J., Jadi, M. P., & Reynolds, J. H. (2019). [Optogenetically induced low-frequency correlations impair perception](https://elifesciences.org/articles/35123). eLife.**

Uses optogenetic perturbation to alter correlated activity in visual cortex. Induced low-frequency correlations impair perceptual performance, providing causal evidence that the temporal structure of shared variability matters for perception.

Data: [paper-associated Dryad release](https://doi.org/10.5061/dryad.8v0k1j3). The prior conversation also identified a [Zenodo release](https://zenodo.org/records/5010799); its exact contents and relationship to the Dryad files were not independently rechecked here.

<a id="ref-31"></a>

**31. Nandy, A. S., Nassi, J. J., Jadi, M. P., & Reynolds, J. H. (2020). [Correction: Optogenetically induced low-frequency correlations impair perception](https://elifesciences.org/articles/55718). eLife, 9:e55718.**

The correction accompanying entry 30 revisits statistical testing, psychometric fitting, and the treatment of false alarms. Read it alongside the original paper when preparing a notebook on behavioral sensitivity or threshold estimation; the supplied discussion specifically highlighted it as a way to examine how analysis choices affect those estimates.

<a id="ref-32"></a>

**32. Shi, Y., et al. (2022). [Cortical state dynamics and selective attention define the spatial pattern of correlated variability in neocortex](https://www.nature.com/articles/s41467-021-27724-4). Nature Communications.**

Studies how cortical state and attention jointly shape the spatial organization of shared variability in V4. Useful for distinguishing a local change in attention from broader state fluctuations when interpreting neural correlations and behavioral performance.

Data/code: [Figshare data and MATLAB resources](https://doi.org/10.6084/m9.figshare.16934326.v3).

<a id="readout"></a>

## 8. Population readout, choice signals, and the unknown decoder

Separate information available in a representation from the information used by the animal. Includes selection and pooling, downstream recordings, feedback, and competing decoder hypotheses.

<a id="ref-18"></a>

**18. Britten, K. H., et al. (1996). [A relationship between behavioral choice and the visual responses of neurons in macaque MT](https://pubmed.ncbi.nlm.nih.gov/8730992/). Visual Neuroscience.**

Shows that trial-to-trial MT responses correlate with perceptual choice even for the same sensory stimulus. Introduces a central example for choice probability: ROC analysis of response distributions conditioned on the animal's decision.

Data: historically distributed through the [Neural Signal Archive](http://www.neuralsignal.org/index_data.html) and reused by Chicharro et al. (2021), below. Current archive download availability was not confirmed.

<a id="ref-19"></a>

**19. Nienborg, H., & Cumming, B. G. (2009). [Decision-related activity in sensory neurons reflects more than a neuron's causal effect](https://www.nature.com/articles/nature07821). Nature.**

Shows why choice-related sensory activity cannot simply be interpreted as the neuron's feedforward contribution to the decision. The temporal relationship between sensory evidence and choice signals implicates additional influences, including decision-related feedback.

<a id="ref-20"></a>

**20. Pitkow, X., Liu, S., Angelaki, D. E., DeAngelis, G. C., & Pouget, A. (2015). [How can single sensory neurons predict behavior?](https://pmc.ncbi.nlm.nih.gov/articles/PMC4683594/). Neuron.**

Relates single-neuron sensitivity, population correlations, and behavioral readout. Explains how individual neurons can remain correlated with behavior despite large population sizes, emphasizing the structure of shared variability and the decoder.

<a id="ref-21"></a>

**21. Yates, J. L., et al. (2017). [Functional dissection of signal and noise in MT and LIP during decision-making](https://www.nature.com/articles/nn.4611). Nature Neuroscience.**

Uses statistical models of MT and LIP activity to separate stimulus-related, history-dependent, and shared components during motion decisions. Useful for moving from simple spike-count correlations to models of how sensory and decision signals evolve over time.

Data/code: [MT–LIP GLM repository](https://github.com/jcbyts/mtlipglm). Contains analysis resources and example data; a complete release of every experimental session was not verified.

<a id="ref-22"></a>

**22. Quinn, K. R., Seillier, L., Butts, D. A., & Nienborg, H. (2021). [Decision-related feedback in visual cortex lacks spatial selectivity](https://www.nature.com/articles/s41467-021-24629-0). Nature Communications.**

Tests the spatial specificity of decision-related signals during visual discrimination. The findings support broadly distributed feedback, helping distinguish sensory selectivity from the spatial organization of choice-related activity.

Data/code: [public subset and analysis code](https://github.com/NienborgLab/Quinn_et_al_2021); the complete dataset is available on request according to the paper.

<a id="ref-23"></a>

**23. Chicharro, D., Panzeri, S., & Haefner, R. M. (2021). [Stimulus-dependent relationships between behavioral choice and sensory neural responses](https://elifesciences.org/articles/54858). eLife.**

Examines how stimulus conditions alter the association between sensory responses and choice. Extends the interpretation of choice probability beyond a single pooled value and reanalyzes classic MT data in a framework that accommodates stimulus-dependent effects.

Code: [CP/DP analysis repository](https://github.com/DanielChicharro/CP_DP). Access to the historical input data depends on the archive noted under Britten et al. (1996).

<a id="ref-53"></a>

**53. Pestilli, F., Carrasco, M., Heeger, D. J., & Gardner, J. L. (2011). [Attentional enhancement via selection and pooling of early sensory responses in human visual cortex](https://pubmed.ncbi.nlm.nih.gov/22153378/). Neuron.**

Combines psychophysics, fMRI, and modeling to argue that attention improves how sensory responses are selected and pooled into a decision. Provides an alternative to explaining behavioral improvement solely through increased sensitivity of the measured sensory representation.

<a id="ref-75"></a>

**75. Ruff, D. A., & Cohen, M. R. (2019). [Simultaneous multi-area recordings suggest that attention improves performance by reshaping stimulus representations](https://www.nature.com/articles/s41593-019-0477-1). Nature Neuroscience.**

Combines sensory and oculomotor population recordings with behavior. Supports attention changing sensory representations so that they more effectively influence downstream processing. Particularly relevant to separating information available in a representation from information accessible to the animal's readout.

<a id="ref-76"></a>

**76. Ni, A. M., Huang, C., Doiron, B., & Cohen, M. R. (2022). [A general decoding strategy explains the relationship between behavior and correlated variability](https://elifesciences.org/articles/67258). eLife.**

A decoder optimized over a broader orientation range better matches the monkeys' decoding strategy than one optimized for the particular discrimination. Its greater sensitivity to shared variability helps explain why attentional correlation changes predict behavior. Makes the unknown biological decoder an explicit, testable modeling question.

Data/code: [electrophysiological data on OSF](https://doi.org/10.17605/OSF.IO/RN7TU) and [simulation/analysis code](https://github.com/hcc11/GeneralDecoder). A suitable notebook comparison is fixed weights versus task-specific refitted weights versus weights trained across a broad stimulus range.

<a id="learning"></a>

## 9. Perceptual learning: tuning, maps, correlations, and readout

Longer-term improvements can change sensory tuning, cortical allocation, shared variability, downstream interpretation, or several of these together.

<a id="ref-56"></a>

**56. Schoups, A., Vogels, R., Qian, N., & Orban, G. (2001). [Practising orientation identification improves orientation coding in V1 neurons](https://www.nature.com/articles/35087601). Nature.**

Links orientation training to changes in V1 tuning that improve local coding around the trained orientation. Particularly relevant to useful tuning slopes, rather than the simpler claim that training makes more neurons peak at the target.

<a id="ref-57"></a>

**57. Raiguel, S., et al. (2006). [Learning to see the difference specifically alters the most informative V4 neurons](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674023/). Journal of Neuroscience.**

Shows that learning-related changes preferentially involve V4 neurons informative for the trained discrimination. A visual example linking plasticity to task-relevant sensitivity and the geometry of tuning curves.

<a id="ref-59"></a>

**59. Recanzone, G. H., Schreiner, C. E., & Merzenich, M. M. (1993). [Plasticity in the frequency representation of primary auditory cortex following discrimination training in adult owl monkeys](https://www.jneurosci.org/content/13/1/87). Journal of Neuroscience.**

Demonstrates expansion of cortical representation around sound frequencies used in discrimination training. A direct auditory example of reallocating representation to a behaviorally relevant stimulus range through perceptual learning.

<a id="ref-60"></a>

**60. Polley, D. B., Steinberg, E. E., & Merzenich, M. M. (2006). [Perceptual learning directs auditory cortical map reorganization through top-down influences](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674159/). Journal of Neuroscience.**

Shows that auditory cortical reorganization depends on the stimulus dimension relevant to the task. Helps separate learning driven by behavioral demands from changes caused by repeated sensory exposure alone.

<a id="ref-61"></a>

**61. Law, C.-T., & Gold, J. I. (2008). [Neural correlates of perceptual learning in a sensory-motor, but not a sensory, cortical area](https://pmc.ncbi.nlm.nih.gov/articles/PMC2424192/). Nature Neuroscience.**

Compares MT and LIP during learning and finds changes consistent with improved interpretation or use of sensory evidence. An essential alternative to accounts in which perceptual improvement requires sharper or otherwise improved sensory tuning.

<a id="ref-62"></a>

**62. Gu, Y., et al. (2011). [Perceptual learning reduces interneuronal correlations in macaque visual cortex](https://pmc.ncbi.nlm.nih.gov/articles/PMC3163063/). Neuron.**

Examines learning-related reductions in shared variability in macaque MSTd during heading discrimination. Broadens the learning discussion from changes in mean tuning to covariance; the consequences for population sensitivity depend on correlation structure and readout.

<a id="ref-24"></a>

**24. Liu, S., Pletenev, A., Haefner, R. M., & Snyder, A. C. (2026). [Task learning increases information redundancy of neural responses in macaque visual cortex](https://doi.org/10.1126/science.adw7707). Science.**

Tracks V4 populations as monkeys learn visual discrimination tasks. Learning increases redundancy across neurons while increasing the information available from individual neurons, supporting predictions of generative-inference models and challenging the assumption that learning must make sensory representations less redundant.

Data/code: [Zenodo record](https://zenodo.org/records/17460460), recovered from the earlier conversation as a release of neural responses, task variables, and behavior. The record's current download contents were not independently inspected for this merge.

<a id="adaptation"></a>

## 10. Adaptation and redistribution of coding precision

Examples of changed encoding, its dependence on stimulus statistics, and whether the decoder compensates.

<a id="ref-35"></a>

**35. Seriès, P., Stocker, A. A., & Simoncelli, E. P. (2009). [Is the homunculus “aware” of sensory adaptation?](https://pmc.ncbi.nlm.nih.gov/articles/PMC3134250/). Neural Computation.**

Asks whether downstream decoding compensates for adaptation-induced changes in sensory encoding. Comparing an adaptation-aware decoder with a fixed or mismatched decoder links altered tuning curves to perceptual biases and discrimination thresholds.

<a id="ref-44"></a>

**44. Dean, I., Harper, N. S., & McAlpine, D. (2005). [Neural population coding of sound level adapts to stimulus statistics](https://www.nature.com/articles/nn1541). Nature Neuroscience, 8, 1684–1689.**

Shows that auditory midbrain neurons adjust their response functions to sound-level statistics, improving population coding near frequently encountered levels. An empirical example of redistributing coding precision through adaptation. The computational question can also motivate attention models, while the manipulation demonstrated here is adaptation.

<a id="motion-context"></a>

## 11. Motion context, surround effects, and causal inference

Motion perception as a population and inference problem, extending beyond the isolated tuning curve.

<a id="ref-13"></a>

**13. Liu, L. D., Haefner, R. M., & Pack, C. C. (2016). [A neural basis for the spatial suppression of visual motion perception](https://elifesciences.org/articles/16167). eLife.**

Combines macaque motion psychophysics, MT recordings, and a population model to explain why larger moving stimuli can be harder to discriminate. Surround suppression, structured noise correlations, and the population readout jointly account for the effect; single-neuron sensitivity alone is insufficient.

Data/code: [Pack lab archive](https://packlab.mcgill.ca/suppression%20data%20and%20code.zip). The recovered release contains tuning/variance summaries, psychometric parameters, and modeling resources; it should not be treated as a verified release of trial-level spike trains.

<a id="ref-14"></a>

**14. Shivkumar, S., DeAngelis, G. C., & Haefner, R. M. (2025). [Hierarchical motion perception as causal inference](https://www.nature.com/articles/s41467-025-58797-0). Nature Communications.**

Explains motion grouping and relative-motion judgments through hierarchical inference about shared causes and reference frames. Human behavioral results support the model; posterior sampling provides the best fit among the tested mappings from inferred distributions to reports.

Data/code: [OSF project](https://osf.io/f9dsg/).

<a id="ref-15"></a>

**15. Lengyel, G., Shivkumar, S., DeAngelis, G. C., & Haefner, R. M. (2026). [Bayesian causal inference unifies perceptual and neuronal processing of center-surround motion in area MT](https://elifesciences.org/reviewed-preprints/110621). eLife reviewed preprint.**

Extends a causal-inference account of motion perception toward neuronal responses in MT. A sampling interpretation connects inferred motion structure to center–surround response modulation and variability. The comparisons concern previously reported neural phenomena; this is not a new simultaneous neural-and-behavioral validation of the entire framework.

<a id="reviews"></a>

## 12. Broad reviews connecting attention, coding, and behavior

Orientation readings and lecture preparation. Focused reviews also appear under SDT (66, 67) and probability frameworks (11, 12).

<a id="ref-63"></a>

**63. Maunsell, J. H. R. (2015). [Neuronal mechanisms of visual attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC8279254/). Annual Review of Vision Science. Review.**

A broad neural introduction to attention, covering response modulation, population variability, and links to behavior. A strong general companion to an SDT lecture that moves from individual tuning curves to population responses.

<a id="ref-64"></a>

**64. Reynolds, J. H., & Chelazzi, L. (2004). [Attentional modulation of visual processing](https://doi.org/10.1146/annurev.neuro.26.041002.131039). Annual Review of Neuroscience. Review.**

Classic review of attentional changes in neural responses and competition among stimuli. Provides detailed background for gain, selection, and normalization accounts; particularly useful for lecture preparation.

<a id="ref-65"></a>

**65. Anton-Erxleben, K., & Carrasco, M. (2013). [Attentional enhancement of spatial resolution: linking behavioural and neurophysiological evidence](https://www.nature.com/articles/nrn3443). Nature Reviews Neuroscience. Review.**

Connects behavioral improvements in spatial resolution with changes in receptive-field size, position, and allocation. The closest review in this collection to the question of whether receptive-field shrinkage or relocation can explain attentional benefits.

<a id="ref-68"></a>

**68. Carrasco, M. (2011). [Visual attention: the past 25 years](https://pubmed.ncbi.nlm.nih.gov/21549742/). Vision Research. Review.**

Broad review of behavioral attention research, including contrast sensitivity, spatial resolution, and feature attention. Useful for linking proposed changes in neural coding to the psychophysical tasks used to measure their consequences.

<a id="ref-69"></a>

**69. Ruff, D. A., Ni, A. M., & Cohen, M. R. (2018). [Cognition as a window into neuronal population space](https://doi.org/10.1146/annurev-neuro-080317-061936). Annual Review of Neuroscience. Review.**

Explains cognitive effects through population activity, shared variability, and behaviorally relevant dimensions. A conceptual extension of the scalar-readout approach from single-neuron tuning to population geometry and downstream computation.

<a id="ref-73"></a>

**73. Lindsay, G. W. (2020). [Attention in Psychology, Neuroscience, and Machine Learning](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.00029/full). Frontiers in Computational Neuroscience. Review.**

Connects attention across behavior, biological mechanisms, and machine learning. Treats sensory modulation and altered readout as complementary possibilities. Use for broad orientation alongside the explicit computational tests in Lindsay & Miller (72).

<a id="datasets"></a>

## 13. Choice, reward, action, and reusable neural datasets

The broader neural-perception and dataset material retained from the original merged collection.

<a id="ref-36"></a>

**36. Steinmetz, N. A., et al. (2019). [Distributed coding of choice, action, and engagement across the mouse brain](https://pmc.ncbi.nlm.nih.gov/articles/PMC6913580/). Nature.**

Large-scale recordings reveal how sensory, choice, movement, and engagement signals are distributed across the brain during a visual task. A useful dataset-oriented entry for separating signals associated with a decision from those associated with the accompanying action or behavioral state.

Data: [Figshare dataset](https://figshare.com/articles/dataset/Distributed_coding_of_choice_action_and_engagement_across_the_mouse_brain/9974357), linked in the supplied conversation and the [paper’s data-availability statement](https://www.nature.com/articles/s41586-019-1787-x). A related [UCL repository record](https://rdr.ucl.ac.uk/articles/dataset/Distributed_coding_of_choice_action_and_engagement_across_the_mouse_brain/9970907) also identifies the dataset. The archive contents were not downloaded for this bibliography check.

<a id="ref-37"></a>

**37. Kunimatsu, J., Yamamoto, M., Maeda, K., & Hikosaka, O. (2021). [Environment-based object values learned by local network in striatum tail](https://doi.org/10.1073/pnas.2013623118). Proceedings of the National Academy of Sciences.**

Examines how striatal circuitry learns object values in relation to the surrounding environment. Broadens the collection from sensory uncertainty to learned value representations relevant to visually guided behavior.

Data: [Mendeley Data release](https://data.mendeley.com/datasets/t9gp59g8p5/1). The precise alignment of spikes, individual choices, and saccade reaction times was not verified in the recovered discussion.

<a id="ref-38"></a>

**38. Larry, N., Zur, G., & Joshua, M. (2024). [Organization of reward and movement signals in the basal ganglia and cerebellum](https://www.nature.com/articles/s41467-024-45921-9). Nature Communications.**

Compares reward and movement coding across basal ganglia and cerebellar recordings from the same monkeys. Useful for models that separate reward effects from eye-movement timing and kinematics, rather than interpreting all behavioral covariation as sensory evidence.

Data/code: [Dryad dataset](https://doi.org/10.5061/dryad.73n5tb33r) and [Zenodo code](https://doi.org/10.5281/zenodo.10252467). The dataset includes neural recordings, task/reward events, and eye-movement information.

<a id="ref-39"></a>

**39. Stoll, F. M., & Rudebeck, P. H. (2024). [Preferences reveal dissociable encoding across prefrontal-limbic circuits](https://doi.org/10.1016/j.neuron.2024.03.020). Neuron.**

Uses choices involving reward identity and probability to distinguish coding across prefrontal–limbic circuits. The findings separate aspects of outcome quality and availability, making this the substantive research companion to the broader London et al. dataset below.

Data connection: the London et al. release includes recordings from the experimental program underlying this work; the papers should remain distinct bibliography entries.

<a id="ref-40"></a>

**40. London, L., Love, M., Zeisler, Z. R., Rudebeck, P. H., & Stoll, F. M. (2026). [Dataset of cortical and subcortical single neuron activity during value-based tasks in macaque monkey](https://www.nature.com/articles/s41597-026-07129-y). Scientific Data.**

A public dataset of 16,495 neurons from 22 anatomically verified areas, recorded in two macaques across 340 behavioral sessions. Single- and two-option tasks vary juice identity and reward probability; the release includes spike times, behavior, and recording locations, with examples suited to neural coding, choice, and reaction-time analyses.

Data/code: [Zenodo dataset and examples](https://zenodo.org/records/17524410) and [FlavorProba analysis repository](https://github.com/RudebeckLab/FlavorProba-dataset).

## Coverage audit and availability conventions

| Source in this document | Master-list references | Unique references contributed |
|---|---|---:|
| Original 14-paper probabilistic-perception list | 1–12, 14–15 | 14 |
| Supplied “Papers for Neural Perception” Markdown from chat `6aa4a539-31d8-83ea-8d6b-a3a1e8af39e3`: 27 papers and 1 correction | 1–2, 13, 16–40; 1–2 overlap with the row above | 26 |
| Follow-up on Pouget, Harper–McAlpine, and tuning parameters | 41–45 | 5 |
| SDT, attention, tuning curves, and perceptual learning: references previously present only in the discussion | 46–71, including the book chapter at 70 | 26 |
| Latest answer on attention models, recent developments, and the decoder | 72–79, plus Fox et al. already covered at 54 | 8 |
| Jazayeri–Movshon Nature paper mentioned in the teaching question | 80 | 1 |
| Teaching shortlist: five groups of five and suggested two-paper pairings | 25 existing references, all already included above | 0 |
| **Total** | **1–80, each represented once** | **80** |

The total comprises **78 papers/reviews/preprints, one book chapter (70), and one correction (31)**. The initial merge had 40 references (14 + 27 + 1 − 2), and the tuning-parameter follow-up brought it to 45. This audit adds 35 distinct references, including papers embedded in prose and tables. Fox et al. (2023), discussed twice, has one bibliography entry (54). Both the Jazayeri–Movshon Nature Neuroscience (1) and Nature (80) papers are now explicitly listed.

All paper/chapter links in the four preserved discussions have a corresponding entry in the master bibliography. Repeated citations, alternative links to the same paper, and associated data/code resources are not counted as additional papers. Coverage refers to this document and its incorporated discussions, rather than to all publications in the field.

Paper links identify the publication or an authoritative full-text/abstract copy. Data and code links are separately labeled because an article, code repository, example session, processed summary, and complete trial-level release are different resources. Where current accessibility or exact trial fields were not verified, the limitation is recorded at the relevant entry. No raw neural datasets were downloaded as part of this bibliography merge.

---

## Added discussion: SDT, attention, tuning curves, and perceptual learning

### User question

good. i discussed jazayeri movshon nature and nature neuroscience this week to inform on decoding, likelihood etc. later i will itnroduce bayesian perspective. but now i want to talk about sdt anda ttention.. do any of these papers introduce that ? i could talk about why or how to improve performance by scaling up tc, (increasing firing rate), sharpening tuning curve, moving more curves to overlap desired lcoation (perceptual learning)etc. are any of these 40 papers good for that ? also look for other papers not in this list.. could one work best ? review good too. go wide

### Full answer

**Yes. Several of the 40 fit—and I found some additions that match your proposed lecture even more closely.** You can develop this entirely through tuning curves, response distributions, decoding and \(d'\), before introducing Bayesian inference.

For your particular purpose, my first choices would be:

| What you want the reading to accomplish | My choice |
|---|---|
| **Compare gain and sharpening, with a model connecting them to performance** | **Ling, Liu & Carrasco (2009)** |
| **Start with actual neuronal tuning curves under attention** | **McAdams & Maunsell (1999)**—already #25 |
| **Explicitly connect attention to SDT sensitivity and criterion** | **Luo & Maunsell (2015)** |
| **Assign a broad neural review** | **Maunsell (2015)** |
| **Explain why the most active neurons are not necessarily the most useful** | **Butts & Goldman (2006)**, followed by **Scolari et al. (2012)** |

**Ling et al. is probably the closest single paper to the gain-versus-tuning lecture you described.** For a pair of empirical neurophysiology papers, I would choose **McAdams–Maunsell 1999 + Cohen–Maunsell 2009**: first change the mean responses, then examine the variability of the resulting population readout.

The main point I would build the class around is **how a manipulation changes the separation of the response distributions relative to their variability**. Increasing firing, sharpening tuning and reallocating preferred stimuli are possible mechanisms; their benefit depends on the discrimination and noise model.

For a scalar readout \(y=\mathbf w^\top\mathbf r\), with approximately equal conditional variances,

\[
d'_y=
\frac{|\mathbb E[y\mid s_2]-\mathbb E[y\mid s_1]|}
{\sqrt{\tfrac12\{\operatorname{Var}(y\mid s_1)+\operatorname{Var}(y\mid s_2)\}}}.
\]

That preserves your “many neurons → one computed variable → perception” approach.

**From the existing 40, these are the strongest fits.**

| Paper | What it gives you for this class |
|---|---|
| **#25. [McAdams & Maunsell (1999)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782389/)**, *Effects of attention on orientation-tuning functions…* | **The cleanest starting point.** Attention approximately scales V4 orientation responses without systematically narrowing their tuning. Students can ask what this does to discrimination under different assumptions about response variance. |
| **#26. [Treue & Martínez-Trujillo (1999)](https://www.nature.com/articles/21176)**, *Feature-based attention influences motion processing gain…* | Gain depends on the relationship between the attended feature and neuronal preference. Introduces **selective gain across a population**, beyond multiplying every neuron by the same factor. |
| **#27. [David et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2948549/)**, *Attention to stimulus features shifts spectral tuning…* | Your best existing example of **attention changing tuning preferences**, rather than only response magnitude. Natural-image spectral tuning makes it richer, but less elementary than orientation tuning. |
| **#28. [Cohen & Maunsell (2009)](https://doi.org/10.1038/nn.2439)**, *Attention improves performance primarily by reducing interneuronal correlations* | The natural second paper: improvement can come from changing **population variability**, even when rate changes alone explain relatively little. The quantitative result is specific to their experiment and analysis. |
| **#29. [Cohen & Maunsell (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3045704/)**, *A neuronal population measure of attention predicts behavioral performance…* | Especially close to your scalar-readout constraint: reduce population activity to an attention-axis projection and relate it to behavior. Less directly about how tuning modifications improve sensitivity. |
| **#30–31. [Nandy et al. (2019)](https://elifesciences.org/articles/35123), with [2020 correction](https://elifesciences.org/articles/55718)** | Adds a **causal perturbation** of correlated variability, behavioral discrimination and an existing public dataset. Good later in the sequence; the correction belongs in the reading. |
| **#35. [Seriès, Stocker & Simoncelli (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3134250/)**, *Is the homunculus “aware” of sensory adaptation?* | Excellent for **changed encoding + unchanged decoder versus updated decoder**. It concerns adaptation, but directly supports your proposed notebook manipulations and their effects on bias and threshold. |

I would use Shi et al. (#32) as a more advanced continuation on cortical state and covariance, rather than the introductory paper.

**The most useful additions are these.**

1. **Ling, Liu & Carrasco (2009), “How spatial and feature-based attention affect the gain and tuning of population responses.” *Vision Research*.**  
   This paper explicitly poses your question. It measures motion-discrimination thresholds across external-noise levels and implements a population model. Within that framework, spatial attention is explained by gain, while feature attention involves both gain and sharpening of the population response. **Figure 1 is particularly useful for teaching.** The evidence is human psychophysics plus modeling; the inferred population sharpening is not a direct measurement that individual neurons narrow their tuning curves. [Paper PDF](https://sites.bu.edu/vision/files/2014/01/LingLiuCarrasco_2009.pdf)

2. **Luo & Maunsell (2015), “Neuronal modulations in visual cortex are associated with only one of multiple components of attention.” *Neuron*.**  
   **Probably the best explicit SDT–attention introduction.** They separate changes in behavioral sensitivity from changes in criterion, and relate those changes to V4 activity. In their task, the V4 modulations correspond to sensitivity changes rather than criterion shifts. This makes “attention improves performance” a question students can analyze rather than an assumption. [Paper](https://doi.org/10.1016/j.neuron.2015.05.007)

3. **Butts & Goldman (2006), “Tuning curves, neuronal variability, and sensory coding.” *PLOS Biology*.**  
   **My strongest conceptual companion to your tuning-curve simulations.** It explains when the informative part of a tuning curve is its steep flank and when peak responses become more useful, depending on noise and the discrimination being considered. It prevents the inference that a neuron necessarily contributes most to discriminating stimuli near its preferred value. [Open paper](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0040092)

4. **Scolari, Byers & Serences (2012), “Optimal deployment of attentional gain during fine discriminations.” *Journal of Neuroscience*.**  
   A very direct extension of the slope argument. For fine discrimination, enhancing neurons tuned **away from the target**, whose flanks distinguish the alternatives, can be more useful than enhancing neurons peaking at the target. Their fMRI and encoding-model results support adaptive off-target enhancement when the discriminanda are specified precisely in advance. **Figure 1 provides a clear coarse-versus-fine discrimination comparison.** [Open paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3384562/)  
   The earlier **Scolari & Serences (2009), “Adaptive allocation of attentional gain,”** is a useful behavioral companion. [Paper](https://www.jneurosci.org/content/29/38/11933)

5. **Martínez-Trujillo & Treue (2004), “Feature-based attention increases the selectivity of population responses in primate visual cortex.” *Current Biology*.**  
   An excellent bridge from your existing Treue paper to “sharpening.” **A population response can become more selective through differential gain across neurons without every individual neuron becoming more narrowly tuned.** That distinction is worth making explicitly in class. [Paper](https://pubmed.ncbi.nlm.nih.gov/15120065/)

6. **Reynolds & Heeger (2009), “The normalization model of attention.” *Neuron*.**  
   Provides a mechanistic framework for why attention can produce different apparent response changes under different stimulus and attention conditions. Good for response gain, contrast gain and stimulus competition. Distinguish a horizontal shift of a **contrast-response function** from narrowing an **orientation-tuning curve**. [Open paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2752446/)

7. **Pestilli, Carrasco, Heeger & Gardner (2011), “Attentional enhancement via selection and pooling of early sensory responses in human visual cortex.” *Neuron*.**  
   Adds another possibility: better performance through **which signals dominate the decision**. Their combination of psychophysics, fMRI and modeling favors a selection/pooling account over explaining behavior solely through greater sensitivity of the measured sensory responses. Useful for separating changes in encoding from changes in readout. [Paper](https://pubmed.ncbi.nlm.nih.gov/22153378/)

8. **Fox, Birman & Gardner (2023), “Gain, not concomitant changes in spatial receptive field properties, improves task performance in a neural network attention model.” *eLife*.**  
   This is unusually close to your proposed manipulation experiment: they separately examine gain, receptive-field shifts and shrinkage. In their network and tasks, gain explains the benefit, while the accompanying receptive-field changes are insufficient. **A good later discussion paper**, especially for asking whether an observed tuning change actually causes the behavioral improvement. Its conclusion is about the tested model and tasks, not a universal result about cortex. [Open paper](https://elifesciences.org/articles/78392)

There is also a valuable newer counterpoint to Luo–Maunsell:

**Chandrasekaran et al. (2024), “Dissociable components of attention exhibit distinct neuronal signatures in primate visual cortex.” *Science Advances*.** They dissociate covert attention from planned eye movements. In this task, V4 firing-rate modulation tracks choice bias at the covertly attended location, whereas reduced correlated variability tracks sensitivity at the planned saccade location. This makes a strong discussion pairing with Luo–Maunsell: **the mapping between firing-rate enhancement and improved sensitivity is not invariant across attention tasks.** [Open paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10836731/)

**For sharpening, shifting preferences and perceptual learning, I would add a separate set.** Rapid attentional modulation and changes after prolonged training should be distinguished, even when you simulate them using similar parameter changes.

| Paper | Why it matters for your examples |
|---|---|
| **[Schoups, Vogels, Qian & Orban (2001)](https://www.nature.com/articles/35087601)**, *Practising orientation identification improves orientation coding in V1 neurons*, **Nature** | A classic link between learning and improved local orientation coding. Particularly relevant to **changes in tuning-curve slopes around the trained orientation**, rather than simply “more neurons peak at the target.” |
| **[Raiguel et al. (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674023/)**, *Learning to see the difference specifically alters the most informative V4 neurons*, **J Neurosci** | Very close to your teaching goal: learning-related changes preferentially involve neurons informative for the trained discrimination. Pairs well with the slope-based analysis. |
| **[Womelsdorf et al. (2006)](https://doi.org/10.1038/nn1748)**, *Dynamic shifts of visual receptive fields in cortical area MT by spatial attention*, **Nature Neuroscience** | Direct evidence for receptive fields moving toward the attended location. This is **spatial receptive-field relocation during attention**, not necessarily a change in preferred motion direction or long-term learning. |
| **[Recanzone, Schreiner & Merzenich (1993)](https://www.jneurosci.org/content/13/1/87)**, *Plasticity in the frequency representation of primary auditory cortex following discrimination training in adult owl monkeys*, **J Neurosci** | A particularly direct example of your **increased representation of the trained stimulus range** idea: expansion of cortical representation around trained sound frequencies. |
| **[Polley, Steinberg & Merzenich (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6674159/)**, *Perceptual learning directs auditory cortical map reorganization through top-down influences*, **J Neurosci** | Useful for showing that reorganization depends on the **task-relevant stimulus dimension**, not merely repeated exposure. A good auditory complement to the visual examples. |
| **[Law & Gold (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2424192/)**, *Neural correlates of perceptual learning in a sensory-motor, but not a sensory, cortical area*, **Nature Neuroscience** | An essential alternative: learning-related improvement need not require better sensory tuning. Their MT/LIP comparison supports changes in how sensory evidence is interpreted or used. |

For a correlation-based learning extension, **[Gu et al. (2011), “Perceptual learning reduces interneuronal correlations in macaque visual cortex”](https://pmc.ncbi.nlm.nih.gov/articles/PMC3163063/)** adds another mechanism beyond changing mean tuning curves.

**There are several good reviews, with different teaching strengths. All of these are outside the original 40.**

| Review | Best use |
|---|---|
| **[Maunsell (2015), “Neuronal mechanisms of visual attention”](https://pmc.ncbi.nlm.nih.gov/articles/PMC8279254/)**, *Annual Review of Vision Science* | **My first general neural review for this class.** Covers neuronal modulation, population variability and connections to behavioral attention. |
| **[Reynolds & Chelazzi (2004), “Attentional modulation of visual processing”](https://doi.org/10.1146/annurev.neuro.26.041002.131039)**, *Annual Review of Neuroscience* | Detailed classic background on neural response changes and stimulus competition. Good lecture preparation; a substantial assigned reading. |
| **[Anton-Erxleben & Carrasco (2013), “Attentional enhancement of spatial resolution: linking behavioural and neurophysiological evidence”](https://www.nature.com/articles/nrn3443)**, *Nature Reviews Neuroscience* | **The closest review to your receptive-field shrinkage/relocation idea.** Explicitly connects receptive-field size, position and allocation to behavioral spatial resolution. |
| **[Luo & Maunsell (2019), “Attention can be subdivided into neurobiological components corresponding to distinct behavioral effects”](https://pmc.ncbi.nlm.nih.gov/articles/PMC6936701/)**, *PNAS* | **The most focused review for sensitivity versus criterion and their neural correlates.** Read with the later Chandrasekaran result in mind. |
| **[Verghese (2001), “Visual search and attention: a signal detection theory approach”](https://www.sciencedirect.com/science/article/pii/S0896627301003920)**, *Neuron* | Explicitly organizes attention and search through SDT. Especially useful for distractors, uncertainty and selection among multiple sensory signals. |
| **[Carrasco (2011), “Visual attention: the past 25 years”](https://pubmed.ncbi.nlm.nih.gov/21549742/)**, *Vision Research* | Broad behavioral coverage, including contrast sensitivity, spatial resolution and feature attention. Useful for connecting your neural manipulations to psychophysical tasks. |
| **[Ruff, Ni & Cohen (2018), “Cognition as a window into neuronal population space”](https://doi.org/10.1146/annurev-neuro-080317-061936)**, *Annual Review of Neuroscience* | A later extension from single-neuron tuning to population geometry and behaviorally relevant projections. Fits your scalar population readout approach, although conceptually more advanced. |

For the quantitative background, there is also **Pouget, Deneve & Latham (2001), “The relevance of Fisher information for theories of cortical computation and attention,”** in *Visual Attention and Cortical Circuits*. It is particularly relevant to translating tuning and noise into discrimination precision; you can introduce the local discrimination result without teaching Bayesian inference. [Book and chapter listing](https://direct.mit.edu/books/edited-volume/2997/Visual-Attention-and-Cortical-Circuits)

**I would make the notebook test the proposed improvements, rather than guarantee them.** These comparisons would give you a coherent progression:

| Manipulation | What students should discover |
|---|---|
| Multiply mean responses by \(g\), with Poisson spike counts | Mean separation scales as \(g\), noise SD as \(\sqrt g\), so the corresponding \(d'\) scales as \(\sqrt g\). |
| Multiply the already noisy response itself by \(g\) | Signal and noise scale together: discriminability is unchanged. This distinguishes added information from a change of units. |
| Add a stimulus-independent firing baseline | With fixed variance, mean separation is unchanged. With Poisson variability, the extra baseline adds noise and can worsen discrimination. |
| Narrow tuning curves | Improvement depends on where the alternatives lie, population coverage, and whether peak rate or total activity is held fixed. |
| Shift preferred stimuli toward the reference | For fine discrimination, useful **slopes** must cover the alternatives. Putting every symmetric tuning peak exactly at the midpoint can make the alternatives indistinguishable. |
| Change covariance while holding tuning fixed | Evaluate noise along the actual readout direction. Reducing every pairwise correlation is not a universal improvement rule. |
| Keep the encoding fixed and change readout weights or criterion | Separate improved use of available information from a different response policy. |

The gain-scaling examples above are consequences of the stated noise assumptions. The slope-versus-peak distinction is developed especially well by [Butts & Goldman](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0040092). For covariance, [Ruff & Cohen (2014), “Attention can either increase or decrease spike count correlations in visual cortex,”](https://www.nature.com/articles/nn.3835) is a useful counterweight to a blanket “decorrelation is good” interpretation.

For each manipulation, students could plot the **tuning curves → two scalar response distributions → ROC/\(d'\) → psychometric threshold**, first with fixed decoding weights and then with refitted weights. That would connect directly to the Jazayeri–Movshon material you have already taught.

---

## Added discussion: Pouget, Harper–McAlpine, and tuning-curve parameters

### User question

after that, didnt pouget talk about this aspect of sharpening tuning curves ? didnt mcalpine and nicol harper also ? even if not attention ? effectively if i look at impact of parameter x of tuning curveon rpresentation, attention could act on that parameter ?

### Full answer

**Yes—that is exactly the broader framing to use.** You can first ask how changing a property of the neural representation changes discrimination performance, then ask whether attention, adaptation, or learning produces that change. The coding paper does not itself have to study attention.

I focused too narrowly on attention-specific papers earlier. **Pouget’s work and Harper–McAlpine belong prominently in this discussion.**

These are the particularly relevant papers:

| Paper | Main point | Why it fits your lecture |
|---|---|---|
| **Pouget, Deneve, Ducom & Latham (1999), “Narrow versus wide tuning curves: What’s best for a population code?”** *Neural Computation.* [Full paper](https://physiology.med.cornell.edu/faculty/nirenberg/lab/pel/papers/nc-width.pdf) | Whether sharpening improves coding depends on the noise and its correlations. It also distinguishes information in the representation from performance with a particular decoder. **It explicitly discusses attention, sharpening, and gain.** | Probably the closest short reading to your exact question. It connects tuning-curve changes to discrimination while showing why curve shape alone cannot establish improvement. |
| **Seriès, Latham & Pouget (2004), “Tuning curve sharpening for orientation selectivity: coding efficiency and the impact of correlations.”** *Nature Neuroscience.* [Paper](https://www.nature.com/articles/nn1321) | Tests sharpening in spiking-network models. In their models, sharpening through lateral interactions produces correlations and substantial information loss compared with the alternative architecture. | The crucial follow-up: **changing a tuning-width parameter in an idealized encoding model and implementing sharpening through a neural circuit can have different consequences.** |
| **Harper & McAlpine (2004), “Optimal neural population coding of an auditory spatial cue.”** *Nature.* [Paper](https://www.nature.com/articles/nature02768) | Optimizes the distribution of preferred interaural time differences. Depending on head size and sound frequency, optimal tuning peaks can lie outside the naturally encountered range, placing informative slopes inside it. | Excellent for your idea of moving tuning curves around. It asks **where neurons should be tuned to discriminate the relevant stimuli most precisely**. |
| **Dean, Harper & McAlpine (2005), “Neural population coding of sound level adapts to stimulus statistics.”** *Nature Neuroscience.* [Paper](https://www.nature.com/articles/nn1541) | Auditory neurons adjust their response functions to sound-level statistics, improving population coding near frequently encountered levels. | An empirical example of redistributing coding precision by changing response properties. Adaptation supplies the manipulation; the computational question transfers naturally to attention. |

For **Harper–McAlpine**, your recollection is right about the broader issue of tuning and representation. Their 2004 paper particularly concerns **the placement of tuning curves relative to the relevant stimulus range**, rather than demonstrating attentional sharpening. Its Figure 2 would be especially useful for teaching the distinction between tuning peaks and informative slopes. [Full paper](https://science.umd.edu/biology/soareslab/NACS641web/hearing.pdf)

Another directly relevant theoretical paper is **Zhang & Sejnowski (1999), “Neuronal tuning: To sharpen or broaden?”** It shows that the effect of tuning width depends on the dimensionality of the encoded variable under their model assumptions—another reason sharpening has no universal benefit. [Paper](https://direct.mit.edu/neco/article/11/1/75/6232/Neuronal-Tuning-To-Sharpen-or-Broaden)

Your proposed organizing question could therefore be:

> **Which changes to a neural population make two nearby stimuli easier to distinguish, and which biological processes could implement those changes?**

That leads naturally to SDT. For a population read out into one scalar decision variable, \(z=\mathbf w^\top\mathbf r\), local discriminability is approximately

\[
d'_{\mathbf w}\approx
\frac{|\Delta s|\;|\mathbf w^\top\mathbf f'(s)|}
{\sqrt{\mathbf w^\top C(s)\mathbf w}},
\]

assuming a small stimulus difference and approximately common response covariance. Here, \(\mathbf f'(s)\) describes tuning-curve slopes, \(C\) describes response variability and correlations, and \(\mathbf w\) specifies the readout.

This gives you concrete manipulations to investigate:

- **Increase gain:** how much does the separation between stimulus-evoked responses increase, relative to the accompanying noise?
- **Sharpen tuning:** do the relevant slopes become more informative? What changes if you hold peak firing rate fixed versus total activity?
- **Shift preferred stimuli or redistribute tuning:** do more useful slopes cover the discrimination region? Concentrating peaks exactly at the target is not automatically optimal.
- **Change variability or correlations:** does noise decrease along the direction used to distinguish the stimuli?
- **Change readout weights:** can the observer use the existing representation more effectively?

**Attention could act on these parameters, yes.** A parameter analysis establishes the predicted consequence *if* attention changes that parameter. Physiological evidence then establishes whether attention actually does so, including any accompanying changes in noise or other parameters. Keeping those two questions separate lets you draw freely on adaptation, learning, and general coding papers.

For your teaching sequence, I would now choose **Pouget et al. (1999) as the short conceptual reading**, **Harper–McAlpine Figure 2 for the placement/slopes demonstration**, and **Seriès et al. (2004) for the deeper discussion of sharpening and correlations**. You can develop all of this through likelihoods and SDT before introducing Bayesian priors.

---

## Added discussion: Recent attention models, tuning changes, and the unknown decoder

### User question

newer and most recent developments along these lines ? also anything that connects that theory to specific attentional influences ? perhaps grace lindsay discusses ? limitation is we dont know decoder ?

### Full answer

**Yes—and Grace Lindsay is particularly relevant to the connection you want.** The newer work increasingly asks how a tuning or gain change affects the **specific downstream computation**, rather than treating sharper tuning or more activity as sufficient evidence of better perception.

**The unknown decoder is a central limitation, but it does not prevent us from studying the information available in a representation. It limits the inference from that information to the animal’s performance.**

Lindsay’s work addresses this directly:

**Lindsay & Miller (2018), “How biological attention mechanisms improve task performance in a large-scale visual system model,” *eLife*.** They introduce biologically inspired attentional gain into a convolutional network and measure its effects on task performance. Crucially, they compare modulation based on **what a unit prefers** with modulation based on **how changing that unit’s activity would improve the task output**. These can differ substantially: strong selectivity does not guarantee that increasing activity helps classification. Their gradient-based modulation can outperform tuning-based modulation, especially at intermediate layers. [Full paper](https://elifesciences.org/articles/38105)

This is unusually well suited to your lecture because it also explicitly uses **SDT to separate sensitivity and criterion**. In their tested conditions, feature attention predominantly affects criterion, whereas spatial attention predominantly affects sensitivity. That is a result of their model and tasks, rather than a universal division between attention types. Figures 2, 6 and 7 are especially relevant. [Results and figures](https://elifesciences.org/articles/38105)

Her **2020 review, “Attention in Psychology, Neuroscience, and Machine Learning,”** provides the broader context, including sensory modulation and altered readout as complementary mechanisms. I would use the review for orientation and the 2018 paper for the actual computational argument. [Review](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.00029/full)

The following papers extend the discussion, including directly relevant developments in **2025–2026**:

| Paper | Specific connection to your question |
|---|---|
| **Kozyrev, Daliri, Schwedhelm & Treue (2019), “Strategic deployment of feature-based attentional gain in primate visual cortex.”** *PLOS Biology.* [Paper](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000387) | **A direct physiological connection between attentional gain and useful tuning slopes.** In macaque MT, attentional modulation peaked along the flanks of the population activity profile rather than exactly at the target feature. Supports allocating gain to populations useful for distinguishing a target from similar distractors. |
| **Ruff & Cohen (2019), “Simultaneous multi-area recordings suggest that attention improves performance by reshaping stimulus representations.”** *Nature Neuroscience.* [Paper](https://www.nature.com/articles/s41593-019-0477-1) | Uses sensory and oculomotor population recordings plus behavior. Supports attention reshaping sensory activity so that it more effectively influences downstream processing. **The relationship between representation and readout becomes the explanatory object.** |
| **Ni, Huang, Doiron & Cohen (2022), “A general decoding strategy explains the relationship between behavior and correlated variability.”** *eLife.* [Paper](https://elifesciences.org/articles/67258) | **Probably the most direct answer to your decoder concern.** A decoder optimized over a broader range of orientations better matched the monkeys’ decoding strategy than one optimized for the particular change. Such a decoder is more affected by shared variability, explaining why attentional reductions in correlations can matter behaviorally even when a narrowly optimized decoder could largely ignore them. |
| **Fox, Birman & Gardner (2023), “Gain, not concomitant changes in spatial receptive field properties, improves task performance in a neural network attention model.”** *eLife.* [Paper](https://elifesciences.org/articles/78392) | **A particularly strong continuation of the sharpening/shift question.** Spatial gain produces downstream changes in receptive-field position, size and structure. By separating these effects within the model, they find that gain explains the performance benefit; the accompanying receptive-field changes are neither necessary nor sufficient in their tested setting. |
| **Abdirashid, Knapen & Dumoulin (2025), “The precision of attention controls attraction of population receptive fields.”** *Journal of Vision.* [Paper](https://doi.org/10.1167/jov.25.11.3) | Manipulates focused versus distributed attention and measures pRFs with 7T fMRI. Focused attention attracts pRFs more strongly toward the attended location; an attention-field model explains the pattern. **A specific attentional parameter—spatial precision—predicts changes in measured tuning.** These are population-level fMRI estimates. |
| **Tünçok, Carrasco & Winawer (2025), “Spatial attention selectively alters visual cortical representation during target anticipation.”** *Nature Communications.* [Paper](https://www.nature.com/articles/s41467-025-63795-3) | Combines psychophysics and fMRI. Before target onset, attention produces spatially selective baseline changes and shifts in pRF centers, alongside behavioral benefits at cued locations and costs elsewhere. Useful for showing that attention can prepare the representation before the relevant stimulus arrives. It does not isolate pRF shifts as the sole cause of improvement. |
| **Salehi, Lei, Benjamin, Müller & Kording (2026), “Modeling attention and binding in the brain through bidirectional recurrent gating.”** *Nature Communications*, published May 5. [Paper](https://www.nature.com/articles/s41467-026-72146-9) | A recent extension to a recurrent visual architecture: top-down and lateral signals multiplicatively modulate processing. The model performs orienting, filtering and visual-search tasks and reproduces several attentional and binding phenomena. **It connects a specified modulation mechanism to complete task behavior**, although this establishes computational sufficiency in the model rather than identifying the brain’s implementation. [Code](https://github.com/ssnio/bio-attention) |

The **Fox–Abdirashid–Tünçok combination** makes a useful teaching comparison: attention demonstrably changes measured spatial tuning, while the behavioral contribution of that change still requires an explicit computation or intervention. A tuning change can be useful, incidental, or part of a larger coordinated change.

On your decoder question, I would distinguish three claims:

1. **The representation contains more stimulus information.**  
   This concerns \(p(\mathbf r\mid s)\). In principle, it can be assessed without knowing the animal’s decoder.

2. **A specified decoder performs better.**  
   This is conditional on its weights, nonlinearities, integration window and downstream noise.

3. **The animal performs better because it uses that improvement.**  
   This requires evidence connecting the proposed representation and readout to behavior.

For two stimuli with mean-response difference \(\Delta\boldsymbol\mu\) and common covariance \(C\), the separation obtained by a linear readout is

\[
d_{\mathbf w}'^{\,2}
=
\frac{(\mathbf w^\top\Delta\boldsymbol\mu)^2}
{\mathbf w^\top C\mathbf w}.
\]

Optimizing over linear weights gives

\[
d_{\mathrm{best\ linear}}'^{\,2}
=
\Delta\boldsymbol\mu^\top C^{-1}\Delta\boldsymbol\mu.
\]

**These answer different questions.** Attention could rotate the stimulus-related activity toward the animal’s existing readout, improving its performance while leaving the best achievable linear separation unchanged. Conversely, a change could improve the best achievable separation without helping the weights the animal actually uses. Ruff–Cohen and Ni et al. provide concrete experimental and modeling approaches to this distinction. [Ruff & Cohen](https://www.nature.com/articles/s41593-019-0477-1), [Ni et al.](https://elifesciences.org/articles/67258)

Also, means and covariance characterize this **linear separation measure**, not the full information in arbitrary response distributions; higher-order dependencies can matter for nonlinear decoding.

We can constrain the decoder by comparing predictions across stimuli, tasks and attention conditions; testing whether weights generalize; recording downstream populations; and perturbing candidate pathways. Predicting choices alone is insufficient to establish causal readout, because feedback and unrecorded activity can also generate choice-related signals.

**For your class, my strongest pair would be Lindsay & Miller (2018) plus Ni et al. (2022).** The first connects specific attentional modulations to SDT and known downstream computation; the second makes uncertainty about biological decoding an explicit, testable hypothesis. If you want the lecture to stay centered on **gain versus receptive-field changes**, use Fox et al. (2023) as the main paper.

Ni et al. is also promising for your notebook: the paper provides [electrophysiological data on OSF](https://doi.org/10.17605/OSF.IO/RN7TU) and [simulation/analysis code](https://github.com/hcc11/GeneralDecoder). A useful exercise would compare each attentional manipulation under **fixed weights, refitted optimal weights, and weights trained across a broad stimulus range**.

---

<a id="recommended-reading-selection"></a>

## Added discussion: Choosing two papers for representation changes and attention

### User question

go through everything and pick 5 papers, or groups of 5 recommended papers from which i could pick two for my itnetion of covering various representation changes and how attention can influence..include those you recommended in discussion above as particularly relevant.. add to doc and giev me answer here as well

### Full answer

**For your stated lecture, my first choice is Pouget et al. (1999) + Ling, Liu & Carrasco (2009).** Pouget gives the general question—when do changes in tuning improve coding?—and Ling connects gain and population sharpening to specific attentional manipulations and discrimination thresholds. Together they support a notebook organized around tuning curves, response distributions, and SDT, before introducing Bayesian priors. [Pouget paper](https://physiology.med.cornell.edu/faculty/nirenberg/lab/pel/papers/nc-width.pdf), [Ling paper](https://sites.bu.edu/vision/files/2014/01/LingLiuCarrasco_2009.pdf)

I reviewed all 80 entries and the earlier recommendations. Below are **five groups of five papers**, selected for different parts of that teaching objective. The numbers are the document's existing reference IDs. **★ marks my five strongest individual candidates for your particular framing:** Pouget, Ling, Butts–Goldman, Lindsay–Miller, and Fox. The other papers provide empirical examples, broader coverage, or a different emphasis.

**A. General principles: what makes a changed representation better?**

| Paper | What it contributes to your class |
|---|---|
| **★ #41. [Pouget et al. (1999), Narrow versus wide tuning curves](https://physiology.med.cornell.edu/faculty/nirenberg/lab/pel/papers/nc-width.pdf)** | **My first conceptual reading.** A short treatment of sharpening, noise, and decoder dependence; explicitly discusses attention. Ideal for challenging the assumption that a narrower or taller curve must be a better code. |
| **★ #48. [Butts & Goldman (2006), Tuning curves, neuronal variability, and sensory coding](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0040092)** | **My first choice for the slopes-versus-peaks intuition.** Which stimuli are best encoded depends on noise and discrimination conditions. Particularly useful for your single-neuron or scalar-readout simulations. |
| **#43. [Harper & McAlpine (2004), Optimal neural population coding of an auditory spatial cue](https://www.nature.com/articles/nature02768)** | **My first choice for tuning placement.** Shows why optimal peaks can lie outside the stimulus range, positioning useful slopes within it. Figure 2 is a strong lecture illustration; the paper concerns coding theory, not an attention manipulation. |
| **#42. [Seriès, Latham & Pouget (2004), Tuning curve sharpening for orientation selectivity](https://www.nature.com/articles/nn1321)** | The deeper follow-up to Pouget: sharpening implemented through a circuit can bring correlations and information loss. Useful once students understand the difference between changing a model parameter and changing a circuit. |
| **#45. [Zhang & Sejnowski (1999), Neuronal tuning: To sharpen or broaden?](https://direct.mit.edu/neco/article/11/1/75/6232/Neuronal-Tuning-To-Sharpen-or-Broaden)** | Shows why the effect of tuning width also depends on stimulus dimensionality. A mathematically focused alternative when sharpening itself is the main topic. |

For a two-paper assignment that includes attention, pair one of these with a paper from B or D. Pouget is the best starting point for width/noise/readout; Butts–Goldman for local sensitivity; Harper–McAlpine for placement.

**B. Attention changing gain and population selectivity**

| Paper | What it contributes to your class |
|---|---|
| **★ #46. [Ling, Liu & Carrasco (2009), How spatial and feature-based attention affect gain and tuning](https://sites.bu.edu/vision/files/2014/01/LingLiuCarrasco_2009.pdf)** | **Closest overall match to the lecture you described.** Compares gain and population sharpening using attention, external noise, psychophysical thresholds, and a population model. Figure 1 is particularly useful. The sharpening is inferred within their model, not measured as narrower tuning in individual neurons. |
| **#25. [McAdams & Maunsell (1999), Effects of attention on orientation-tuning functions](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782389/)** | **Cleanest neuronal starting point.** Attention approximately scales V4 orientation responses without systematically narrowing tuning. Students can ask what that gain does to discriminability under different noise assumptions. |
| **#51. [Martínez-Trujillo & Treue (2004), Feature-based attention increases population selectivity](https://pubmed.ncbi.nlm.nih.gov/15120065/)** | Demonstrates the crucial distinction between sharpening a population activity profile through selective gain and sharpening each neuron's tuning curve. |
| **#49. [Scolari, Byers & Serences (2012), Optimal deployment of attentional gain during fine discriminations](https://pmc.ncbi.nlm.nih.gov/articles/PMC3384562/)** | Human fMRI and encoding-model evidence for enhancing populations tuned away from the target when those populations are useful for fine discrimination. A direct continuation of the Jazayeri–Movshon material. |
| **#74. [Kozyrev et al. (2019), Strategic deployment of feature-based attentional gain](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000387)** | **My preferred physiological companion to the informative-slopes argument.** Macaque MT attentional modulation emphasizes flanks of the population activity profile during target–distractor discrimination. |

For this group, the useful distinction is **uniform gain, selective gain across neurons, and individual-neuron tuning width**. These can produce different changes in the population response and should be separate notebook manipulations.

**C. Moving or reallocating representation: attention, learning, and adaptation**

| Paper | What it contributes to your class |
|---|---|
| **#58. [Womelsdorf et al. (2006), Dynamic shifts of visual receptive fields in MT by spatial attention](https://doi.org/10.1038/nn1748)** | **Clearest direct example of attention moving receptive fields toward the attended location.** This changes spatial sampling, rather than necessarily changing preferred motion direction. |
| **#77. [Abdirashid, Knapen & Dumoulin (2025), The precision of attention controls attraction of population receptive fields](https://doi.org/10.1167/jov.25.11.3)** | A recent human fMRI example connecting a specific attentional parameter—focused versus distributed attention—to the strength of receptive-field attraction. These are population receptive-field estimates. |
| **#56. [Schoups et al. (2001), Practising orientation identification improves orientation coding in V1](https://www.nature.com/articles/35087601)** | Training-related changes in tuning slopes around the trained orientation. Useful for showing how the same parameter questions extend to perceptual learning. |
| **#59. [Recanzone, Schreiner & Merzenich (1993), Plasticity in auditory cortical frequency representation](https://www.jneurosci.org/content/13/1/87)** | The most direct example in this shortlist of increased cortical representation of a trained stimulus range. Connects to your proposal of allocating more neurons to relevant values. |
| **#44. [Dean, Harper & McAlpine (2005), Sound-level coding adapts to stimulus statistics](https://www.nature.com/articles/nn1541)** | Response functions adapt so coding precision is redistributed toward commonly encountered sound levels. A strong auditory example of the general parameter argument. |

The first two manipulate attention; the next two concern learning; the last concerns adaptation. They let you ask a common computational question while keeping the biological processes and timescales explicit. Receptive-field relocation, changes in feature tuning, and cortical map expansion also describe different aspects of representation.

**D. Does the change improve performance through the available readout?**

| Paper | What it contributes to your class |
|---|---|
| **★ #72. [Lindsay & Miller (2018), How biological attention mechanisms improve task performance](https://elifesciences.org/articles/38105)** | **Best model-based bridge to your decoder question.** Modulating a unit according to its preference can differ from modulating it according to its influence on task output. Includes sensitivity/criterion analyses. The downstream computation is known within the network. |
| **★ #54. [Fox, Birman & Gardner (2023), Gain versus accompanying receptive-field changes](https://elifesciences.org/articles/78392)** | **Best direct comparison of candidate mechanisms in this shortlist.** Separates gain, receptive-field shifts, and shrinkage in an attention model. In their tested setting, gain accounts for the benefit; the accompanying receptive-field changes are neither necessary nor sufficient. |
| **#28. [Cohen & Maunsell (2009), Attention improves performance primarily by reducing interneuronal correlations](https://doi.org/10.1038/nn.2439)** | Adds covariance to a lecture otherwise centered on mean tuning. Their result is a strong physiological case study, with conclusions tied to the measured population and analysis. |
| **#75. [Ruff & Cohen (2019), Attention improves performance by reshaping stimulus representations](https://www.nature.com/articles/s41593-019-0477-1)** | Simultaneous recordings connect sensory representations to downstream processing. Good for explaining how an altered representation can become more useful to a particular readout. |
| **#76. [Ni et al. (2022), A general decoding strategy explains behavior and correlated variability](https://elifesciences.org/articles/67258)** | **Best biological-decoder follow-up.** Compares a decoder optimized for the particular discrimination with one trained across a broader orientation range. Includes public data/code and turns uncertainty about the decoder into a testable hypothesis. |

Lindsay + Ni is my preferred pair if decoder dependence becomes the central question. For the introductory lecture, use these papers to distinguish improved available information, improved performance of a specified readout, and demonstrated improvement in the animal.

**E. Reviews and an explicit SDT connection**

| Paper | What it contributes to your class |
|---|---|
| **#63. [Maunsell (2015), Neuronal mechanisms of visual attention](https://pmc.ncbi.nlm.nih.gov/articles/PMC8279254/)** | **Best broad neural review for this lecture.** Provides context for response modulation, population variability, and behavior. Pair with Ling for a review plus a concrete gain/tuning study. |
| **#65. [Anton-Erxleben & Carrasco (2013), Attentional enhancement of spatial resolution](https://www.nature.com/articles/nrn3443)** | **Best review for receptive-field size, position, and spatial allocation.** Helps connect those changes to psychophysical resolution. |
| **#73. [Lindsay (2020), Attention in Psychology, Neuroscience, and Machine Learning](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2020.00029/full)** | Best broad introduction to the biological/computational connection. Use Lindsay & Miller (72) when you want a specific model students can scrutinize. |
| **#47. [Luo & Maunsell (2015), Neuronal modulations correspond to one of multiple attention components](https://doi.org/10.1016/j.neuron.2015.05.007)** | **Best explicit sensitivity-versus-criterion study for the introduction.** Relates V4 modulation to sensitivity in their task. |
| **#55. [Chandrasekaran et al. (2024), Dissociable components of attention exhibit distinct neuronal signatures](https://pmc.ncbi.nlm.nih.gov/articles/PMC10836731/)** | A useful counterpoint: separates covert attention from planned eye movements and relates rate and covariance changes to different behavioral components. Shows why the SDT interpretation depends on task design. |

**The two-paper choices I would actually consider**

| Your emphasis | My pair | Why these two |
|---|---|---|
| **Your stated goal: tuning changes, discrimination, and attention** | **Pouget 1999 (#41) + Ling 2009 (#46)** | General coding principles plus an attention experiment/model comparing gain and population sharpening. **My default recommendation.** |
| **Informative slopes and where to allocate attention** | **Butts–Goldman 2006 (#48) + Kozyrev 2019 (#74)** | Why off-target neurons can be useful, then physiological evidence for allocating attentional gain accordingly. Harper–McAlpine (#43) is the alternative first paper if you want the auditory placement example. |
| **A straightforward physiological progression** | **McAdams–Maunsell 1999 (#25) + Cohen–Maunsell 2009 (#28)** | First change mean tuning responses; then examine shared variability and population sensitivity. |
| **Observed receptive-field changes versus their functional contribution** | **Womelsdorf 2006 (#58) + Fox 2023 (#54)** | Demonstration of attentional receptive-field shifts, then a model that separates shifts/shrinkage from gain. The model result does not settle the causal role of shifts in the biological experiment. |
| **Attention and the unknown decoder** | **Lindsay–Miller 2018 (#72) + Ni 2022 (#76)** | A specified downstream computation in a model, followed by evidence that biological decoding strategy changes the interpretation of neural variability. |
| **Broad coverage with one review** | **Maunsell 2015 (#63) + Ling 2009 (#46)** | General neural context plus a focused behavioral/model comparison of representational changes. |
| **Sensitivity versus criterion as the main learning objective** | **Luo–Maunsell 2015 (#47) + Chandrasekaran 2024 (#55)** | Contrasting task-dependent relationships between neuronal modulation and SDT components. |

For **Pouget + Ling**, I would use **Harper–McAlpine Figure 2** as an additional lecture illustration of tuning placement and **McAdams–Maunsell** as the physiological gain example. This gives the lecture broader coverage while keeping the assigned reading to two papers.

The notebook can ask the same question after each manipulation: **what happens to the two scalar response distributions and their discriminability?** Change gain, width, preferred stimulus/position, or covariance, then compare fixed and refitted readout weights. Introduce criterion as a separate decision parameter. That preserves your “one or many neurons → one computed variable → perception” thread and makes the uncertainty about the biological decoder explicit.
