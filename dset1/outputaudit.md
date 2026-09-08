# Figure output audit

The generated plots reproduce the paper’s data and trends, but they are raw analysis outputs rather than publication-ready replicas. The main substantive discrepancy is Figure 3’s time-axis labeling.

| Generated output | Published figure | Comparison |
|---|---|---|
| [Figure_2a.png](results/Figure_2a.png) | Figure 2a | Curves, points, asymptotes, colors, and error bars visually match. The paper substitutes triangle markers, adds a legend/lightning icons and grid, and uses polished typography. |
| [Figure_2b.png](results/Figure_2b.png) | Figure 2b | Individual participant points and the identity/zero lines match. The generated version omits the published group-mean ±SEM crosses and explanatory annotations. |
| [Figure_3.png](results/Figure_3.png) | Figure 3a–d | All six data panels and trends match. However, the generated x axes increase `0 → 200 ms`, while the paper labels them `200 → 0 ms`. The source script explicitly assigns the increasing labels at [plotres.m:193](osfstorage/Code/plotres.m#L193). The paper also adds significance brackets, asterisks, condition-specific markers, legends, and panel labels. |
| [Figure_S1a.png](results/Figure_S1a.png) and [Figure_S1b.png](results/Figure_S1b.png) | Supplementary Figure 1a–b | Curves, observations, asymptotes, and errors visually match. The publication combines them and adds cue diagrams, marker semantics, citations, legends, and significance annotations. |
| [Figure_S2.png](results/Figure_S2.png) | Supplementary Figure 2 | Means, error bars, line shapes, limits, and 2×2 arrangement match. The publication uses yellow triangles/open inverted triangles, labels Exp. 2a/2b, and includes a legend; the raw code produces grayscale lines without markers. |

## Conclusion

I found no apparent discrepancy in the plotted numerical results. The published figures underwent substantial manual/editorial styling after the supplied MATLAB script was run. Figure 3’s reversed time labeling is the only scientifically meaningful presentation difference and should be corrected before reusing the generated image.

## References used

- Main paper PDF: [2023_NatComm_HanningFernandezCarrasco.pdf](osfstorage/Publication/2023_NatComm_HanningFernandezCarrasco.pdf) — Figures 2 and 3.
- Downloaded supplementary PDF: [Supplementary Information](osfstorage/Publication/2023_NatComm_HanningFernandezCarrasco_Supplementary_Information.pdf) — Supplementary Figures 1 and 2.
- Online [PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10477327/); supplementary files were retrieved through the official [Europe PMC supplementary-files API](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10477327/supplementaryFiles).

## Contents of the CSV and Excel data

The two formats represent different stages of data processing:

- The 26 `Sxx_resMat.csv` files contain subject-level, trial-by-trial behavioral and eye-movement results: 10 subjects in Experiment 1, 9 in Experiment 2a, and 7 in Experiment 2b. Each row represents a trial and contains 13 numeric, unlabeled columns. The columns include categorical condition codes, response/correctness fields, and continuous eye-movement measurements. Based on their ranges and the methods in the paper, column 12 is very likely saccade latency in milliseconds, while column 13 appears to be a saccade-amplitude or landing-position measurement in degrees. No data dictionary was supplied, so the exact identities of all 13 columns cannot be established confidently.

- [Source_Data.xlsx](<osfstorage/Code/Source Data/Source_Data.xlsx>) contains processed, participant-level values used to create the publication figures. It is not raw trial data. Its six worksheets contain:

  - `Figure 2a`: d-prime across six contrasts, fitted `d_max`, validity, and stimulation condition.
  - `Figure 2b`: valid-minus-invalid `d_max` effects for presaccadic, exogenous, and endogenous attention.
  - `Figure 3`: d-prime in four time bins before saccade onset, separated by validity, stimulation, and TMS site.
  - `Figure S1a`: contrast-response data for exogenous covert attention.
  - `Figure S1b`: contrast-response data for endogenous covert attention.
  - `Figure S2`: saccade latency and landing error across TMS timings, separated by saccade direction and TMS site.

The CSV files contain 24,035 rows for Experiment 1, 3,778 for Experiment 2a, and 4,712 for Experiment 2b. These totals do not match the final included-trial counts reported in the paper. The CSV matrices therefore appear to represent a different preprocessing or selection stage, rather than exactly the final analysis table. The paper describes the experimental conditions and exclusion rules but does not provide a 13-column CSV data dictionary.

The supplied `plotres.m` script does not read either the CSV files or the Excel workbook. It reads prepared `fig*.mat` files. `Source_Data.xlsx` is effectively the human-readable figure-source-data counterpart to those MATLAB files.
