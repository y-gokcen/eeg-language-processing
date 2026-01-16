# EEG-Language Processing: Integrating Neural Activity with GPT-2 Surprisal

Analyzing the relationship between brain activity (EEG) and language processing using GPT-2 surprisal as a computational model of linguistic prediction.

## Overview

This project investigates how neural responses during reading relate to predictability in natural language. By combining Event-Related Potential (ERP) data from EEG recordings with GPT-2 surprisal values, we can explore how the brain processes unexpected words during natural story comprehension.

### Key Questions
- How do neural responses correlate with word predictability?
- Can large language models (GPT-2) predict patterns of brain activity during reading?
- What does this tell us about human language processing mechanisms?

## Technical Stack

- **Languages:** R
- **Key Libraries:** `lme4`, `data.table`, `dplyr`, `ggplot2`, `broom.mixed`
- **Data:** Natural Stories Corpus (Futrell et al., 2021)
- **Methods:** 
  - ERP extraction and baseline correction
  - Mixed-effects regression modeling
  - Time-series analysis of neural signals

## Methodology

1. **ERP Extraction:** Extract event-related potentials from continuous EEG data, time-locked to word onsets
2. **GPT-2 Surprisal:** Calculate word-by-word surprisal values using GPT-2 language model
3. **Statistical Modeling:** Use linear mixed-effects models to test relationships between neural activity and surprisal
4. **Visualization:** Generate topographic maps and time-course plots of neural responses

## Key Features

- Custom ERP extraction pipeline handling variable-length time windows
- Baseline correction to control for pre-stimulus activity
- Integration of linguistic features (frequency, length) with neural data
- Robust statistical modeling accounting for by-subject and by-item variability

## Dataset

This analysis uses the [Natural Stories Corpus](https://github.com/languageMIT/naturalstories), which provides:
- Naturalistic reading materials (stories)
- Word-by-word timing information
- EEG recordings from multiple subjects
- Linguistic annotations

## Usage

```r
# Load required libraries
library(lme4)
library(dplyr)
library(ggplot2)

# Run the analysis pipeline
# (See notebook for full workflow)
```

## Research Context

This work contributes to understanding:
- **Predictive processing** in the brain during language comprehension
- **Human-AI alignment** between neural and computational language models
- **Biosensor applications** for measuring cognitive processes during natural behavior

## Related Work

- Futrell, R., Gibson, E., & Levy, R. P. (2021). Natural Stories Corpus
- GPT-2: Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners
- Predictive coding frameworks in neuroscience (Friston, Clark)

## Future Directions

- Extend to other language models (GPT-3, Claude, etc.)
- Real-time prediction of neural responses
- Applications to adaptive reading interfaces

---

**Author:** Yasemin Gokcen  
**Collaborators/Advisors:** Dr. Rachel Ryskin, Dr. David Noelle
**Affiliation:** PhD Candidate, Cognitive & Information Sciences, UC Merced  
**Contact:** ygokcen@ucmerced.edu
