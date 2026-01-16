# GPT-2 and Language Neuroscience

Investigating the relationship between computational language models (GPT-2) and human brain activity during natural reading using EEG.

## Overview

This project bridges artificial intelligence and cognitive neuroscience by:
1. **Computing linguistic predictability** using GPT-2 surprisal values
2. **Measuring neural responses** via EEG during natural story reading
3. **Testing computational-neural alignment** between LLM predictions and brain activity

### Core Research Question
**Can computational language models like GPT-2 predict patterns of human brain activity during reading?**

If yes, this suggests LLMs capture similar statistical regularities that the human brain uses for language processing.

## Repository Structure

```
├── GPT2_Surprisal.ipynb          # Compute surprisal values
├── GPT_2_EEG.ipynb                # Analyze EEG with surprisal
└── README.md                       # This file
```

## Technical Stack

- **Languages:** Python, R
- **AI/ML:** PyTorch, HuggingFace Transformers, GPT-2
- **Neuroscience:** EEG/ERP analysis, mixed-effects modeling
- **Key Libraries:** 
  - Python: `transformers`, `torch`, `numpy`, `pandas`
  - R: `lme4`, `dplyr`, `ggplot2`, `broom.mixed`

## Part 1: GPT-2 Surprisal Calculation

### What is Surprisal?

**Surprisal** quantifies how unexpected a word is given its context:
```
Surprisal = -log₂(P(word | context))
```

- **High surprisal** = unexpected word (e.g., "The cat sat on the **refrigerator**")
- **Low surprisal** = predictable word (e.g., "The cat sat on the **mat**")

### Implementation (`GPT2_Surprisal.ipynb`)

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Load GPT-2
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

# Calculate surprisal for each word in the corpus
# (See notebook for full implementation)
```

### Output
- Word-by-word surprisal values for Natural Stories corpus
- Cosine similarity between GPT-2 hidden states
- Integration with linguistic features (frequency, length, etc.)

## Part 2: EEG-Surprisal Integration

### Methodology (`GPT_2_EEG.ipynb`)

1. **ERP Extraction**
   - Time-lock EEG signals to word onsets
   - Extract event-related potentials (ERPs) 
   - Apply baseline correction (-750 to 0 ms pre-stimulus)

2. **Statistical Modeling**
   ```r
   # Mixed-effects regression
   lmer(ERP_amplitude ~ gpt2_surprisal + 
                         word_frequency + 
                         word_length +
                         (1|subject) + 
                         (1|word), 
        data = combined_data)
   ```

3. **Hypothesis Testing**
   - Do brain responses scale with GPT-2 surprisal?
   - Which EEG components (N400, P600) correlate with predictability?
   - Does GPT-2 explain variance beyond traditional linguistic features?

### Key Features

- Custom ERP extraction pipeline handling variable time windows
- Robust baseline correction procedures
- Integration of corpus-level annotations
- Mixed-effects models accounting for subject/item variability

## Dataset

**Natural Stories Corpus** (Futrell et al., 2021)
- 10 naturalistic stories designed for reading comprehension research
- Word-by-word timing information
- EEG recordings from multiple participants
- Rich linguistic annotations

## Research Context

### Why This Matters

1. **Cognitive Science:** Tests whether LLMs capture human-like language processing
2. **AI Alignment:** Evaluates how well artificial and biological intelligence align
3. **Neuroscience:** Provides computational models of predictive processing in the brain
4. **Applied HCI:** Enables biosensor-based interfaces that adapt to cognitive load

### Theoretical Framework

- **Surprisal Theory** (Hale, 2001; Levy, 2008): Reading difficulty scales with surprisal
- **Predictive Coding** (Friston, 2005): The brain minimizes prediction error
- **N400 Effect:** Neural marker of semantic prediction violations

## Key Findings (Typical Results)

- GPT-2 surprisal correlates with N400 amplitude
- Stronger effects for content words vs. function words
- LLMs explain variance beyond word frequency
- Individual differences in brain-model alignment

## Applications

### Scientific
- Benchmark for evaluating language model cognitive plausibility
- Test computational theories of language comprehension
- Understand neural mechanisms of prediction

### Applied
- **Adaptive reading interfaces:** Adjust text based on predicted difficulty
- **Educational technology:** Identify challenging passages in real-time
- **Accessibility tools:** Design better text-to-speech systems
- **BCI applications:** Brain-computer interfaces for communication

## Performance

- **Surprisal computation:** ~100-500 words/second (CPU/GPU)
- **ERP extraction:** Real-time capable for online experiments
- **Statistical models:** Convergence within minutes on full dataset

## Usage Example

```r
# 1. Generate surprisal (Python - GPT2_Surprisal.ipynb)
surprisal_data <- read_csv("gpt2_surprisal.csv")

# 2. Extract ERPs (R - GPT_2_EEG.ipynb)
erp_data <- extract_ERP(subject, word_num, raw_eeg, story_info)

# 3. Combine and model
combined <- left_join(erp_data, surprisal_data)
model <- lmer(N400 ~ surprisal + (1|subject), data = combined)
```

## Validation

- **Cross-validation:** Test on held-out subjects
- **Permutation tests:** Confirm effects aren't due to chance
- **Alternative models:** Compare GPT-2 vs. simpler n-gram models
- **Robustness checks:** Test across different EEG components and time windows

## Future Directions

- Test newer models (GPT-3, GPT-4, Claude)
- Real-time prediction during reading
- Multi-modal integration (eye-tracking + EEG)
- Individual differences in brain-AI alignment
- Clinical applications (reading disorders, aphasia)
- Adaptive interfaces that adjust based on neural signals

## Related Projects

Check out my other computational cognitive science work:
- [Predictive Coding Networks](https://github.com/USERNAME/predictive-coding-network)
- [LSTM Working Memory Analysis](https://github.com/USERNAME/lstm-rnn-working-memory)

## References

### Key Papers
- **Futrell et al. (2021):** Natural Stories Corpus
- **Radford et al. (2019):** GPT-2 - Language Models are Unsupervised Multitask Learners
- **Hale (2001):** A Probabilistic Earley Parser as a Psycholinguistic Model
- **Levy (2008):** Expectation-based syntactic comprehension
- **Friston (2005):** A theory of cortical responses
- **Kutas & Hillyard (1980):** Reading senseless sentences: N400

### Datasets
- Natural Stories: https://github.com/languageMIT/naturalstories
- GPT-2: https://huggingface.co/gpt2


## Contact

**Yasemin Gokcen**  
PhD Student, Cognitive & Information Sciences  
University of California, Merced  
📧 ygokcen@ucmerced.edu

---
