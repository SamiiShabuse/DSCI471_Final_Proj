# DSCI 471: Final Project Rubric

---

## Final Project Point Distribution

| Component | Evaluated Artifact | Point Value |
|---|---|---:|
| **Phase 1** | In-Class Presentation | 50 Points |
| **Phase 2** | Final Report / Jupyter Notebook | 300 Points |
| **Total** |  | **350 Points** |

---

## Part A: In-Class Presentation (50 Points)

- **Format:** Brief, free-form presentation (5–7 minutes) to share findings with the class.
- **Grading:** Full credit will be awarded for active participation and an informative presentation of the project's current trajectory (Acceptable as a work-in-progress).

### Presentation Score Breakdown

- **Structure & Outline (20 pts):** Follows a logical flow summarizing the core components: Problem Definition → Dataset Exploration → Methodology → Evaluation Metrics.
- **Clarity & Communication (20 pts):** Delivering a clear, informative presentation that successfully conveys the project's scope, motivations, and current results to peers.
- **Experiential Reflection (10 pts):** Thoughtfully sharing the teamwork dynamic and practical model training/testing experiences (e.g., debugging challenges, optimization hurdles).

---

## Part B: Final Report Documentation (300 Points)

Evaluation will heavily weight critical thinking, architecture justification, and proper analytical interpretation over high leaderboard scores or complex raw code.

### 1. Problem Definition & Clarity (40 Points)

*The baseline evaluation of the research definition.*

- **Excellent (36–40 pts):** The research problem is perfectly scoped, highly relevant to data science, and backed by strong, clear domain motivations.
- **Proficient (28–35 pts):** The problem is well-defined and practical, though the description of its domain significance or specific research questions could be slightly expanded.
- **Developing (1–27 pts):** The scope is vague, lacks clear real-world motivation, or has minimal connection to data science applications.

### 2. Data Collection & Exploratory Analytics (50 Points)

*The preparation and processing of textual, visual, or numerical sequence data.*

- **Excellent (45–50 pts):** Data collection is technically sound. Demonstrates deep exploratory data analysis, appropriate preprocessing or data engineering, and explicit handling of spatial or temporal sequence constraints (e.g., masking, resizing, padding).
- **Proficient (35–44 pts):** The data pipeline is functional and clean. Data traits are adequately explored, but preprocessing steps or feature dimension tracking could be more explicitly documented.
- **Developing (1–34 pts):** Minimal exploratory analysis provided. The dataset lacks the structural scale or documentation necessary to sustain a deep learning pipeline.

### 3. Intended Methodology & Model Justification (70 Points)

*The explicit deep learning structural design using TensorFlow/Keras or other relevant frameworks.*

- **Excellent (63–70 pts):** Clear, robust implementation of modern architectures (e.g., CNNs, RNNs, Transformers, or RL). Includes excellent **conceptual justification** for why the specific hidden layers, activation functions (e.g., ReLU, GELU), regularization (Dropout, LayerNorm), or model streams were selected for this problem domain.
- **Proficient (49–62 pts):** The model architecture is technically correct and functional. The model selection is logical, but the underlying text lacks deeper commentary explaining the specific structural parameters or layer choices.
- **Developing (1–48 pts):** The methodology lacks technical soundness or relevance to the course. Code blocks appear copied without an understanding of the layers, or the model choices are left completely unjustified.

### 4. Technical Evaluation & Architecture Comparison (70 Points)

*The comparative performance analysis.*

- **Excellent (63–70 pts):** Model performance is evaluated using appropriate multi-dimensional tracking metrics (e.g., cross-entropy validation tracking, perplexity when applicable). Includes a structured comparison between different configurations, ablation setups, baseline variants, OR alternative hyperparameter dimensions.
- **Proficient (49–62 pts):** Evaluation metrics are correctly applied and tracked. The model is contrasted across baseline metrics, but the comparison lacks broader architectural depth or comprehensive validation logs.
- **Developing (1–48 pts):** Weak evaluation setup. The project relies on a single metric run without baseline comparisons, or shows an inadequate understanding of proper validation protocols.

### 5. Interpretation of Results & Critical Thinking (50 Points)

*The analytical evaluation of model outcomes.*

- **Excellent (45–50 pts):** Provides exceptionally thoughtful interpretation of output tensors and attention maps. Demonstrates clear understanding of whether the system overfitted, underfitted, or hit an architecture-specific ceiling, complete with logical data-driven explanations.
- **Proficient (35–44 pts):** Results are accurately interpreted and discussed. The analysis identifies key performance trends, but stays mostly superficial rather than explaining the underlying model behavior.
- **Developing (1–34 pts):** Missing or flawed interpretation of results. The report provides tables or plots without extracting any meaningful data science insights or analytical conclusions.

### 6. Technical Documentation & Rigor (20 Points)

*The general delivery structure of the project artifact.*

- **Excellent (18–20 pts):** The notebook/report is meticulously organized, utilizing clear markdown cell headers and cohesive inline documentation. Code blocks execute flawlessly, and technical terminology is used with high precision.
- **Proficient (14–17 pts):** Clear and readable documentation. Code blocks execute correctly, though formatting consistency or inline explanations could be polished.
- **Developing (1–13 pts):** Poor presentation and messy notebook layout. Contains broken execution graphs, missing asset descriptions, or unreadable blocks of code.
