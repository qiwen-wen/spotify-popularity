# Predicting Track Popularity from Audio Features

**Author:** Ella Wen (q2wen@ucsd.edu)

*A DSC 80 (UC San Diego) final project exploring what makes a song popular on Spotify.*

---

## Introduction

Streaming platforms like Spotify describe every track with a set of machine-generated **audio features** — numbers capturing how danceable, energetic, acoustic, or upbeat a song is — alongside metadata such as its genre, release date, and whether it contains explicit lyrics. At the same time, Spotify assigns each track a **popularity** score from 0 to 100, based largely on recent play counts. This project asks a question that matters to artists, producers, and the platforms that curate playlists:

> **What audio and metadata features are associated with a track's popularity?**

If certain measurable qualities of a song are consistently linked to popularity, that information is valuable for anyone deciding what to produce, promote, or surface to listeners.

The dataset is **Spotify Music Tracks**, containing **114,000 rows** — 1,000 tracks for each of **114 genres**. Each row is a single track. The columns most relevant to this analysis are:

| Column | Description |
| --- | --- |
| `popularity` | Spotify popularity score (0–100), based on total plays and recency |
| `track_genre` | Genre label assigned by Spotify (114 genres) |
| `danceability` | How suitable a track is for dancing (0–1) |
| `energy` | Perceptual measure of intensity and activity (0–1) |
| `loudness` | Overall loudness in decibels (dB) |
| `tempo` | Estimated tempo in beats per minute (BPM) |
| `valence` | Musical positiveness conveyed by the track (0–1) |
| `acousticness` | Confidence the track is acoustic (0–1) |
| `instrumentalness` | Predicts whether a track contains no vocals (0–1) |
| `speechiness` | Presence of spoken words (0–1) |
| `liveness` | Probability the track was performed live (0–1) |
| `explicit` | Whether the track has explicit lyrics (boolean) |
| `duration_ms` | Track length in milliseconds |
| `release_date` | Release date of the track |

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

I performed the following cleaning steps, each motivated by the data generating process:

1. **Dropped the `Unnamed: 0` column**, which was just a stored row index carrying no information.
2. **Parsed `release_date` into a `release_year`** (and later a `decade`) column. Release dates appear in mixed formats (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`), so I parsed them with a mixed-format datetime parser and extracted the year.
3. **Replaced invalid sentinel values with `NaN`.** A handful of tracks had a `duration_ms` of 0, a `tempo` of 0, or a `time_signature` of 0. These are physically impossible for a real recording and almost certainly indicate a failed measurement rather than a true value, so I converted them to `NaN` so they would be treated as missing rather than dragging down averages. (89 tracks had tempo = 0; these were folded in with the existing missing tempos.)
4. **Created `duration_min`** from `duration_ms` (dividing by 60,000) for easier interpretation.
5. **Confirmed `explicit` is already boolean**, so no conversion was needed.

The cleaned DataFrame has 114,000 rows. Its head (selected columns) looks like:

| track_name | popularity | explicit | danceability | energy | loudness | tempo | track_genre | release_year |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Comedy | 73 | False | 0.676 | 0.461 | -6.746 | 87.917 | acoustic | 1974 |
| Ghost - Acoustic | 55 | False | 0.420 | 0.166 | -17.235 | 77.489 | acoustic | 1995 |
| To Begin Again | 57 | False | 0.438 | 0.359 | -9.734 | 76.332 | acoustic | 1973 |
| Can't Help Falling In Love | 71 | False | 0.266 | 0.060 | -18.515 | 181.740 | acoustic | 2018 |
| Hold On | 82 | False | 0.618 | 0.443 | -9.681 | (missing) | acoustic | 2017 |

### Univariate Analysis

The distribution of **popularity** is roughly bell-shaped but skewed, with a large spike of tracks at popularity 0 (tracks that have essentially no plays). Most tracks cluster in the 20–50 range, and very few tracks reach the high-popularity tail above 70.

<iframe
  src="assets/univariate-popularity.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

The distribution of **release year** is heavily right-skewed toward recent years: the catalog is dominated by music released in the 2000s and 2010s, with a long thin tail reaching back to the 1920s.

<iframe
  src="assets/univariate-year.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

### Bivariate Analysis

Plotting **mean popularity against release year** shows a clear upward trend: newer tracks tend to be more popular. This makes sense given that Spotify's popularity score weights *recent* plays, so older catalog tracks are at a structural disadvantage.

<iframe
  src="assets/bivariate-year.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

Comparing popularity for **explicit vs. non-explicit** tracks, explicit tracks have a noticeably higher median popularity and a distribution shifted upward. This motivates the formal hypothesis test below.

<iframe
  src="assets/bivariate-explicit.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

### Interesting Aggregates

Grouping by **genre** and summarizing popularity highlights how strongly genre relates to popularity. The ten most popular genres by mean popularity:

| track_genre | mean_popularity | median_popularity | track_count |
| --- | --- | --- | --- |
| pop-film | 59.28 | 60.0 | 1000 |
| k-pop | 56.90 | 60.0 | 1000 |
| chill | 53.65 | 57.0 | 1000 |
| sad | 52.38 | 54.0 | 1000 |
| grunge | 49.59 | 55.0 | 1000 |
| indian | 49.54 | 49.0 | 1000 |
| anime | 48.77 | 50.0 | 1000 |
| emo | 48.13 | 51.0 | 1000 |
| sertanejo | 47.87 | 47.0 | 1000 |
| pop | 47.58 | 66.0 | 1000 |

I also built a pivot table of **mean popularity by decade and explicit flag**. Within almost every decade, explicit tracks are more popular than non-explicit ones, and the gap widens in recent decades (e.g. the 2010s and 2020s), suggesting the explicit/popularity relationship is partly a modern phenomenon.

| decade | non-explicit | explicit |
| --- | --- | --- |
| 1960 | 28.71 | 33.55 |
| 1970 | 30.30 | 33.30 |
| 1980 | 30.82 | 33.03 |
| 1990 | 31.61 | 33.80 |
| 2000 | 34.72 | 35.06 |
| 2010 | 38.24 | 43.44 |
| 2020 | 33.06 | 39.24 |

---

## Assessment of Missingness

### NMAR Analysis

The only column with substantial missingness is **`tempo`**, which is missing for about **22,114 tracks (~19%)**. I do **not** believe `tempo` is **NMAR** (Not Missing At Random). For tempo to be NMAR, the probability that it is missing would have to depend on the *unobserved tempo value itself* — for example, Spotify's audio-analysis pipeline systematically failing to lock onto a beat for extremely slow/ambient or extremely fast tracks. I cannot test that directly, because the missing tempos are unobserved by definition.

However, the permutation tests below show that tempo missingness depends strongly on **observed** features such as `energy` and `loudness`. That dependence on other observed columns is exactly the signature of **MAR** (Missing At Random), which is the most defensible description here. To definitively confirm or rule out NMAR, I would need additional data — for instance, the raw audio for the missing tracks, or the analyzer's confidence score for each tempo estimate.

### Missingness Dependency

I tested whether `tempo` missingness depends on other columns using permutation tests, with the **absolute difference in group means** (missing vs. not-missing) as the test statistic and a 0.05 significance level.

**Depends on `energy` (MAR).** Tracks with missing tempo have substantially lower mean energy (0.512) than tracks with observed tempo (0.673). The permutation test gives an observed absolute difference of **0.1611** and a **p-value of 0.0000**, so I reject the null — tempo missingness *does* depend on energy.

<iframe
  src="assets/missingness-energy.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

**Does not depend on `popularity`.** Mean popularity is nearly identical whether tempo is missing (33.31) or present (33.22). The permutation test gives an observed difference of just **0.0863** and a **p-value of ~0.60**, so I fail to reject the null — there is no evidence that tempo missingness depends on popularity.

Together these results support describing `tempo` as **MAR**: its missingness is explained by other observed audio characteristics (energy, loudness), not by popularity and not (as far as the data can show) by tempo itself.

---

## Hypothesis Testing

I tested whether explicit content is associated with higher popularity.

- **Null hypothesis (H₀):** Explicit and non-explicit tracks have the same mean popularity; any observed difference is due to random chance.
- **Alternative hypothesis (H₁):** Explicit tracks have a *higher* mean popularity than non-explicit tracks.
- **Test statistic:** difference in mean popularity, (mean popularity of explicit tracks) − (mean popularity of non-explicit tracks). This directional statistic matches the one-sided alternative.
- **Significance level:** 0.05.

Using a **permutation test** with 1,000 shuffles of the popularity labels, the observed difference was **+3.52** popularity points (explicit tracks higher), and the **p-value was 0.0000** (no permutation reached the observed difference).

<iframe
  src="assets/hypothesis-test.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

Because the p-value is far below 0.05, I reject the null hypothesis. The data are consistent with explicit tracks being more popular on average than non-explicit tracks. This is an observational, probabilistic conclusion, not proof of causation — explicit tracks skew toward more recent, heavily-streamed genres, which could partly explain the association.

---

## Framing a Prediction Problem

**Problem type:** binary **classification** — predict whether a track is **popular** from features available before its play counts accumulate.

**Response variable:** `is_popular`, defined as `1` if `popularity >= 50` and `0` otherwise. I binarize the continuous popularity score because "popular vs. not" is the concrete, actionable decision a playlist or A&R team actually makes. The threshold of 50 was chosen because it produces a workable, roughly balanced class split (~40% popular) on the modeling subset; a threshold of 70 would leave too few positive examples, while splitting at the median would be arbitrary and carry no semantic meaning.

**Modeling subset (genres).** As required for this dataset, modeling is restricted to a selection of **six musically distinct genres**: `hard-rock`, `acoustic`, `edm`, `disco`, `folk`, and `indian`. I chose these because (a) they span the full range of energy, acousticness, danceability, and instrumentalness, so audio features are genuinely informative, and (b) each genre has internal popularity variation (roughly 30–50% popular at threshold 50), so the problem isn't trivially solved by genre alone. The filtered subset has 6,000 tracks.

**Evaluation metric:** **F1-score.** The classes are imbalanced (~40% popular), so plain accuracy is misleading — a model can look acceptable simply by leaning toward the majority class. F1 balances precision and recall, which fits the goal of correctly flagging popular tracks without generating too many false positives.

**Time-of-prediction / leakage.** At prediction time I assume I know only what is available at release: the track's audio features, genre, explicit flag, and release decade. I do **not** use `popularity` itself or anything derived from future play counts as a feature.

---

## Baseline Model

The baseline is a **logistic regression** trained in a single scikit-learn `Pipeline`.

**Features (3 original columns):**
- `danceability` — quantitative
- `energy` — quantitative
- `track_genre` — nominal (categorical)

So the baseline uses **2 quantitative** features and **1 nominal** feature (and no ordinal features).

**Encodings.** The two quantitative features were standardized with `StandardScaler` (logistic regression is sensitive to feature scale). The nominal `track_genre` was one-hot encoded with `drop='first'` to avoid the dummy-variable trap. All preprocessing and the classifier live inside one `Pipeline`, and the model was evaluated on a held-out 20% test set (stratified 80/20 split).

**Performance (test set):**

| Metric | Score |
| --- | --- |
| Accuracy | 0.553 |
| F1-score | 0.521 |
| Precision | 0.460 |
| Recall | 0.599 |

This baseline is **not good**. An F1 of 0.52 and accuracy of 0.55 are only modestly better than guessing, which is expected: just three features (two audio dimensions plus genre) can't capture much of what drives popularity. It serves its purpose as a floor to improve on.

---

## Final Model

The final model is a **random forest classifier**, again wrapped in a single `Pipeline`, trained on the **same train/test split** as the baseline so the comparison is fair.

**Engineered features (beyond the original columns and encodings):**
- **`loudness_per_energy`** = `loudness / (energy + 0.01)` — captures how *efficiently* a track converts energy into loudness. Two tracks can have the same energy but very different production styles; this ratio reflects mastering/compression choices that relate to a track's "modern, radio-ready" sound. (The `+ 0.01` avoids division by zero.)
- **`is_long_track`** = `1` if `duration_min > 4` — a binary flag for unusually long tracks. Very long tracks tend to be less playlist-friendly, which is plausibly related to popularity in a non-linear way a tree can exploit.

**Full feature set and transformations (all inside the Pipeline):**
- `StandardScaler` on roughly symmetric audio features (`danceability`, `energy`, `loudness`, `valence`, `loudness_per_energy`).
- A `SimpleImputer(median)` → `StandardScaler` sub-pipeline for `tempo` (which has missing values).
- `QuantileTransformer` on heavily skewed features (`acousticness`, `instrumentalness`, `speechiness`, `liveness`) to spread out their compressed distributions.
- Passthrough for the binary features (`explicit`, `is_long_track`).
- `OneHotEncoder` for the nominal features (`track_genre`, `decade`).

**Hyperparameter search.** Before tuning, I decided to search over the random forest's `max_depth` (tree complexity / overfitting control), `n_estimators` (number of trees), and `min_samples_split` (regularization of splits), since these most directly govern the bias–variance tradeoff. I ran a `GridSearchCV` (5-fold, scoring on F1) over:

- `max_depth`: [10, 20, None]
- `n_estimators`: [100, 200, 300]
- `min_samples_split`: [2, 5, 10]

**Best hyperparameters:** `max_depth = 10`, `n_estimators = 300`, `min_samples_split = 5` (cross-validated F1 ≈ 0.600).

**Performance (test set):**

| Metric | Baseline | Final |
| --- | --- | --- |
| Accuracy | 0.553 | **0.659** |
| F1-score | 0.521 | **0.607** |
| Precision | 0.460 | **0.569** |
| Recall | 0.599 | 0.650 |

The final model improves on every metric, with F1 rising from 0.52 to 0.61 and accuracy from 0.55 to 0.66. The gains come from (a) using many more informative audio features, (b) appropriate transformations for skewed and missing data, and (c) a non-linear model able to capture feature interactions the linear baseline could not.

<iframe
  src="assets/final-confusion.html"
  width="700"
  height="500"
  frameborder="0"
></iframe>

---

## Fairness Analysis

**Question:** Does the final model perform differently for explicit vs. non-explicit tracks?

- **Group X:** explicit tracks. **Group Y:** non-explicit tracks.
- **Evaluation metric:** **precision** (of the tracks the model labels "popular," how many truly are). Precision is the right lens here because a false "popular" prediction is the costly error for a curator acting on the model.
- **Null hypothesis (H₀):** The model is fair — its precision is the same for explicit and non-explicit tracks, and any difference is due to chance.
- **Alternative hypothesis (H₁):** The model is unfair — its precision differs systematically between the two groups.
- **Test statistic:** |precision(explicit) − precision(non-explicit)|.
- **Significance level:** 0.05.

On the test set, precision was **0.722 for explicit** tracks and **0.564 for non-explicit** tracks, an observed absolute difference of **0.158**. I ran a permutation test (1,000 shuffles of the group labels, reusing the already-fitted model without retraining), which produced a **p-value of ~0.11**.

<iframe
  src="assets/fairness-test.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

Because the p-value (~0.11) exceeds 0.05, I **fail to reject** the null hypothesis. While the model's precision looks higher on explicit tracks in this sample, the difference is not statistically significant — it is plausibly explained by chance, partly because the explicit group is small. I therefore find no significant evidence that the model is unfair with respect to explicit content.
