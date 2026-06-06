# Hit or Miss

**Author:** Ella Wen

---

## Table of Contents

- [Introduction](#introduction)
- [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)
  - [Data Cleaning](#data-cleaning)
  - [Univariate Analysis](#univariate-analysis)
  - [Bivariate Analysis](#bivariate-analysis)
  - [Interesting Aggregates](#interesting-aggregates)
- [Assessment of Missingness](#assessment-of-missingness)
  - [NMAR Analysis](#nmar-analysis)
  - [Missingness Dependency](#missingness-dependency)
- [Hypothesis Testing](#hypothesis-testing)
- [Framing a Prediction Problem](#framing-a-prediction-problem)
- [Baseline Model](#baseline-model)
- [Final Model](#final-model)
- [Fairness Analysis](#fairness-analysis)

---

## Introduction

Streaming platforms like Spotify describe every track with a set of machine-generated **audio features** — numbers capturing how danceable, energetic, acoustic, or upbeat a song is — alongside metadata such as its genre, release date, and whether it contains explicit lyrics. At the same time, Spotify assigns each track a **popularity** score from 0 to 100, based largely on recent play counts. This project asks a question that matters to artists, producers, and the platforms that curate playlists:

> **What audio and metadata features are associated with a track's popularity?**

If certain measurable qualities of a song are consistently linked to popularity, that information is valuable for anyone deciding what to produce, promote, or surface to listeners.

The dataset is **Spotify Music Tracks**, containing **114,000 rows** — 1,000 tracks for each of **114 genres**. Each row is a single track. Aside from the pure identifier columns (`track_id`, `artists`, `album_name`, `track_name`), essentially every column feeds into this analysis:

| Column | Description |
| --- | --- |
| `popularity` | Spotify popularity score (0–100), based on total plays and recency (our response variable) |
| `track_genre` | Genre label assigned by Spotify (114 genres, 1,000 tracks each) |
| `danceability` | How suitable a track is for dancing (0–1) |
| `energy` | Perceptual measure of intensity and activity (0–1) |
| `key` | Estimated musical key as an integer pitch class (0 = C, 1 = C♯/D♭, …, 11 = B; -1 if undetected) |
| `loudness` | Overall loudness in decibels (dB) |
| `mode` | Modality of the track: major (1) or minor (0) |
| `speechiness` | Presence of spoken words (0–1) |
| `acousticness` | Confidence the track is acoustic (0–1) |
| `instrumentalness` | Predicts whether a track contains no vocals (0–1) |
| `liveness` | Probability the track was performed live (0–1) |
| `valence` | Musical positiveness conveyed by the track (0–1) |
| `tempo` | Estimated tempo in beats per minute (BPM) |
| `time_signature` | Estimated time signature (beats per bar, typically 3–7) |
| `explicit` | Whether the track has explicit lyrics (boolean) |
| `duration_ms` | Track length in milliseconds |
| `release_date` | Release date of the track (parsed into `release_year` and `decade`) |

The dataset ships with a second file, `artists.csv`, but I deliberately **did not join it** into the analysis. Its fields (artist names, follower counts, and artist-level genre tags) are either redundant with information already in the tracks table or, in the case of follower counts, a form of leakage: an artist's follower count is itself a popularity signal that accumulates *over time* and would not be cleanly available "at the time of prediction" for a brand-new release. Joining it would also introduce many-to-many matching headaches (tracks list multiple semicolon-separated artists). Since my question is specifically about whether a track's **own audio and metadata** predict its popularity, the per-track features are the right and self-sufficient unit of analysis, so I kept the scope to `music_tracks.csv`.

---

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning

I performed the following cleaning steps, each motivated by the data generating process:

1. **Dropped the `Unnamed: 0` column**, which was just a stored row index carrying no information.
2. **Parsed `release_date` into a `release_year`** (and later a `decade`) column. Release dates appear in mixed formats (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`), so I parsed them with a mixed-format datetime parser and extracted the year.
3. **Replaced invalid sentinel values with `NaN`.** A handful of tracks had a `duration_ms` of 0, a `tempo` of 0, or a `time_signature` of 0. These are physically impossible for a real recording and almost certainly indicate a failed measurement rather than a true value, so I converted them to `NaN` so they would be treated as missing rather than dragging down averages. (89 tracks had tempo = 0; these were folded in with the existing missing tempos.)
4. **Created `duration_min`** from `duration_ms` (dividing by 60,000) for easier interpretation.
5. **Confirmed `explicit` is already boolean**, so no conversion was needed.

The cleaning steps dropped no analytical columns — only the redundant index was removed — so the cleaned DataFrame keeps **all** of the original columns plus the two derived ones (`release_year`, `duration_min`), giving 114,000 rows and 23 columns. Its head (scroll right to see every column) looks like:

| track_id               | artists                | album_name                                             | track_name                 |   popularity |   duration_ms | release_date   | explicit   |   danceability |   energy |   key |   loudness |   mode |   speechiness |   acousticness |   instrumentalness |   liveness |   valence |   tempo |   time_signature | track_genre   |   release_year |   duration_min |
|:-----------------------|:-----------------------|:-------------------------------------------------------|:---------------------------|-------------:|--------------:|:---------------|:-----------|---------------:|---------:|------:|-----------:|-------:|--------------:|---------------:|-------------------:|-----------:|----------:|--------:|-----------------:|:--------------|---------------:|---------------:|
| 5SuOikwiRyPMVoIQDJUgSV | Gen Hoshino            | Comedy                                                 | Comedy                     |           73 |        230666 | 1974           | False      |          0.676 |    0.461 |     1 |     -6.746 |      0 |         0.143 |          0.032 |                  0 |      0.358 |     0.715 |  87.917 |                4 | acoustic      |           1974 |          3.844 |
| 4qPNDBW1i3p13qLCt0Ki3A | Ben Woodward           | Ghost (Acoustic)                                       | Ghost - Acoustic           |           55 |        149610 | 1995-04        | False      |          0.42  |    0.166 |     1 |    -17.235 |      1 |         0.076 |          0.924 |                  0 |      0.101 |     0.267 |  77.489 |                4 | acoustic      |           1995 |          2.494 |
| 1iJBSr7s7jYXzM8EGcbK5b | Ingrid Michaelson;ZAYN | To Begin Again                                         | To Begin Again             |           57 |        210826 | 1973           | False      |          0.438 |    0.359 |     0 |     -9.734 |      1 |         0.056 |          0.21  |                  0 |      0.117 |     0.12  |  76.332 |                4 | acoustic      |           1973 |          3.514 |
| 6lfxq3CG4xtTiEg7opyCyx | Kina Grannis           | Crazy Rich Asians (Original Motion Picture Soundtrack) | Can't Help Falling In Love |           71 |        201933 | 2018-08-10     | False      |          0.266 |    0.06  |     0 |    -18.515 |      1 |         0.036 |          0.905 |                  0 |      0.132 |     0.143 | 181.74  |                3 | acoustic      |           2018 |          3.366 |
| 5vjLSffimiIP26QG5WcN2K | Chord Overstreet       | Hold On                                                | Hold On                    |           82 |        198853 | 2017-02-03     | False      |          0.618 |    0.443 |     2 |     -9.681 |      1 |         0.053 |          0.469 |                  0 |      0.083 |     0.167 |  nan    |                4 | acoustic      |           2017 |          3.314 |

### Univariate Analysis

This histogram shows the distribution of the `popularity` score across all 114,000 tracks, where each bar counts how many tracks fall into a given popularity range from 0 to 100. The distribution is right-skewed with a pronounced spike at 0 — a large block of tracks that have accumulated essentially no recent plays — while the bulk of the remaining tracks form a broad mound between roughly 20 and 50. Very few tracks reach the high-popularity tail above 70, confirming that genuine hits are rare relative to the catalog as a whole. This shape directly informs the rest of the project: because popularity is continuous and so unevenly distributed, the modeling steps later binarize it into a "popular vs. not" label rather than trying to predict the exact score.

<iframe
  src="assets/univariate-popularity.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

This histogram shows how the catalog is distributed over time, with each bar counting the number of tracks released in a given range of years. The distribution is heavily right-skewed toward the present: the overwhelming majority of tracks were released in the 2000s and 2010s, while a long, thin tail stretches all the way back to the 1920s. This tells us the dataset is dominated by relatively modern music, which is important context for the rest of the analysis — release recency is itself tied to how Spotify computes popularity, a relationship we examine directly in the bivariate analysis below.

<iframe
  src="assets/univariate-year.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

### Bivariate Analysis

This line chart plots the **mean `popularity` of all tracks released in each year**, letting us trace how average popularity has shifted over time. Each point is the average popularity of every track from that release year, and the connecting line makes the temporal trend easy to follow. There is a clear upward trend: tracks from recent years score substantially higher on average than older ones, with the curve rising steeply after roughly the 1990s. The earliest years are noticeably noisier because they contain far fewer tracks (as the release-year histogram above showed), so a handful of songs can swing the average. This pattern is consistent with the way Spotify's popularity metric weights *recent* listening activity, which structurally disadvantages older catalog music regardless of its musical qualities.

<iframe
  src="assets/bivariate-year.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

This box plot compares the **`popularity` distribution of explicit versus non-explicit tracks**. Each box spans the interquartile range (the middle 50% of tracks) with the inner line marking the median, so we can compare both the center and the spread of the two groups side by side. Explicit tracks sit noticeably higher — a larger median and an upward-shifted box — indicating they tend to be more popular than non-explicit tracks across the dataset. The difference is visible but not dramatic, and the two distributions overlap heavily, which is exactly why we do not stop at this visual impression: we test the difference formally with a permutation test in the Hypothesis Testing section to determine whether it is more than just noise.

<iframe
  src="assets/bivariate-explicit.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

### Interesting Aggregates

For this aggregate, we group the dataset by `track_genre` and compute the **mean popularity, median popularity, and track count** for each of the 114 genres, then sort to surface the most popular ones. Because every genre contains exactly 1,000 tracks, the `track_count` column confirms the grouping is perfectly balanced, so the comparison across genres is fair and not distorted by sample size. The resulting table makes it clear that popularity is strongly genre-dependent: production- and chart-oriented genres such as `pop-film`, `k-pop`, and `chill` top the list with mean popularities near 54–60, while many niche or instrumental genres sit far lower. This wide spread is the main motivation for a later modeling decision — we restrict the prediction task to a handful of genres that each have real internal variation in popularity, so the problem isn't trivially solved by genre alone. The ten most popular genres by mean popularity:

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

I also built a pivot table that aggregates the data by **`decade` (derived from release year) and `explicit` flag**, computing the mean popularity for every decade-by-explicit combination. Laying the two groups side by side in a single table lets us see how the explicit/non-explicit popularity gap has evolved over time, rather than collapsing it into a single overall average. Within almost every decade explicit tracks are more popular than non-explicit ones, and — crucially — the gap widens sharply in the 2010s and 2020s, where explicit tracks average roughly 5–6 points higher. The empty cells in the earliest decades reflect the fact that explicit tracks were essentially nonexistent (or simply weren't labeled as such) before the 1960s. Together this suggests the link between explicit content and popularity is largely a modern phenomenon, which adds useful context to the hypothesis test that follows.

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

The only column with substantial missingness is **`tempo`**, which is missing for about **22,114 tracks (~19%)**. I believe `tempo` is plausibly **NMAR** (Not Missing At Random), because whether a tempo can be measured at all is tied to the rhythmic nature of the track itself — which is precisely the quantity that goes missing. Spotify derives tempo from an automated audio analysis that locks onto a steady, repeating beat. For tracks that have no clear, consistent pulse — ambient soundscapes, free-tempo classical or rubato pieces, spoken-word recordings — the algorithm has nothing stable to latch onto, so no tempo value is recorded. In other words, the missingness is driven by the (unobserved) tempo-related character of the song, not by random chance spread evenly across the catalog.

There are several plausible mechanisms that could produce this pattern:

- **Beatless or free-tempo audio** – ambient, drone, classical-rubato, or spoken-word tracks lack a steady pulse, so the analyzer simply cannot estimate a BPM.
- **Low-energy, sparse recordings** – soft or thinly-produced tracks give the beat-detection algorithm a weak signal to work with, making a confident tempo estimate less likely.
- **Pipeline or coverage gaps** – older or less-processed catalog entries may never have been run through the full audio-analysis pipeline that produces tempo.

To confirm whether `tempo` is truly NMAR, we would need information we do not currently have — for example, the analyzer's per-track *confidence score* for its tempo estimate, or the raw audio itself, which would let us verify that the missing tempos really do correspond to beatless or rhythmically ambiguous tracks. Alternatively, if observed columns such as `energy`, `loudness`, or `genre` turn out to fully explain the missingness, we would instead classify it as **MAR** (Missing At Random). We investigate that possibility with permutation tests in the next section.

### Missingness Dependency

To move beyond the NMAR reasoning above and test the missingness empirically, I ran permutation tests on the `tempo_missing` indicator against other columns. In every test the **test statistic is the absolute difference in the column's group means** (tracks where tempo is missing vs. tracks where it is present), and the **significance level is 0.05**. The procedure is the same each time: compute the observed difference between the two groups, then shuffle the `tempo_missing` indicator 1,000 times (keeping the column's values fixed) to build a null distribution of differences under the assumption of independence, and finally compute the p-value as the proportion of shuffled differences at least as extreme as the observed one. I report one column the missingness **does** depend on (`energy`) and one it **does not** (`popularity`).

#### Tempo and Energy — Missingness *Depends* on Energy (MAR)

**Hypotheses.**
- **Null (H₀):** The missingness of `tempo` is independent of `energy` — the mean energy is the same for tracks with and without a recorded tempo.
- **Alternative (H₁):** The mean energy differs between tracks with missing vs. present tempo, indicating the missingness is associated with energy.

**Results.** Tracks with a missing tempo have a substantially lower mean energy (**0.512**) than tracks with an observed tempo (**0.673**) — an observed absolute difference of **0.1611**. Across 1,000 permutations, *no* shuffled difference came close to this value, giving a **p-value of 0.0000**. Since this is far below 0.05, I **reject the null hypothesis**: tempo missingness is significantly associated with energy. This is exactly the MAR signature — the chance a tempo is missing depends on another *observed* column. It also lines up with the NMAR intuition above: low-energy, sparse tracks (ambient, acoustic, beatless recordings) are precisely the ones a beat-detector struggles to read.

<iframe
  src="assets/missingness-energy.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

The red line marks the observed statistic, which sits far to the right of the entire null distribution — a visual confirmation that a difference this large would essentially never arise by chance if missingness and energy were unrelated.

#### Tempo and Popularity — Missingness *Does Not* Depend on Popularity

**Hypotheses.**
- **Null (H₀):** The missingness of `tempo` is independent of `popularity` — the mean popularity is the same for tracks with and without a recorded tempo.
- **Alternative (H₁):** The mean popularity differs between tracks with missing vs. present tempo.

**Results.** Here the two groups are almost indistinguishable: mean popularity is **33.31** when tempo is missing and **33.22** when it is present, an observed absolute difference of only **0.0863**. This difference falls right in the middle of the null distribution, yielding a **p-value of ≈0.60** — nowhere near significant. I therefore **fail to reject the null hypothesis**: there is no evidence that tempo missingness depends on popularity. This is reassuring for the rest of the project, because it means the ~19% of tracks with missing tempo are *not* a popularity-biased subset, so dropping or imputing them is unlikely to distort the popularity analysis.

<iframe
  src="assets/missingness-popularity.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

Unlike the energy plot, here the observed statistic (red line) sits squarely inside the bulk of the null distribution, visually confirming that a difference this small is entirely consistent with chance.

**Overall, the evidence points to `tempo` being MAR rather than NMAR** in practice: its missingness is explained by other *observed* audio characteristics (energy, and similarly loudness), while showing no dependence on popularity. We cannot fully rule out an NMAR component without the analyzer's confidence scores, but for the purposes of this analysis treating it as MAR is well supported.

---

## Hypothesis Testing

The exploratory box plot suggested explicit tracks tend to be more popular, but the distributions overlapped heavily, so I tested the difference formally. The question: **are explicit tracks genuinely more popular on average, or could the gap we see just be random noise?**

### Hypotheses

- **Null hypothesis (H₀):** Explicit and non-explicit tracks have the *same* mean popularity. Any observed difference is due to random chance.
- **Alternative hypothesis (H₁):** Explicit tracks have a *higher* mean popularity than non-explicit tracks.

### Test Statistic and Significance Level

Because the alternative is **directional** (explicit *higher*, not merely *different*), I use a one-sided test statistic: the **signed difference in mean popularity**, computed as (mean popularity of explicit tracks) − (mean popularity of non-explicit tracks). A large positive value is evidence for H₁. I use a **significance level of 0.05**. A permutation test is appropriate here because we are asking whether the `explicit` label and the `popularity` values are associated; under H₀ the label is exchangeable, so shuffling it simulates the null world directly without assuming any particular distribution for popularity.

### Explanation of Results

The observed difference in means is **+3.52 popularity points** in favor of explicit tracks. I then shuffled the `explicit` labels across all tracks 1,000 times (holding the popularity values fixed) to build the null distribution of the statistic, and computed the p-value as the share of permuted differences that were at least as large as the observed one. **Not a single one of the 1,000 permutations** reached +3.52, giving a **p-value of 0.0000**.

<iframe
  src="assets/hypothesis-test.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

The red line (observed +3.52) sits far to the right of the entire null distribution, which is centered on 0. Because the p-value is far below 0.05, I **reject the null hypothesis** in favor of the alternative: the data are consistent with explicit tracks being more popular, on average, than non-explicit tracks. Two caveats keep this honest. First, this is a probabilistic, observational result — it says the gap is very unlikely under pure chance, not that explicitness *causes* popularity. As the decade pivot table showed, explicit tracks skew toward recent, heavily-streamed eras and genres, which plausibly drives much of the association. Second, statistical significance here is partly a function of the very large sample (≈114,000 tracks), so even a modest 3.5-point gap is detected with extreme confidence; the *effect size* is real but moderate.

---

## Framing a Prediction Problem

### Prediction Problem: Classifying Popular Tracks

I frame a **binary classification** problem: predict whether a track will be **popular** using only information knowable about the track itself around release time. I define the response variable as:

- `is_popular = 1` if `popularity >= 50`
- `is_popular = 0` otherwise

### Why This Response Variable?

I chose to predict `is_popular` (rather than regress on the raw 0–100 score) because "popular vs. not" is the concrete, actionable decision a playlist curator or A&R team actually makes — they decide whether to invest in or feature a track, not to estimate its exact score to the integer. Binarizing also tames the awkward shape of the raw popularity distribution (the big spike at 0 plus a long tail). The **threshold of 50** is deliberate: on the modeling subset it produces a workable, roughly balanced split (~40% popular), so the classifier has enough examples of both classes to learn from. A threshold of 70 would leave too few positive examples to model reliably, while splitting at the median would be arbitrary and carry no real-world meaning.

### Modeling Subset

Following this dataset's guidance, I restrict modeling to **six musically distinct genres** — `hard-rock`, `acoustic`, `edm`, `disco`, `folk`, and `indian`. These were chosen because (a) together they span the full range of energy, acousticness, danceability, and instrumentalness, so the audio features are genuinely informative rather than redundant, and (b) each genre has real internal popularity variation (roughly 30–50% popular at the threshold), so the task is non-trivial and not simply solved by reading off the genre label. The filtered subset has **6,000 tracks** (1,000 per genre).

### Features and Information Known at Time of Prediction

Every feature I use is knowable about a track at or before its release — none of it depends on how the track *later* performs. This avoids data leakage: I am predicting popularity, so I cannot use popularity (or anything derived from accumulated play counts, like follower-based signals) as an input. The features fall into a few groups:

- **Core audio features** — `danceability`, `energy`, `loudness`, `valence`, `tempo`. These describe the intrinsic sonic character of the recording and are computed directly from the audio, so they are available the moment the track exists.
- **Skewed audio features** — `acousticness`, `instrumentalness`, `speechiness`, `liveness`. Also audio-derived, but heavily skewed, so they get special transformation in the final model.
- **Metadata** — `track_genre` (nominal), `explicit` (boolean), and the release `decade` (derived from `release_date`). All three are known at release time: a track's genre, explicit tag, and release date are fixed properties, not future outcomes.

### Evaluation Metric

I evaluate with the **F1-score**. The classes are imbalanced (~40% popular), which makes plain **accuracy** misleading — a lazy model could score reasonably just by leaning toward the majority "not popular" class while failing to identify the popular tracks I actually care about. F1 is the harmonic mean of precision and recall, so it rewards a model only when it identifies popular tracks (recall) *without* drowning in false alarms (precision). That balance fits the use case — surfacing likely hits while keeping the shortlist trustworthy — better than accuracy, precision, or recall alone, which is why I use it as the primary metric (while still reporting the others for a fuller picture).

---

## Baseline Model

The baseline is a deliberately simple **logistic regression**, built so that the more elaborate final model has a meaningful reference point to beat. Everything — preprocessing and the classifier — lives inside a single scikit-learn `Pipeline`, and the model is evaluated on a held-out 20% test set using a stratified 80/20 split (the same split the final model reuses, so the two are directly comparable).

### Feature Selection and Encoding

The baseline uses **three original columns**: two **quantitative** features and one **nominal** feature (no ordinal features).

- `danceability` — quantitative
- `energy` — quantitative
- `track_genre` — nominal

Encodings: the two quantitative features are standardized with `StandardScaler`, because logistic regression is sensitive to feature scale. The nominal `track_genre` is one-hot encoded with `drop='first'` to avoid the dummy-variable trap (perfect multicollinearity among the dummy columns).

### Pipeline Implementation

```python
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), ['track_genre']),
    ('num', StandardScaler(),            ['danceability', 'energy']),
])
baseline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000)),
])
```

`class_weight='balanced'` lets the model account for the ~40/60 class split so it doesn't simply default to predicting the majority class.

### Model Performance

On the held-out test set:

- **Accuracy:** 0.553
- **F1-score:** 0.521
- **Precision:** 0.460
- **Recall:** 0.599

<iframe
  src="assets/baseline-confusion.html"
  width="700"
  height="500"
  frameborder="0"
></iframe>

### Confusion Matrix Insight

Reading the confusion matrix on the 1,200 test tracks: the model catches a fair share of truly popular tracks (291 true positives, recall ≈ 0.60), but it pays for that with a large number of **false positives** (341) — non-popular tracks it wrongly flags as popular. That is exactly why precision is low (0.460): fewer than half of its "popular" predictions are correct. In other words, the baseline is trigger-happy, casting a wide net that pulls in many misses along with the hits.

### Interpretation and Next Steps

I consider this baseline **not good, but useful**. An F1 of 0.52 and accuracy of 0.55 are only modestly better than guessing — unsurprising, since two audio dimensions plus a genre label simply cannot capture much of what drives popularity. Its value is as a floor. The next section improves on it by (1) bringing in many more audio and metadata features, (2) transforming skewed and missing features appropriately, and (3) switching to a non-linear model that can capture feature interactions a single linear boundary cannot.

---

## Final Model

### Motivation for Model Change

The baseline's weakness was twofold: too few features, and a linear decision boundary that can't represent interactions (e.g. "high energy is good *for edm* but not *for acoustic*"). The final model addresses both. I switch from logistic regression to a **random forest classifier**, which is non-linear and naturally captures feature interactions and thresholds, and I feed it a much richer, properly-transformed feature set. To keep the comparison honest, the final model is trained and evaluated on the **exact same stratified 80/20 split** as the baseline.

### Feature Engineering and Added Features

Beyond simply adding more of the existing audio columns, I engineered **two new features**, each grounded in how the data is generated:

- **`loudness_per_energy`** = `loudness / (energy + 0.01)` — captures how *efficiently* a track converts its energy into loudness. Two tracks can share the same energy yet be mastered very differently; this ratio reflects production/compression choices associated with a polished, "radio-ready" modern sound, which plausibly relates to popularity. (The `+ 0.01` guards against division by zero when energy is 0.)
- **`is_long_track`** = `1` if `duration_min > 4` — a binary flag for unusually long tracks. Long tracks tend to be less playlist- and radio-friendly, a relationship that is non-linear (it kicks in past a duration threshold) and therefore well-suited to a tree.

### Model Pipeline and Hyperparameter Tuning

All preprocessing is bundled into one `ColumnTransformer` inside the `Pipeline`, with each feature group getting the transformation its distribution calls for:

- **`StandardScaler`** on the roughly symmetric audio features (`danceability`, `energy`, `loudness`, `valence`, `loudness_per_energy`).
- A **`SimpleImputer(median)` → `StandardScaler`** sub-pipeline for `tempo`, which has missing values (imputing before scaling so the MAR gaps don't break the model).
- **`QuantileTransformer`** on the heavily right-skewed features (`acousticness`, `instrumentalness`, `speechiness`, `liveness`) to spread their compressed distributions into something the trees split on more cleanly.
- **Passthrough** for the binary features (`explicit`, `is_long_track`).
- **`OneHotEncoder`** for the nominal features (`track_genre`, `decade`).

Before tuning, I decided which hyperparameters to search and why: `max_depth` (controls tree complexity / overfitting), `n_estimators` (number of trees in the forest), and `min_samples_split` (regularizes how eagerly nodes split) — the three that most directly govern the bias–variance tradeoff for a random forest. I searched them with a 5-fold **`GridSearchCV`** scoring on F1:

- `max_depth`: [10, 20, None]
- `n_estimators`: [100, 200, 300]
- `min_samples_split`: [2, 5, 10]

The best combination was **`max_depth = 10`, `n_estimators = 300`, `min_samples_split = 5`** (cross-validated F1 ≈ 0.600). Notably, the search preferred the *shallowest* depth offered (10), which makes sense — it regularizes the forest and prevents the deep trees from overfitting the training set.

### Performance

The final model improves on **every** metric relative to the baseline, on the same test set:

| Metric | Baseline | Final |
| --- | --- | --- |
| Accuracy | 0.553 | **0.659** |
| F1-score | 0.521 | **0.607** |
| Precision | 0.460 | **0.569** |
| Recall | 0.599 | **0.650** |

<iframe
  src="assets/final-confusion.html"
  width="700"
  height="500"
  frameborder="0"
></iframe>

### Conclusion and Next Steps

F1 rose from 0.52 to 0.61 and accuracy from 0.55 to 0.66 — a clear, across-the-board improvement. The confusion matrix shows where the gains come from: false positives drop sharply (from 341 down to 239) while true positives actually rise (291 → 316), so precision climbs from 0.46 to 0.57 *without* sacrificing recall. The richer feature set, distribution-aware transformations, and non-linear model together let it separate popular from non-popular tracks far more reliably than the linear baseline. That said, an F1 of ~0.61 leaves clear room to grow: popularity is driven heavily by factors *outside* the audio itself (marketing, playlist placement, artist fame, virality), which no amount of audio-feature engineering can recover. Natural next steps would be gradient-boosted trees (often a notch above random forests on tabular data), calibrated probability thresholds tuned to the precision/recall balance a curator wants, or — if leakage could be controlled — carefully-timed external signals.

---

## Fairness Analysis

Having a model that performs well *on average* isn't enough — I also want to know whether it works **equally well across groups**, or whether it systematically does better for one kind of track than another. I split the tracks into two groups by their `explicit` flag and ask whether the model's quality differs between them.

- **Group X:** explicit tracks.
- **Group Y:** non-explicit tracks.

### Evaluation Metric

I measure fairness using **precision** — of the tracks the model labels "popular," what fraction truly are. Precision is the right lens for this use case: the cost of unfairness here is a curator *acting* on a wrong "popular" prediction, so I care whether the model's "popular" calls are equally trustworthy for explicit and non-explicit tracks. The test statistic is the **absolute difference in precision** between the two groups — that is, the absolute value of precision(explicit) − precision(non-explicit).

### Hypotheses

- **Null hypothesis (H₀):** The model is **fair** — its precision is the same for explicit and non-explicit tracks, and any observed difference is due to random chance.
- **Alternative hypothesis (H₁):** The model is **unfair** — its precision differs systematically between the two groups.

### Permutation Test Methodology

I evaluate the **already-fitted** final model on the test set (no retraining), record each track's true label, predicted label, and `explicit` status, and compute the observed precision gap between the two groups. To build the null distribution, I shuffle the `explicit` labels across the test tracks 1,000 times — breaking any real association between group membership and the model's correctness — and recompute the precision gap each time. The p-value is the share of shuffled gaps at least as large as the observed one. The **significance level is 0.05**.

### Observed Results

On the test set the model's precision was **0.722 for explicit** tracks versus **0.564 for non-explicit** tracks — an observed absolute difference of **0.158**. That looks sizable at first glance, but the permutation test returned a **p-value of ≈0.11**.

<iframe
  src="assets/fairness-test.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

As the plot shows, the observed gap (red line) falls within the upper-middle of the null distribution rather than out in its tail. Because the **p-value (≈0.11) exceeds 0.05, I fail to reject the null hypothesis.** Although the model's precision looks higher on explicit tracks in this particular sample, the difference is not statistically significant — a gap this large arises fairly often just from chance, largely because explicit tracks are a small minority of the test set, so their precision estimate is noisy. **I therefore find no significant evidence that the model is unfair with respect to explicit content.** This is a non-finding rather than a guarantee of fairness: with more explicit tracks to estimate precision more precisely, a real gap could still surface, so it would be worth revisiting on a larger or more balanced sample.
