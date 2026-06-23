# Graph-Based Job and Skill Recommendation

This project investigates how graph-based machine learning can improve job recommendations by modeling the relationships between developers, technical skills, and job roles. It compares a traditional latent-factor recommender with homogeneous and heterogeneous graph neural networks using the 2024 Stack Overflow Developer Survey.

The central question was whether representing users, skills, and roles as an interconnected graph produces better ranked recommendations than treating the data as a conventional user-item matrix.

## Project outcome

The heterogeneous graph neural network produced the strongest result in the reported experiments:

| Method | Test accuracy | Test MRR |
| --- | ---: | ---: |
| Matrix factorization | Not reported | 0.26 |
| Homogeneous GNN (GAT) | 0.32 | 0.49 |
| Heterogeneous GNN (GraphSAGE/HeteroConv) | **0.52** | **0.58** |

Mean Reciprocal Rank (MRR) measures how early the first relevant job appears in a recommendation list. The result supports the project's main hypothesis: preserving different entity and relationship types helps the model learn more useful user-skill-role representations.

## What was implemented

### 1. Data preparation

The notebooks use selected fields from the Stack Overflow survey and:

- combine language, database, platform, web-framework, and tooling fields into a unified skill profile;
- remove records without a usable developer-role label;
- split multi-valued skill fields into individual skills;
- encode categorical developer roles;
- create a 90/10 training and test split;
- construct either an interaction matrix or graph structure, depending on the model.

### 2. Matrix-factorization baseline

The baseline builds a job profile from skills commonly associated with each developer role. Jaccard similarity measures user-to-role skill overlap, producing an interaction matrix that is reduced with `TruncatedSVD`.

Recommendations are generated from the reconstructed latent preference scores. This provides a conventional collaborative-filtering benchmark for the graph models.

### 3. Homogeneous graph attention network

The GAT approach places users, skills, and developer roles in one graph and connects:

- users to their reported skills;
- users to their developer roles.

A two-layer `GATConv` model uses eight attention heads in the first layer. Attention lets the model weight more relevant neighboring nodes when updating each node representation.

### 4. Heterogeneous graph neural network

The final model preserves three node types:

- `user`
- `skill`
- `dev_type`

It also preserves typed relationships and their reverse directions:

- user `has_skill` skill;
- developer type `requires_skill` skill;
- user `has_type` developer type.

Two `HeteroConv` layers containing relation-specific `SAGEConv` operations perform message passing. The resulting user representations are used to predict developer roles and produce personalized recommendations.

### 5. Training and evaluation

The graph models were trained in PyTorch Geometric for up to 50 epochs using Adam, cross-entropy loss, weight decay, and early-stopping logic. Experiments were run on a Tesla T4 GPU in a hosted notebook environment.

The original implementation initially produced an unrealistically high MRR of 0.95. Investigation showed that test information was leaking into training and that the model was repeatedly predicting dominant classes. An explicit test mask was added so test nodes did not participate in training. The final reported HGNN MRR is 0.58.

## Repository contents

| File | Purpose |
| --- | --- |
| `Kshitij_Final_Report.pdf` | Full academic report, methodology, results, challenges, and references |
| `user_basedCF.ipynb` | Jaccard skill matching and SVD-based recommendation baseline |
| `aml-gat-hgnn.ipynb` | Development version of the homogeneous and heterogeneous graph models |
| `aml-gat-hgnn1.ipynb` | Final experiment variant, including model evaluation plots used in the report |

## Data

The project uses the **2024 Stack Overflow Developer Survey**, containing responses from more than 65,000 developers.

The survey CSV is not included in this repository. The graph notebooks currently expect:

```text
/kaggle/input/survey-results/survey_results_public.csv
```

Download the public survey data and either place it at that path in Kaggle or update the `file_path` argument in the notebooks.

## Running the project

Use Python 3 with:

```text
pandas
numpy
scikit-learn
torch
torch-geometric
tqdm
matplotlib
```

Recommended execution order:

1. Run `user_basedCF.ipynb` to reproduce the matrix-factorization baseline.
2. Run `aml-gat-hgnn1.ipynb` to build and evaluate the GAT and HGNN models.
3. Use `aml-gat-hgnn.ipynb` when reviewing earlier implementation choices or experiments.

A CUDA-capable environment is strongly recommended for graph training.

## Interpretation

The comparison shows why heterogeneous graphs suit this problem. A developer, a skill, and a role are not interchangeable entities, and “has skill” conveys a different meaning from “has type” or “requires skill.” Relation-specific message passing preserves those distinctions, allowing the HGNN to outperform both the latent-factor baseline and the single-type graph.

The work also demonstrates an important modeling lesson: high recommendation scores must be checked for leakage and class-collapse behavior before they are treated as genuine performance.

## Limitations and future work

- The model uses survey responses rather than real application, click, or hiring-outcome data.
- Developer roles are used as recommendation targets, so this is a role recommender rather than a production job-posting matcher.
- Survey fields contain missing values, self-reporting bias, and class imbalance.
- The notebooks depend on an external dataset path and do not package trained weights.
- Future work proposed in the report includes edge-level attention, additional contextual attributes such as industry and company size, richer meta-paths, and real-time job updates.

## Author

Kshitij Desai, University of Adelaide, November 2024.
