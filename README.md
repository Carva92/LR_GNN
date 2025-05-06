# GNN Pipeline for Clinical Patient Classification

This repository provides a full pipeline for transforming clinical datasets into graph structures and training a Graph Neural Network (GNN) for multi-class classification based on chronic condition profiles.

---

## 📁 Dataset Requirements

Your dataset must include the following columns per patient:

| Variable         | Description                                                  | Expected Format/Values          |
|------------------|--------------------------------------------------------------|----------------------------------|
| `RRID`           | Unique patient identifier                                    | Integer or string                |
| `STUDY`          | Study identifier (e.g., 'BASELINE')                          | String                           |
| `AGE`            | Patient age                                                  | Integer                          |
| `BMI`            | Body Mass Index level (1–5 encoded)                          | Integer                          |
| `HIGHBP`         | Hypertension status                                          | 1 = Yes, 2 = No                  |
| `HI_CHOL`        | Hyperlipidemia status                                        | 1 = Yes, 2 = No                  |
| `FADIAB`         | Father's diabetes status                                     | 1 = Yes, 2 = Borderline, 3/4 = No |
| `MODIAB`         | Mother's diabetes status                                     | Same encoding as FADIAB         |
| `SMOKENOW`       | Current smoking status                                       | 1 = Yes, 2 = No, 3 = Other       |
| `DIABETIC`       | Diabetes diagnosis                                           | 1 = Yes, 2 = No                  |
| `FABP`           | Family history of blood pressure                             | 1 = Yes, 2 = No, 3 = Other       |
| `MCORRSYS`       | Systolic blood pressure                                      | Integer (e.g., 120)              |
| `MCORRDIA`       | Diastolic blood pressure                                     | Integer (e.g., 80)               |
| `MMSE_SCORE`     | Mini-Mental State Examination score                          | Integer (0–30)                   |
| `GENDER`         | Biological sex                                               | 1 = Male, 2 = Female             |

> ⚠️ If your dataset uses different names or encodings, map them to match this schema before running the pipeline or modify the function according to the new dataset features.

---

## 🔄 Preprocessing Workflow

### 1. Format and Clean Your Dataset

```python
# If the dataset is in a different format from ".sas7bdat", change the function "fc.read_data" according to the correct format.
dataset = fc.read_data('dataset_name.sas7bdat')
```

- Cleans missing values and duplicates
- Encodes categorical features
- Derives binary conditions and age categories
- Creates 32-category multi-label for prediction

### 2. Compute Patient Similarity

```python
distances = e_distance(features, patients)
```

Computes Euclidean distances used as edge weights in the graph.

### 3. Define Patient-Feature Relationships

```python
rels = relationships(features, patients)
```

Links patients to their condition categories (e.g., BMI group, age category, smoking status).

### 4. Build Bipartite Graph

```python
G, patient_nodes, condition_nodes = bipartite(patients, rels)
PG, node_list = graph_projection(G, patient_nodes, distances, data['CATEGORY'].tolist())
```

- Projects a bipartite patient-feature graph into a unipartite patient-patient graph.
- Adds class labels and edge weights.

### 5. Simplify Graph (Optional)

```python
PG = simplify_graph(PG, simplify=True, distance=2.0)
```

Prunes weak connections based on distance threshold.

---

## 🧠 GNN Model Architecture

The model includes:

- Preprocessing dense layers
- Custom `GraphConvLayer` with message passing
- Post-processing layers
- Final dense classifier
- Optional Laplacian regularization via normalized graph Laplacian

---

## ⚙️ Model Training and Evaluation

### 1. Prepare the Inputs

```python
edges = np.array(list(PG.edges)).T
node_features = tf.cast(data.to_numpy()[:,2:9], dtype=tf.float64)
graph_info = (node_features, edges, edge_weights)
```

### 2. Define and Compile the Model

```python
base_model = GNNNodeClassifier(...)
gnn_model = MyModel(base_model)
gnn_model.compile(...)
```

### 3. Train the Model

```python
history = gnn_model.fit(
    x=[x_train_RRIDs, A_train, D_train],
    y=y_train,
    epochs=6000,
    batch_size=32,
    validation_split=0.15,
    callbacks=[early_stopping]
)
```

### 4. Evaluate and Predict

```python
_, test_acc = gnn_model.evaluate(...)
predictions = gnn_model.predict(...)
```

---

## 📊 Output

- Model accuracy
- Plots of predicted vs. real classes
- Trained model with node embeddings and edge weights

---

## 📦 Dependencies

Install required packages:

```bash
pip install tensorflow numpy pandas networkx scikit-learn matplotlib
```

---

## 🔁 How to Use with Your Dataset

1. Format your dataset to match the required schema.
2. Replace the input path in `read_data()` with your file.
3. Follow the same pipeline functions:
   - `categorize_data()`
   - `e_distance()`
   - `relationships()`
   - `bipartite()`
   - `graph_projection()`
4. Run the training script.

---

## 🧩 Notes

- You **must** preprocess your own dataset to match the variable names, formats, and encodings described above.
- Feel free to modify `categorize_data()` to support different encodings or additional preprocessing rules.
- This pipeline assumes all required variables are available and interpretable numerically or categorically.
- The model performs classification on patient nodes using clinical similarities and conditions.
- Laplacian regularization enforces spatial smoothness.
- Model can be extended to multi-modal or multi-task settings.

---

