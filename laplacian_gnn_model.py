import os
####*IMPORANT*: Have to do this line *before* importing tensorflow
os.environ['PYTHONHASHSEED']=str(1)

import os
import sys
import scipy
import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from networkx.algorithms.assortativity import neighbor_degree

from keras import backend
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import random

import model_functions as fc

def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)

reset_random_seeds()

# If the dataset is in a different format from ".sas7bdat", change the function "fc.read_data" according to the correct format.
dataset = fc.read_data('dataset_name.sas7bdat')

patients, conditions_features, data = fc.categorize_data(dataset[0])

data['CATEGORY']

df = fc.e_distance(conditions_features, patients)

relationships = fc.relationships(conditions_features, patients)

G, patients_nodes, condition_node = fc.bipartite(patients, relationships)

G.number_of_nodes()

# Plot the Bipartite Graph (optional)
# color_map = []
# for node in G.nodes:
#     if type(node) == int:
#         color_map.append('blue')
#     else:
#         color_map.append('red')

# X, Y = bipartite.sets(G)
# pos = dict()
# plt.figure(2,figsize=(30,30))
# pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
# pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
# nx.draw(G, with_labels = True, pos=pos, node_color=color_map)
# plt.show()

# Plot Projected Graph (optional)
# from itertools import count
# groups = set(nx.get_node_attributes(PG,'y').values())
# nodes = PG.nodes()
# mapping = dict(zip(sorted(groups),count()))
# colors = [mapping[PG.nodes[n]['y']] for n in nodes]
# plt.figure(4,figsize=(30,30))
# #pos = nx.spring_layout(PG, k=10)
# nx.draw(PG, with_labels = True, node_color = colors, font_weight = 'normal')
# # labels = {e: PG.edges[e]['weight'] for e in PG.edges}
# # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()
# print('We have %d nodes.' % PG.number_of_nodes())
# print('We have %d edges.' % PG.number_of_edges())

category = (data["CATEGORY"]).to_numpy()
category = category.tolist()
PG, node_list = fc.graph_projection(G, patients_nodes, df, category)

distance = 5
PG, edges_list = fc.simplify_graph(PG, True, distance)

# Check the order of the graph (should be the number of patients)
PG.order()

# Eigen Values Normalized Laplacian Matrix for Regularization
A = nx.to_numpy_array(PG)
I = np.identity(A.shape[0])
D = np.diag(A.sum(axis=1))
D_inv_sqrt = np.linalg.inv(np.sqrt(D))
L_n = I - np.dot(D_inv_sqrt, A).dot(D_inv_sqrt)
# fig = plt.figure(figsize=(5, 5)) # in inches
# plt.imshow(A,cmap="Greys",
#                   interpolation="none")

eigen_values, eigen_vectors = np.linalg.eig(L_n)
eigen_values = np.real(eigen_values)
eigen_vectors = np.real(eigen_vectors)
idx = eigen_values.argsort()
eigen_values = eigen_values[idx]
eigen_vectors = eigen_vectors[:,idx]

print("Largest eigenvalue:", max(eigen_values))
print("Smallest eigenvalue:", min(eigen_values))

patients_relationships = pd.DataFrame(edges_list, columns =['PatientX', 'PatientY'])

# Check if the graph is connected
nx.is_connected(PG)

# Classes Representations (Classes are the patients categories)
class_values = sorted(data["CATEGORY"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
patient_idx = {name: idx for idx, name in enumerate(sorted(data["RRID"].unique()))}

data["RRID"] = data["RRID"].apply(lambda name: patient_idx[name])
patients_relationships["PatientX"] = patients_relationships["PatientX"].apply(lambda name: patient_idx[name])
patients_relationships["PatientY"] = patients_relationships["PatientY"].apply(lambda name: patient_idx[name])
data["CATEGORY"] = data["CATEGORY"].apply(lambda value: class_idx[value])

hidden_units = [64,64]
dropout_rate = 0.02
num_epochs = 6000
batch_size = 32

starter_learning_rate = 0.00005
end_learning_rate = 0.00001
decay_steps = 9000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=0.5)

learning_rate = learning_rate_fn

# Display Learning Curves after Training
def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "val"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    #ax1.set_yscale('log')

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "val"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()

# Data Division Train - Test
#Divide data into Train and Test (around 80% - 20%)

train_data, test_data = [], []

for _, group_data in data.groupby("CATEGORY"):
    random_selection = np.random.rand(len(group_data.index)) <= 0.80
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

data

# Feed Forward Network
def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        #fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))
        #print(fnn_layers)
    return keras.Sequential(fnn_layers, name=name)

# Data Division
feature_names = set(data.columns) - {"RRID", "DIABETIC", "OBESE", "CATEGORY"}
num_features = len(feature_names)
num_classes = len(class_idx)

# Create train and test features as a numpy array.
x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()
# Create train and test targets as a numpy array.
y_train = train_data["CATEGORY"]
y_test = test_data["CATEGORY"]

# New Dataset after preprocessing
# Create train and test features as a numpy array.
edges_arr = list(PG.edges)
pandas = pd.DataFrame(edges_arr)
pandas.rename(columns = {0:'PatientX', 1:'PatientY'}, inplace = True)
df1 = pandas
#@title df2
df
df2 = df.sort_values(['PatientX', 'PatientY'])
df2 = df2.reset_index()
del df2['index']
intersection = df2[['PatientX', 'PatientY']].merge(df1[['PatientX', 'PatientY']]).drop_duplicates()
df3 = pd.concat([df1.merge(intersection), df2.merge(intersection)])
df3 = df3.dropna()
df3 = df3.reset_index()
del df3['index']

# Graph Representation for GNN
# Create an edges array (sparse adjacency matrix) of shape [2, num_edges].
edges_arr = list(PG.edges)
edges = map(np.array, edges_arr)
edges = np.array(list(edges)).T
# Create an edge weights array of ones.
edge_weights = tf.convert_to_tensor(df3['Euclidean_Dist'])
edge_weights = tf.cast(edge_weights, tf.float32)
# Create a node features array of shape [num_nodes, num_features].
data_nparray = data.to_numpy()[:,2:9]
node_features = tf.cast(data_nparray, dtype=tf.float64)

# Create graph info
graph_info = (node_features, edges, edge_weights)

print("Edges shape:", edges.shape)
print("Nodes Features shape:", node_features.shape)

# Custom Graph CONV Layer

class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.4,
        normalize=True,
        *args,
        **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)
        self.normalize = normalize
        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_repesentations, node_indices, neighbour_messages):
        # Aggregatopm type: Sum
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # num_nodes = tf.math.reduce_max(node_indices) + 3
        num_nodes = len(node_repesentations)
        aggregated_message = tf.math.unsorted_segment_sum(neighbour_messages, node_indices, num_segments=num_nodes)
        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        # Concatenate the node_repesentations and aggregated_messages.
        h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(node_repesentations, node_indices, neighbour_messages)
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)

num_classes

# @title Classifier
class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        dropout_rate=0.4,
        *args,
        **kwargs,
    ):
        super(GNNNodeClassifier, self).__init__(*args, **kwargs)
        import tensorflow as tf
        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # Create a pre process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            name="graph_conv1",
        )
        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")

        # Create a compute logits layer.
        init = tf.keras.initializers.GlorotNormal(seed=123)
        self.compute_logits = layers.Dense(units=num_classes, kernel_initializer=init, bias_initializer = tf.zeros, name="logits")

    def call(self, input_node_indices):
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.node_features)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x1 + x
        # Postprocess node embedding.
        x = self.postprocess(x)
        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, tf.cast(input_node_indices, tf.int32))
        # Compute logits
        return self.compute_logits(node_embeddings)

class MyModel(tf.keras.Model):
    def __init__(self, base_model, lambda_= 0.001, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.base_model = base_model
        self.lambda_ = lambda_
        self.dense = tf.keras.layers.Dense(600)

    def call(self, inputs):
        x, A, D = inputs
        y_pred = self.base_model(x)

        # Adjust y_pred to have 600 units (Number of patients)
        y_pred = self.dense(y_pred)

        # Modify y_pred to have an extra dimension
        y_pred = tf.expand_dims(y_pred, -1)

        # Compute predictions at the vertices
        #y_pred_vertices = tf.matmul(A, y_pred)
        L = D - A
        y_pred_vertices = tf.matmul(L, y_pred)

        # Compute Laplacian loss as squared difference
        laplacian_loss = tf.reduce_mean(tf.square(y_pred - y_pred_vertices))

        self.add_loss(self.lambda_ * laplacian_loss)

        return y_pred


# Instantiate base model
base_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    name="gnn_model",
)

# Wrap it with custom model that adds the Laplacian regularization loss
gnn_model = MyModel(base_model)

# Compile the model
gnn_model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

# Create an early stopping callback.
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_acc", patience=1500, restore_best_weights=True)

x_train_RRIDs = train_data.RRID.to_numpy()

A_train = np.repeat(A[None, :], len(x_train_RRIDs), axis=0)
D_train = np.repeat(D[None, :], len(x_train_RRIDs), axis=0)

# Commented out IPython magic to ensure Python compatibility.
# # Fit the model.
# %%time
# history = gnn_model.fit(
#     x=[x_train_RRIDs, A_train, D_train],
#     y=y_train,
#     epochs=num_epochs,
#     batch_size=batch_size,
#     validation_split=0.15,
#     callbacks=[early_stopping]
# )
#

display_learning_curves(history)

facex_test = test_data.RRID.to_numpy()
x_test_RRIDs = test_data.RRID.to_numpy()
x_test_RRIDs
A_test = np.repeat(A[None, :], len(x_test_RRIDs), axis=0)
D_test = np.repeat(D[None, :], len(x_test_RRIDs), axis=0)
_, test_accuracy = gnn_model.evaluate(x=[x_test, A_test, D_test], y=y_test, verbose=1)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

x_test_RRIDs

test_inputs = [x_test, A_test, D_test]
predictions = np.argmax(gnn_model.predict(test_inputs), axis=1)
predictions1 = gnn_model.predict(test_inputs)

predictions

# Make individual predictions
node_id = 125
node_features = np.array([x_test[node_id]]) # Add an extra dimension to represent a batch size of 1
node_adjacency = np.array([A_test[node_id]])
node_degree = np.array([D_test[node_id]])

predictions_125 = np.argmax(gnn_model.predict([node_features, node_adjacency, node_degree]), axis=1)
predictions_125

real = np.array(y_test.values.tolist())

real = real.astype('int')
print(len(real))
print(real)

def compute_accuracy(y_true, y_pred):
    correct_predictions = 0
    # iterate over each label and check
    for true, predicted in zip(y_true, y_pred):
        if true == predicted:
            correct_predictions += 1
    # compute the accuracy
    accuracy = correct_predictions/len(y_true)
    return accuracy

compute_accuracy(real, predictions)

fig, ax = plt.subplots(figsize=(40,10))
# ax.scatter(np.arange(0, x_train.shape[0]), x_train)
# ax.scatter(np.arange(0, predictions.shape[0]), predictions, s=3)
ax.scatter(x_test, real, label = 'Real', s=30)
ax.scatter(x_test, predictions, label = 'Prediction', s=10)
ax.legend()

