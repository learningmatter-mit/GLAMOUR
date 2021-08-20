### Details on hyperparameters for model training

CUSTOM_PARAMS : dictionary of custom hyperparameters, as detailed in descriptions below

Common to all models:
<br>
lr : Learning rate for updating model parameters (default = 0.0003, 0.02 for GCN)
<br>
weight_decay : Strength for L2 penalty in objective function (default = 0)
<br>
patience : Number of epochs to wait before early stopping when validation performance no longer gets improved
<br>
batch_size : Batch size for mini-batch training
<br>

Weave:
<br>
gnn_hidden_feats : Size for hidden node and edge representations (for GNN layers)
<br>
num_gnn_layers : Number of GNN (Weave) layers to use
<br>
graph_feats : Size for hidden graph representations (for MLP predictor)
<br>
gaussian_expand : Whether to expand each dimension of node features by gaussian histogram in computing graph representations
<br>

MPNN:
<br>
node_out_feats : Hidden size for node representations in GNN layers
<br>
edge_hidden_feats : Hidden size for edge representations in GNN layers
<br>
num_step_message_passing : Number of times for message passing, which is equivalent to number of GNN layers
<br>
num_step_set2set : Number of set2set steps
<br>
num_layer_set2set : Number of set2set layers
<br>

Attentive FP:
<br>
num_layers : Number of GNN layers
<br>
num_timesteps : Times of updating graph representations with GRU
<br>
graph_feat_size : Hidden size for graph representations
<br>
dropout : Dropout probability
<br>

GCN:
<br>
gnn_hidden_feats : Learning rate for updating model parameters
<br>
predictor_hidden_feats : Hidden size for MLP Predictor
<br>
num_gnn_layers : Hidden size for GNN layers
<br>
residual : Whether to use residual connection for each GCN layer
<br>
batchnorm : Whether to apply batch normalization to the output of each GCN layer
<br>
dropout : Dropout probability
<br>

GAT:
<br>
gnn_hidden_feats : Hidden size for each attention head in GNN layers
<br>
num_heads : Number of attention heads in each GNN layer
<br>
alpha : Slope for negative values in LeakyReLU
<br>
predictor_hidden_feats : Hidden size for the MLP Predictor
<br>
num_gnn_layers : Number of GNN layers to use
<br>
residual : Whether to use residual connection for each GAT layer
<br>
dropout: Dropout probability
