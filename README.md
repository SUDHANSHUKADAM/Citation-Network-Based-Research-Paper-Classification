# Citation-Network-Based-Research-Paper-Classification

# Abstract
This project uses Graph Neural Networks (GNNs) to classify academic research publications inside a citation network. By leveraging the structural and feature-based information from the PubMed Diabetes dataset, we increase classification accuracy in comparison to traditional techniques. Our approach demonstrates how GNNs can handle high-dimensional feature spaces and capture interdependencies in graph-structured data. Experiments show how well GNNs learn from node attributes and interactions, opening the door for further advancements in graph-based machine learning applications.
Keywords- Graph Neural Networks (GNNs), Node Classification, Citation Networks, Feature Engineering, Feedforward Neural Networks (FNNs), Relational Data Modeling
# 1. INTRODUCTION
Graph Neural Networks (GNNs) are now a powerful tool for resolving complex problems with graphed data. These networks are designed to capitalize on the inherent organization of graph data, which consists of nodes (which represent objects) and edges (which show relationships). By gathering both local node characteristics and the connectivity patterns among them, GNNs can learn from non-Euclidean data with efficiency. This makes them perfect for a wide range of applications, including recommendation systems, social network analysis, and molecular biology. Graph Neural Networks will be used in this research to address the node classification issue. Node classification is the process of predicting a node's label or category in a network based on its features and the structure of the graph. This is a basic problem in many fields, including fraud detection in transaction networks, biological network protein classification, and social network user classification.[1]
# 1.1 BACKGROUND
Machine learning has been transformed by Graph Neural Networks (GNNs), which make it possible to analyze graph-structured data. Non-Euclidean relationships between entities (nodes) and their connections (edges) can be effectively captured by GNNs, in contrast to standard models. They are extensively utilized in fields including recommendation systems, molecular biology, and social networks.
# 1.2 PROBLEM
Classifying study papers is crucial for arranging scholarly material, enhancing information retrieval, and helping researchers find pertinent studies. Nevertheless, difficulties include:
- High-dimensional and diverse feature space.
- Interdependencies among research papers.
- Scalability and diversity in large datasets.
# 1.3 EXISTING LITERATURE
Working with graph-structured or non-Euclidean data presents substantial hurdles for traditional machine learning models, such as Feedforward Neural Networks (FNNs). The intricate relational dependencies between data points are not taken into consideration by these models since they are built to analyze data in a tabular or Euclidean format. In particular, FNNs ignore the structural links in the data, such as connections or interactions between entities in a network, and only consider the feature qualities of nodes. Because graph-based data is inherently connected and hierarchical, it poses special difficulties. Techniques like handmade feature extraction were used in early attempts to address graph-related challenges, but they frequently did not transfer well to bigger graphs or a variety of datasets. Because traditional models do not take advantage of the rich contextual information included in graph edges and nodes, relational topologies such as citation networks, social networks, and molecular graphs highlight these shortcomings. However, neighborhood aggregation approaches are introduced by Graph Neural Networks (GNNs) to address these constraints. In order to improve a node's representation, this method iteratively spreads and compiles data from its neighbors. GNNs capture both global and local structural patterns in graphs by combining relational dependencies and feature attributes. It has been shown that this feature greatly improves performance in applications including molecular property prediction, social network classification, and citation network analysis. GNNs fill the gap by offering more comprehensible representations of the underlying data in addition to increasing classification accuracy.
Additionally, the adaptability of graph-based machine learning has increased due to developments in GNN architectures, including Graph Neural Networks (GNNs), Graph Attention Networks (GATs), and message-passing neural networks (MPNNs). By overcoming the limitations of conventional neural network models, these techniques have established a basis for addressing issues that call for a complex comprehension of relationships within data.[2]
# 1.4 SYSTEM OVERVIEW
By making use of the structural connections between nodes (articles) and their attributes, the suggested system uses a multi-step method to categorize research papers inside a citation network. First, a baseline Feedforward Neural Network (FNN) model is built without any graph structure to assess classification performance using only node-level information, including binary TF-IDF vectors. The citation graph's complex dependencies and relationships are not captured by the baseline model, despite its respectable accuracy.
The system switches to a Graph Neural Network (GNN) model to overcome this constraint. In contrast to conventional models, GNNs are made especially to aggregate data from a node's neighbors to learn from graph-structured data. This enables every node to add contextual information from its connections to its feature representation. A paper's classification in a citation network, for example, benefits from the names and characteristics of the articles it cites or is cited by.
To manage missing data, create adjacency matrices, and normalize features, the system also has a strong data preprocessing pipeline. This guarantees that the models' input is clear and accurate representations of the structure of the graph. The system shows a notable increase in classification accuracy by fusing the benefits of feature-based learning (FNN) and graph-based learning (GNN). By avoiding overfitting, early pausing during training further improves efficiency. All things considered, the system demonstrates how graph-aware models, such as GNNs, perform better than conventional methods in encapsulating the intricacies of citation networks.
# 1.5 DATA COLLECTION
The Pubmed dataset is the primary dataset for this study. It contains 2,708 scientific articles displayed as nodes in a network, with edges indicating citations between papers. Each node is represented by a binary word vector that indicates the presence or absence of certain words from a given lexicon. This dataset provides a solid foundation for assessing the performance of GNNs in categorizing scientific writings.
The PubMed Diabetes dataset includes:
- Nodes: 19,717 publications with binary TF-IDF word vectors as features.
- Edges: 44,338 citation links forming a directed graph.
- Labels: Publications categorized into three classes:
- Diabetes Mellitus, Experimental
- Diabetes Mellitus Type 1
- Diabetes Mellitus Type 2
# 1.6 EXPERIMENTAL RESULTS
After evaluating the baseline Feedforward Neural Network (FNN) model, its performance was determined to be satisfactory but left room for improvement. The FNN achieved an accuracy of 75.43%, with a precision of 75.84%, recall of 75.43%, and an F1 score of 75.29%. While these metrics highlight the model's ability to learn from node-level features, the reliance solely on node attributes limited its capacity to exploit the relational structure inherent in the citation network.
To address this limitation, a Graph Neural Network (GNN) was implemented, leveraging the relational information within the graph. By incorporating the features of connected nodes and capturing neighborhood relationships, the GNN achieved a significant performance boost. The model's accuracy increased to 87.33%, precision to 87.33%, recall to 87.33%, and F1 score to 87.32%.
This marked improvement underscores GNN’s ability to outperform the baseline FNN by effectively utilizing the graph structure, demonstrating its superior capability for learning in tasks involving relational and structured data.
# 2. IMPORTANT DEFINATIONS
The project uses the PubMed Diabetes dataset, which is a structured graph that shows academic research papers and the relationships between their citations. The dataset offers both feature-based and structural information, making it possible to utilize sophisticated machine learning methods for classification tasks, such as Graph Neural Networks (GNNs). The essential elements are listed below.[9]
# 2.1 DATA
The Pubmed dataset is a collection of academic publications that spans seven different categories and includes 2,708 publications. Each publication is represented as a node in a graph, and the relationships between publications are represented by edges, denoting citations between articles.
- Node Features: Each node's features in the Pubmed dataset indicate whether 1,433 distinct phrases are present or absent in the publication. These terms are recorded as binary word vectors. These characteristics are used for classification and capture the content of every publication.
- Graph Structure: articles are represented by nodes, while citations between articles are represented by edges in the graph structure of the Pubmed dataset. Leveraging relationships between publications for classification tasks requires an understanding of the network structure.[9]
# 2.2 PREDCITION TARGET
Sorting scientific articles into predetermined groups according to their content and citation patterns is the main goal. Theory, Reinforcement Learning, Genetic Learning, Neural Networks, Probabilistic Methods, Case-Based, or Rule Learning are the seven categories to which each article is placed.
# 2.3 CONCEPTS IN DATA
- Nodes—in this context, academic publications like books, articles, or research papers—represent distinct entities within the collection. Each node is represented as a vertex in the citation network and relates to a distinct publication.
- Edges: The dataset's edges show the connections between nodes, particularly the citations between publications. There is a directed edge from Publication A to Publication B if Publication A refers Publication B.
- Graph connectedness: Determining the impact of one publication on another requires an understanding of the connectedness between publications as indicated by citation relationships.
- Publication Categories: Publications are categorized into seven specified categories, which make up the target variable. The classification problem is framed by these categories.
# 2.4 PROBLEM STATEMENT
The dataset is made up of academic publications that are represented as a citation network. Each publication is identified by binary TF-IDF word vectors that encode its text. Our goal is to create a categorization model that can reliably group articles into pre-established classes according to their citation linkages and content.
The following are some of the main limitations related to this issue:
- Disconnected Graph Components: The citation network can include disconnected components that make it more difficult for information to move across the graph and make it harder for the model to use global relationships to classify data accurately.
- Oversmoothing: Graph Neural Networks (GNNs) are susceptible to oversmoothing, especially when employing numerous layers. This happens when node representations get too similar, which lowers classification accuracy and discriminative capacity. To alleviate this problem and guarantee meaningful feature extraction, effective solutions are needed.
- Model Complexity: It's crucial to strike a balance between model complexity and performance. While simpler models may perform poorly, too complicated models may overfit. To guarantee generalization to unknown data, model architecture and hyperparameter optimization are crucial.
By addressing these challenges and incorporating advanced techniques like Graph Neural Networks (GNNs), the goal is to revolutionize the classification of scientific publications. This will enable more accurate categorization, improved knowledge organization, and enhanced accessibility of scholarly literature.
# 3. OVERVIEW OF THE PROPOSED SYSTEM
![image](https://github.com/user-attachments/assets/300804b3-8b08-4808-89a7-35ee42e5fc73)

Fig 3.1 DM Pipeline
A methodical methodology created to use and enhance the use of Graph Neural Networks (GNNs) for citation network analysis is the Data Mining Pipeline for the research paper classification system.
The first step is data collecting, which involves compiling a dataset of scholarly articles, where nodes stand for individual publications and edges for citation associations. For feature representation, the title and abstract of each paper are taken out. To handle missing or unnecessary features, lower noise, and design useful features, the data is cleaned during the preprocessing stage. The connection of the citation network is then captured by creating a graph with an adjacency matrix.
The baseline model building stage establishes a standard for comparisons by processing node characteristics for initial classification using a feedforward neural network. The integration of GNNs comes next, in which the model is improved to make use of graph structure via layer-wise propagation and neighbor aggregation, enabling nodes to learn from the network as a whole as well as from their neighbors.
Using node characteristics and the graph structure, the baseline and GNN models are trained during the training phase. Performance is tracked using measures like accuracy and early stopping to prevent overfitting.
The models are compared in the assessment and outcomes phase, which demonstrates that the GNN performs better than the baseline by successfully integrating graph relationships for increased classification accuracy. Future improvements to the pipeline include investigating more complex GNN architectures, such as Graph Attention Networks (GAT), and testing with bigger datasets to increase scalability. By methodically utilizing GNNs' advantages in processing graph-structured data, this iterative pipeline guarantees a reliable method for classifying research papers.[3]
# 4. OVERVIEW OF THE PROPOSED SYSTEM
# 4.1 OVERVIEW
This section details the technical implementation of the project, including:
- Preprocessing the PubMed dataset to prepare it for machine learning tasks.
- The baseline Feedforward Neural Network (FFN) model.
- The Graph Neural Network (GNN) model, designed to leverage both node-level features and graph structural information.
- Training methodologies, evaluation metrics, and model optimizations.
# 4.2 DATA PREPROCESSING
The data preprocessing pipeline ensures the PubMed dataset is structured and optimized for graph-based learning tasks. The steps undertaken are detailed below:
Feature Extraction
- Each paper is represented as a binary word vector of size 500, indicating the presence or absence of specific terms from the dataset's vocabulary.
- Stop words and terms with low document frequency (<10 occurrences) were removed to improve feature relevance and reduce dimensionality.
- A custom transformation function processed raw files (Pubmed-Diabetes. NODE.paper.tab) into structured feature matrices.

Normalization
- Node features were normalized to ensure numerical stability and to accelerate convergence during model training.
- This step also helps maintain uniform scaling across different feature dimensions, preventing any feature from dominating the learning process.

Graph Construction
- The citation relationships between papers were represented as a directed adjacency matrix, with each edge (source, target) indicating a citation.
- Edges were initialized with uniform weights of 1.0, assuming equal significance for all connections in the graph.
- The graph was constructed using data from PubmedDiabetes. DIRECTED.cites.tab, with paper IDs remapped to unique, zero-based indices for computational efficiency.

Target Variable Encoding:
- Papers were categorized into three classes: [1, 2, 3], representing different publication types or research areas.
- These categorical labels were encoded into zero-based indices to make them compatible with machine learning frameworks.
- The processed file contained features for each node and its corresponding label, saved in a structured format for further use.

Train-Test Split:
- To ensure a balanced evaluation, the dataset was split into training and testing subsets, stratified by class labels to maintain proportional representation of all categories.
- This stratification prevents class imbalance, which could skew the model's performance toward overrepresented classes.

Visualization:
A subset of the citation graph was visualized to validate preprocessing. Nodes were color-coded by their class labels, and edges represented citation relationships. The visualization confirmed the integrity of both the node and edge processing pipelines.
![image](https://github.com/user-attachments/assets/bfb469b7-e478-4df8-8b3f-7df2c5bf61b9)

Fig. 4.1 Illustrates the class distribution of papers in the PubMed dataset.
Each bar represents the count of papers belonging to a specific subject class (0, 1, or 2). This distribution reflects the stratified nature of the dataset, ensuring balanced representation of the classes for training and evaluation purposes.[9]
# 4.3 BASELINE MODEL: FEEDFORWARD NETWORK (FFN)
The FFN model served as a baseline to evaluate the added value of incorporating graph structure in the GNN. Its key components include:

Architecture:
- Input Layer: Processes the binary feature vector (500 dimensions for PubMed).
- Hidden Layers: Five fully connected layers.
- Each layer includes:
- Batch Normalization: Stabilizes and accelerates training.
- Dropout: Set to 50% to reduce overfitting.
- Activation Function: GELU for efficient gradient transitions.
- Skip Connections: Preserve gradient flow and prevent vanishing gradients.
- Output Layer: A dense layer with three units corresponding to the target classes, followed by softmax activation to produce probabilities.[4]

Training:
- Loss Function: Sparse Categorical Crossentropy to minimize classification error.
- Optimizer: Adam optimizer with a learning rate of 0.01.
- Early Stopping: Training stopped after 50 epochs without validation accuracy improvement.

Limitations:
The FFN model only uses node features, neglecting the valuable graph structure, which limits its performance in graph-based datasets.[5]
![image](https://github.com/user-attachments/assets/49902b6a-9620-4fe0-9a81-dde767634a88)

Fig. 4.2: Base Line model (FFN) Architecture
# 4.4 GRAPH NEURAL NETWORK
The GNN model was designed to leverage both node-level features and graph structural information for improved node classification. Key technical aspects include:
Input Representation:
- Node Features: Binary feature vectors (500 dimensions) representing each paper.
- Edges: Citation relationships between papers, represented as a sparse adjacency matrix.
- Edge Weights: Uniform weights were assigned to all edges.

Architecture:
- Preprocessing: Initial node embeddings were generated using a feedforward network.
- Graph Convolution Layers: Two layers were used to aggregate and update node embeddings.
- Aggregation: Features from neighboring nodes were aggregated using mean pooling.
- Update: Aggregated messages were combined with the node’s current embedding via concatenation.
- Skip Connections: Added to prevent oversmoothing, a common issue in GNNs where embeddings for all nodes converge to similar values.
- Postprocessing: Node embeddings were refined using a feedforward network.
- Output Layer: A dense layer mapped embeddings to class probabilities for classification.

Training:
- Loss Function: Sparse Categorical Crossentropy.
- Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.01.
- Regularization: Dropout (20%) was applied to reduce overfitting. L2 normalization ensured numerical stability.

Advantages:
The GNN outperformed the baseline FFN by utilizing graph relationships for feature aggregation and classification.
It captured contextual information by leveraging multi-hop connections.
![image](https://github.com/user-attachments/assets/5cd410cb-845f-489e-839b-3dbbb8a5a69f)

Fig. 4.3: GNN Model Architecture
# 4.5 CHALLENGES AND SOLUTIONS

Disconnected Components: Addressed by introducing skip connections and ensuring robust message passing across layers.

Oversmoothing: Mitigated by limiting the number of graph convolution layers and incorporating residual connections.

Class Imbalance:Stratified sampling ensured proportional representation of all classes during training and testing.
# 4.6 EVALUATION AND OBSERVATIONS
The models were evaluated on key metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. Significant performance improvements were observed in the GNN model compared to the FFN:
- Baseline Accuracy: 75.84%.
- GNN Accuracy: 87.33%.
GNN demonstrated superior class-wise precision and recall, especially for underrepresented classes.
# 5. EXPERIMENTS
# 5.1 DATASET DESCRIPTION
Researchers working in the domains of machine learning and medical literature analysis can benefit greatly from the PubMed dataset. It includes 19,717 research articles about diabetes that were taken from the PubMed database. The three different classes into which these papers are divided offer a structured dataset for a range of classification tasks.
The representation of publications in the PubMed dataset is one of its primary characteristics. A TF/IDF weighted word vector, which is generated from a vocabulary of 500 distinct words, is used to represent each scientific paper. This representation is appropriate for a variety of natural language processing jobs since it enables effective processing and analysis of the textual material.
The PubMed dataset has a citation network with 44,338 linkages in addition to the textual material. Researchers can investigate the connections between articles and conduct graph-based analyses thanks to this network structure, which gives the dataset an additional dimension. This dataset is especially adaptable for machine learning applications due to its network topology and linguistic content.
The PubMed dataset's structure and size make it perfect for a wide range of research applications. It offers a significant amount of data for machine learning model training and testing, with about 20,000 articles and more than 44,000 citation linkages. Because of its emphasis on diabetes research, the dataset is also a useful tool for medical professionals, enabling the creation of specific models and algorithms for the interpretation of medical literature.
# 5.2 EVALUATION METRICS
Assessment Criteria for the GNN Model. A range of metrics are used to assess a Graph Neural Network (GNN) model's performance to learn more about its generalization, accuracy, and resilience. The GNN model's performance on the classification job was evaluated using the following metrics:
- Accuracy
The percentage of correctly identified cases relative to the total number of examples is known as accuracy, and it is a key metric. It gives an unambiguous assessment of the model's overall performance. To make sure the model generalizes well beyond the training data, accuracy was computed for both the training and validation datasets in this evaluation.

- Loss
Loss functions measure the degree to which a model's predictions match the actual results. To evaluate how well the model is learning over time, training and validation loss values were tracked during training. A declining loss trend means that the model is getting better at making predictions.
Baseline Accuracy-Loss Curve
![image](https://github.com/user-attachments/assets/19d99f12-4637-436f-aa9f-eba44f630860)

Fig 5.1 Baseline Accuracy-Loss Curve
GNN Accuracy-Loss Curve
![image](https://github.com/user-attachments/assets/36a68370-d16f-4a06-b8c7-c53213bea3e5)

Fig 5.2 GNN Accuracy-Loss Curve
The Baseline model shows slower convergence and lower final accuracy (~80%) with more pronounced overfitting between train/test curves. In contrast, the GNN model demonstrates faster convergence, reaching higher accuracy (~90%) within just 5 epochs, and maintains more stable performance with minimal gap between training and validation curves, indicating better generalization.
- Training & Validation Loss Values
We can detect overfitting or underfitting problems by monitoring the training and validation loss levels. As training goes on, both losses should ideally drop and converge. Overfitting, in which the model performs well on training data but badly on unseen data, may be indicated by a substantial difference between training and validation loss.
- Confusion Matrix
Each class's true positives, false positives, true negatives, and false negatives are broken down in depth in the confusion matrix. This matrix aids in identifying the classes that are being incorrectly identified and can direct future model enhancements or class weighting modifications.

![image](https://github.com/user-attachments/assets/83989e9f-cacd-47ff-b186-6e306b3e7937)

Fig 5.3 Baseline Accuracy-Loss Matrix 
![image](https://github.com/user-attachments/assets/d92cf29b-0f86-4436-940e-b1e5f298b912)

Fig 5.4 GNN Accuracy-Loss Matrix
The GNN model shows improved classification performance with fewer misclassifications across all diabetes types, particularly reducing confusion between Type 2 and Experimental (422 vs 576 misclassifications). The diagonal elements are stronger in the GNN model (1789, 3354, 3539) compared to baseline (1782, 3141, 3132), indicating better overall classification accuracy and reduced false positives between classes.
- ROC Curve
A graphical representation known as the Receiver Operating Characteristic (ROC) curve shows how well a binary classifier system can diagnose problems as its discrimination threshold is changed. A one-vs-rest method can be used to plot a ROC curve for each class in multi-class classification jobs. The model's capacity to differentiate between classes is then summarized by calculating the Area Under the Curve (AUC).
![image](https://github.com/user-attachments/assets/c4a94438-81ec-4b4a-8f48-fe5ddf1cf021)

Fig 5.5 ROC for Baseline Model
![image](https://github.com/user-attachments/assets/69f55c09-3b59-49a0-abfa-f2a52e41467e)

Fig 5.6 ROC for Multi-Class Classification
The GNN model demonstrates superior performance with consistently higher AUC scores across all diabetes types (0.95-0.98) compared to the baseline model's varied performance (0.66-0.97), particularly showing dramatic improvement in Type 2 diabetes classification (AUC 0.95 vs 0.66). The GNN's ROC curves are more tightly clustered with higher micro and macro-average AUCs (0.97), indicating more balanced and robust performance across all classes compared to the baseline's scattered performance (micro=0.86, macro=0.84).

- Classification Report
The classification report is a thorough overview of a classification model’s performance that offers specific insights into the model’s performance in several classes. It contains a number of important metrics that aid in assessing the model’s support for every class, precision, recall, and F1-score. The following provides a more thorough description of each section of the classification report:
Classification Report Components
  - Precision
  -Recall (Sensitivity)
  -F1 Score
  - Support
![image](https://github.com/user-attachments/assets/59bd339a-8f64-4265-825f-b512abbede95)

Fig 5.7 Baseline Model Classification Report Heatmap
![image](https://github.com/user-attachments/assets/f7cd55f9-5e41-4de3-8c68-9a5b3db1efb1)

Fig 5.8 GNN Model Classification Report Heatmap
The GNN model shows consistently higher performance across all metrics, with improved precision (0.87-0.88), recall (0.85-0.89), and F1-scores (0.86-0.88) compared to the baseline model's lower and more varied metrics (precision: 0.79-0.84, recall: 0.79-0.88, F1: 0.80-0.85). The GNN model also achieves more balanced performance across all diabetes types, with macro averages around 0.87-0.88 versus the baseline's 0.82-0.83.
# 5.3 BASELINE AND ENHANCED MODELS

Baseline Model:

A feedforward neural network (FNN), which is intended to handle the publications' TF/IDF weighted word vectors, serves as the baseline model for the classification task of the PubMed dataset. Comparing this model to more sophisticated designs is a good place to start. The 500-dimensional TF/IDF word vectors are processed by the input layer of the architecture, which first eliminates irrelevant information like paper_id and subject. Five feedforward neural network (FFN) blocks are then included; each one consists of a dense layer and an activation function, most likely ReLU due to its efficacy in deep networks.
After every FFN block, skip connections are introduced, which enables the model to preserve outputs from earlier layers and avoid disappearing gradients. This design decision improves the network's information flow and makes it easier to train deeper designs. Logits for classification are produced by the output layer, which is a dense layer with neurons corresponding to the number of classes (in this example, three). [7]
Backpropagation and an optimizer such as Adam or SGD are used in the training process. In order to avoid overfitting, early stopping is used to track validation accuracy and halt training when gains reach a plateau. Although this baseline model can identify intricate patterns in the textual data, it is unable to make use of the PubMed dataset's citation network structure, which may result in the underutilization of important relationship information.

GNN Model:

Graph Neural Network (GNN) models, which take advantage of the rich structural information found in the citation network, offer an improved method for the classification challenge of the PubMed dataset. Because nodes in a graph represent entities (in this case, publications) and edges reflect relationships between them (citations), GNNs are specifically made to cope with graph-structured data. The model can learn from the publications' textual content as well as their relationships within the citation network thanks to this architecture.
Multiple GNN layers that carry out neighborhood aggregation let each node to learn from the properties of its neighbors in the GNN model used for PubMed categorization. Each publication's 500-dimensional TF/IDF word vectors serve as the initial node features, while the 44,338 citation linkages provide the edge information. By combining data from nearby nodes, each GNN layer modifies node attributes, allowing the model to identify more general patterns as nodes "see" beyond their immediate neighbors.
The potential of GNNs to facilitate contextual learning is one of its main benefits in this regard. Every publication gains knowledge from the context offered by cited and referencing papers in addition to its own characteristics. Because the model can learn global graph patterns, this leads to better feature representation, particularly for articles with few or no citations. Like the baseline model, the last layer of the GNN model is usually a classification layer that generates outputs for the three classes.
By simultaneously training the GNN model with node and edge characteristics, the model is able to backpropagate across the graph structure and learn intricate related patterns. Better generalization to unseen data, increased accuracy over the baseline FNN, and possibly more interpretable results are anticipated from this method since the learned node representations can shed light on the connections between various publications and research areas in the diabetes domain.[6]

Analysis

The GNN model significantly outperforms the Baseline model across all metrics:
- Accuracy: The GNN model achieves an accuracy of 87.33%, which is 11.90 percentage points higher than the Baseline model's 75.43%. This indicates that the GNN model correctly classifies a larger proportion of instances overall.
- Precision: With a precision of 87.33%, the GNN model shows a 11.49 percentage point improvement over the Baseline model's 75.84%. This suggests that when the GNN model predicts a positive class, it is more likely to be correct.
- Recall: The GNN model's recall of 87.33% is 11.90 percentage points higher than the Baseline model's 75.43%. This means the GNN model is better at identifying all relevant instances in the dataset.
- F1 Score: The F1 score, which is the harmonic mean of precision and recall, is 87.32% for the GNN model, compared to 75.29% for the Baseline model. This 12.03 percentage point difference indicates that the GNN model has a better balance between precision and recall.[8]
# 6. RELATED WORKS
Graph Neural Networks (GNNs) and conventional Feedforward Neural Networks (FFNs) have been used in several research to investigate node classification in citation networks. When used on datasets like Pubmed, a baseline FFN model that was trained only on node properties like text-based features or metadata performed mediocrely. A well-researched FFN implementation with skip connections, for instance, achieved a test accuracy of almost 81%. However, these models' capacity to depict intricate relationships between papers is limited because they frequently miss the structural information included in citation graphs. The potential of FFNs in graph-related tasks is limited by their incapacity to use edge-level information, such as bibliographic couplings or co-citation patterns.
On the other hand, initiatives such as CitGraph have started using GNNs to categorize nodes in citation networks by utilizing both graph topology and node attributes. The CitGraph project used the Planetoid dataset to create simple GNN topologies using tools such as PyTorch Geometric. Despite showing that GNNs are superior to FFNs in identifying relational patterns, CitGraph's performance was still restricted by issues such as a small selection of datasets and simple model setups. To fully utilize GNNs' potential for node categorization, these limitations emphasize the significance of sophisticated preprocessing, feature engineering, and architecture optimization.
Key insights from related works include:
- Feature Integration: While FFNs focus on isolated node attributes, GNNs integrate both attributes and relationships, resulting in more comprehensive learning.
- Dataset Limitations: Many existing studies rely on small, well-studied datasets like Cora and PubMed, limiting generalizability.
- Model Complexity: Basic GNN architectures often lack the sophistication needed to fully capture complex graph dynamics, underscoring the importance of architectural advancements.
# CONCLUSION
To solve classification challenges requiring both structured feature data and relational graph structures, this study explores the combined potential of feedforward neural networks (FNNs) and graph neural networks (GNNs). The GNN used graph topology to capture dependencies between connected nodes, whereas the FNN efficiently handled high-dimensional node properties. The findings of the experiment showed that GNNs performed better than the baseline FNN, achieving a higher accuracy (87.33%), proving the importance of using graph-based relational data in classification models.
The significance of tuning hyperparameters like learning rates, hidden units, and dropout rates to improve model performance and generalization is one of the main lessons learned. The GNN's capacity to represent relational patterns significantly increased precision and recall, whereas FNNs by themselves only demonstrated mediocre performance. Subsequent research endeavors may investigate sophisticated GNN structures such as Graph Attention Networks (GATs) and evaluate the model's scalability on more extensive and intricate datasets. These results highlight how combining feature-based and graph-based learning techniques can revolutionize categorization, recommendation, and prediction applications.
