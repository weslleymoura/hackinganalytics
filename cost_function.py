# Create function to calculate the logspace
def genLogSpace( array_size, num ):
    lspace = np.around(np.logspace(0,np.log10(array_size),num)).astype(np.uint64)
    return np.array(sorted(set(lspace.tolist())))-1
ls=genLogSpace(100,10)
ls=np.delete(ls, 0)


#def calc_positive_precision(target, predictions):
#    df = pd.DataFrame({'target': target, 'predictions': predictions[:,0]})
#    y_pred = pd.Series([1 if i >= 0.5 else 0 for i in df['predictions']])
#    res = confusion_matrix(target, y_pred)
#    pos_precision = res[1][1] / (res[0][1] + res[1][1])
#    return pos_precision

#positive_precision = metrics.make_scorer(calc_positive_precision, greater_is_better=True, needs_proba=True, needs_threshold = False)


# This function aims to find a scenario with the best positive precision regarding the number of observations in the leaf nodes
# The more positive precision and the more number of observation in the leaf nodes, the best.
def calc_positive_precision_with_penalty(estimator, X, y_true):
    # Penalty rate
    k = 0.1
    
    # Internal variables
    sum_pos_precision_node_penalty = 0
    model = estimator.named_steps['model']
    
    # Compute predictions for the processing k-fold
    predictions = estimator.predict(X)
    
    # Get the leaf node index for each observation
    nodes = model.apply(X)
    
    # Orgnize previous computation in a DataFrame
    df = pd.DataFrame({'target': y_true, 'predictions': predictions, 'node': nodes})
    
    # Compute the number of unique leaf nodes
    unique_nodes = np.unique(df[['node']])
    
    # For each leaf node: compute positive precision, penalty and final score (positive precision - penalty)
    for n in unique_nodes:
        # Filter data from the processing node
        df_node = df[(df['node'] == n)]
        target_node = df[['target']]
        pred_node = df[['predictions']]
        
        # Compute node positive precision
        res = confusion_matrix(target_node, pred_node)
        pos_precision_node = res[1][1] / (res[0][1] + res[1][1])
        
        # Compute node penalty
        num_samples_leaf = df_node.shape[0]
        node_penalty = k / num_samples_leaf
        
        # Compute positive precision with penalty
        pos_precision_node_penalty = pos_precision_node - node_penalty
        sum_pos_precision_node_penalty = sum_pos_precision_node_penalty + pos_precision_node_penalty
    
    # Compute the average positive precision with penalty for all nodes from the k-fold
    return sum_pos_precision_node_penalty / (unique_nodes.size)


# Create pipeline
model = DecisionTreeClassifier(random_state = seed, presort = True, splitter = 'best')
estimators = []
estimators.append(('model', model))
gridModel = Pipeline(estimators)

# Grid search parameters
param_grid = {
    'model__min_samples_leaf' : ls
}

# Execute the grid search
CV_model = GridSearchCV(estimator=gridModel, param_grid=param_grid, scoring=calc_positive_precision_with_penalty, cv=kfold)
CV_model_result = CV_model.fit(X_train, Y_train)

# Print results
print("Best: %f using %s" % (CV_model_result.best_score_, CV_model_result.best_params_))
print(time.strftime("%H:%M:%S"))