%DEMO Demonstration of parametric t-SNE


    % Load MNIST dataset
    load 'mnist_train.mat'
    load 'mnist_test.mat'
    new_train_x = [];
    new_train_labels = [];
    new_test_x = [];
    new_test_labels = [];
    for label = 1:10
        idx = find(train_labels==label);
        len = length(idx);
        %rand = randi([1, len],[1,round(len/10)]);
        rand = randperm(len);
        rand = rand(1:round(len/10));
        new_train_x = [new_train_x; train_X(idx(rand),:)];
        new_train_labels = [new_train_labels; train_labels(idx(rand),:)]; 
    end
    for label = 1:10
        idx = find(test_labels==label);
        len = length(idx);
        %rand = randi([1, len],[1,round(len/10)]);
        rand = randperm(len);
        rand = rand(1:round(len/10));
        new_test_x = [new_test_x; test_X(idx(rand),:)];
        new_test_labels = [new_test_labels; test_labels(idx(rand),:)]; 
    end
    % Set perplexity and network structure
    perplexity = 30;
    layers = [500 500 2000 2];
    
    % Train the parametric t-SNE network
    %[network, err] = train_par_tsne(train_X, train_labels, test_X, test_labels, layers, 'CD1');
    [network, err] = train_par_tsne(new_train_x, new_train_labels, new_test_x, new_test_labels, layers, 'CD1');
    
    % Construct training and test embeddings
    mapped_train_X = run_data_through_network(network, train_X);
    mapped_test_X  = run_data_through_network(network, test_X);
    
    % Compute 1-NN error and trustworthiness
    disp(['1-NN error: ' num2str(knn_error(mapped_train_X, train_labels, mapped_test_X, test_labels, 1))]);
    disp(['Trustworthiness: ' num2str(trustworthiness(test_X, mapped_test_X, 12))]);
    
    % Plot test embedding
    scatter(mapped_test_X(:,1), mapped_test_X(:,2), 9, test_labels);
    title('Embedding of test data');