net.layers{1} = struct(...
    'name', 'conv1', ...
    'type', 'conv', ...
    'weights', {{randn(10,10,3,2,'single'), randn(2,1,'single')}}, ...,
    'dilate', 1, ...
    'pad', 0, ...
    'stride', 1, ...
    'opts', {}) ;
net.layers{2} = struct(...
    'name', 'relu1', ...
    'type', 'relu');

data = randn(300, 500, 3, 5, 'single') ;
res = vl_simplenn(net, data) ;

net = dagnn.DagNN();
convBlock = dagnn.Conv('size', [3 3 256 16], 'hasBias', true);
net.addLayer('conv1', convBlock, {'x1'}, {'x2'}, {'filters', 'biases'});
reluBlock = dagnn.ReLU();
net.addLayer('relu1', reluBlock, {'x2'}, {'x3'}, {});

net.initParams();

input = randn(10,15,256,1,'single');

%one must specify each input as a pair 'variableName',variableValue 
net.eval({'x1', input});

i = net.getVarIndex('x3');
output = net.vars(i).value;


%save network
netStruct = net.saveobj();
save('myfile.mat', '-struct', 'netStruct');
clear netStruct;

%load network
netStruct = load('myfile.mat');
net = dagnn.DagNN.loadobj(netStruct);
clear netStruct;