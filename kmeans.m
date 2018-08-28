%% Kmeans Code by Shriraj (15CSE1036)
close all; clear; clc;

%% Input args (and print them)
%k = 3;     % number of clusters
%numP = 200; % number of points
%xMax = 100; % x between 0 and xMax
%yMax = 100; % y between 0 and yMax

%% Create random data points

% create a random matrix of size 2-by-numP 
% with row 1/2 (x/y coordiante) ranging in [minX,maxX]/[minY,maxY]
%xP = xMax * rand(1,numP);
%yP = yMax * rand(1,numP);
%points = [xP; yP];

ax = gca;
fig = ancestor(ax, 'figure');
points = [];
hold(ax, 'on');
while true
  c = ginput(1);
  sel = get(fig, 'SelectionType');
  if strcmpi(sel, 'alt'); break; end
  scatter(c(1),c(2));
  points=[points;c];
end
hold(ax, 'off')

points=points.';
numP = length(points);
xP = points(1,:);
yP = points(2,:);
M = max(points,[],2);
xMax = M(1);
yMax = M(2);

k=4;
fprintf('k-Means will run with %d clusters and %d data points.\n',k,numP);
%% run kMeans and measure/print performance
tic;
[cluster, centr] = kMeans(k, points); % my k-means
myPerform = toc;
fprintf('Computation time for kMeans.m: %d seconds.\n', myPerform);

%% All visualizations
figure('Name','Visualizations','units','normalized','outerposition',[0 0 1 1]);

% visualize the clustering
subplot(2,2,1);
scatter(xP,yP,200,cluster,'.');
hold on;
scatter(centr(1,:),centr(2,:),'xk','LineWidth',1.5);
axis([0 xMax 0 yMax]);
daspect([1 1 1]);
xlabel('x');
ylabel('y');
title('Random data points clustered (own implementation)');
grid on;

% number of points in each cluster
subplot(2,2,2);
histogram(cluster);
axis tight;
[num,~] = histcounts(cluster);
yticks(round(linspace(0,max(num),k)));
xlabel('Clusters');
ylabel('Number of data points');
title('Histogram of the cluster points (own implementation)');

%%KMEANS
function [ cluster, centr ] = kMeans( k, P )

%kMeans Clusters data points into k clusters.
%   Input args: k: number of clusters; 
%   points: m-by-n matrix of n m-dimensional data points.
%   Output args: cluster: 1-by-n array with values of 0,...,k-1
%   representing in which cluster the corresponding point lies in
%   centr: m-by-k matrix of the m-dimensional centroids of the k clusters


numP = size(P,2); % number of points
dimP = size(P,1); % dimension of points


%% Choose k data points as initial centroids

% choose k unique random indices between 1 and size(P,2) (number of points)
randIdx = randperm(numP,k);
% initial centroids
centr = P(:,randIdx);


%% Repeat until stopping criterion is met

% init cluster array
cluster = zeros(1,numP);

% init previous cluster array clusterPrev (for stopping criterion)
clusterPrev = cluster;

% for reference: count the iterations
iterations = 0;

% init stopping criterion
stop = false; % if stopping criterion met, it changes to true

while stop == false
    
    % for each data point 
    for idxP = 1:numP
        % init distance array dist
        dist = zeros(1,k);
        % compute distance to each centroid
        for idxC=1:k
            dist(idxC) = norm(P(:,idxP)-centr(:,idxC));
        end
        % find index of closest centroid (= find the cluster)
        [~, clusterP] = min(dist);
        cluster(idxP) = clusterP;
    end
    
    % Recompute centroids using current cluster memberships:
        
    % init centroid array centr
    centr = zeros(dimP,k);
    % for every cluster compute new centroid
    for idxC = 1:k
        % find the points in cluster number idxC and compute row-wise mean
        centr(:,idxC) = mean(P(:,cluster==idxC),2);
    end
    
    % Checking for stopping criterion: Clusters do not chnage anymore
    if clusterPrev==cluster
        stop = true;
    end
    % update previous cluster clusterPrev
    clusterPrev = cluster;
    
    iterations = iterations + 1;
    
end


% for reference: print number of iterations
fprintf('kMeans.m used %d iterations of changing centroids.\n',iterations);
end