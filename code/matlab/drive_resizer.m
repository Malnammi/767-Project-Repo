listing = dir('.');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [512 512]),strcat('G:\767-Project\datasets\drive\DRIVE\test\images_512\', ...
        name, '.png'));
end

listing = dir('.');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [512 512]),strcat('G:\767-Project\datasets\drive\DRIVE\test\labels_512\', ...
        name, '.png'));
end

listing = dir('.');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [512 512]),strcat('G:\767-Project\datasets\drive\DRIVE\test\mask_512\', ...
        name, '.png'));
end