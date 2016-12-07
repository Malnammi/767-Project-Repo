cd G:\767-Project\datasets\stare\vessel_segmentation\image
listing = dir('G:\767-Project\datasets\stare\vessel_segmentation\image');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [128 128]),strcat('G:\767-Project\datasets\stare\vessel_segmentation\images_128\', ...
        name, '.png'));
end

cd G:\767-Project\datasets\stare\vessel_segmentation\labels
listing = dir('G:\767-Project\datasets\stare\vessel_segmentation\labels');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [128 128]),strcat('G:\767-Project\datasets\stare\vessel_segmentation\labels_128\', ...
        name, '.png'));
end