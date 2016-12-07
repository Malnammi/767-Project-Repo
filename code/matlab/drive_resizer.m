cd G:\767-Project\datasets\drive\DRIVE\training\images

listing = dir('G:\767-Project\datasets\drive\DRIVE\training\images');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [128 128]),strcat('G:\767-Project\datasets\drive\DRIVE\training\images_128\', ...
        name, '.png'));
end

cd G:\767-Project\datasets\drive\DRIVE\training\1st_manual
listing = dir('G:\767-Project\datasets\drive\DRIVE\training\1st_manual');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [128 128]),strcat('G:\767-Project\datasets\drive\DRIVE\training\labels_128\', ...
        name, '.png'));
end

cd G:\767-Project\datasets\drive\DRIVE\training\mask
listing = dir('G:\767-Project\datasets\drive\DRIVE\training\mask');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [128 128]),strcat('G:\767-Project\datasets\drive\DRIVE\training\mask_128\', ...
        name, '.png'));
end

%%%%%%%%%%%%%%%%%%%TEST SET%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd G:\767-Project\datasets\drive\DRIVE\test\images
listing = dir('G:\767-Project\datasets\drive\DRIVE\test\images');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [128 128]),strcat('G:\767-Project\datasets\drive\DRIVE\test\images_128\', ...
        name, '.png'));
end

cd G:\767-Project\datasets\drive\DRIVE\test\1st_manual
listing = dir('G:\767-Project\datasets\drive\DRIVE\test\1st_manual');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [128 128]),strcat('G:\767-Project\datasets\drive\DRIVE\test\labels_128\', ...
        name, '.png'));
end

cd G:\767-Project\datasets\drive\DRIVE\test\mask
listing = dir('G:\767-Project\datasets\drive\DRIVE\test\mask');
listing = listing(3:end);

for i=1:length(listing)
    name = listing(i).name;
    img = imread(name);
    ind = strfind(name, '.');
    name = name(1:ind-1);
    imwrite(imresize(img, [128 128]),strcat('G:\767-Project\datasets\drive\DRIVE\test\mask_128\', ...
        name, '.png'));
end