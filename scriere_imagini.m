date = readtable('archive/fashion-mnist_test.csv');
for i=1:200
    label = date{i,1};
    imagine = date{i,2:end};
    imagine = reshape(imagine, 28, 28)';
    imagine = mat2gray(imagine);
    name = strcat(int2str(label),'/');
    name = strcat(name, int2str(i));
    name = strcat(name,'.png');
    imwrite(imagine, name);
end