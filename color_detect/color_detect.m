clear all;
close all;
a = imread('color.png');
imshow(a);
% tach anh mau thanh 3 ma tran mau RGB
R = a(:,:,1);
G = a(:,:,2);
B = a(:,:,3);
% phan nguong theo muc xam o 3 kenh mau de giu lai phan mau can nhan dang
[m,n] = size(R);
for i = 1:m
    for j = 1:n
        if ((G(i,j) <40)&&(R(i,j) < 45)&&(B(i,j) > 120))%chon nguong cho mau xanh
            b(i,j) = 1;
        else
            b(i,j) = 0;
        end
    end
end
figure, imshow(b);
% dung phuong phap xoi mon de loai bo nhieu
se = strel('disk',3);
c = imerode(b,se);
figure, imshow(c);