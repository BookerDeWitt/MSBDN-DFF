% HR_patch LR_patch LR_Blur_patch
clear ;
close all;
folder ='/tmp1/home/Downloads/Datasets/RESIDE';
%folder = '/4TB/datasets/RESIDE/test';
save_root ='/tmp1/home/Downloads/Datasets/RESIDE/HDF5_RESIDE_0329'



%% scale factors
scale = 4;

size_label = 256;
size_input = size_label/scale;
stride = 128;

%% generate data
HR_image_path = fullfile(folder, 'rand_GT')
Blur_image_path = fullfile(folder ,'rand_hazy')
%HR_image_path = fullfile(folder, 'GT1')
%Blur_image_path = fullfile(folder ,'hazy1')
filepaths_HR = dir(fullfile(HR_image_path, '*.png')); 
filepaths_BLur = dir(fullfile(Blur_image_path, '*.png')); 
downsizes= [0.5,0.7,1];

%for n=1:10
for n=1:10 %n
data = zeros(size_label, size_label, 3, 1);
label = zeros(size_label, size_label, 3, 1);
count = 0;
margain = 0;       
%x= [1 1201 2401 3601 4801 6001 7201 8401 9601 10801]
%y = [1200 2400 3600 4800 6000 7200 8400 9600 10800 12000]
x= [1 1001 2001 3001 4001 5001 6001 7001 8001 9001];
y = [1000 2000 3000 4000 5000 6000 7000 8000 9000 10005];


for i = x(n) :y(n)
%for i = 1:length(filepaths_HR)
    for downsize = 1:length(downsizes)
                image = imread(fullfile(HR_image_path,filepaths_HR(i).name));
                image_Blur = imread(fullfile(Blur_image_path,filepaths_BLur(i).name));
                image = imresize(image,downsizes(downsize),'bicubic');
                image_Blur = imresize(image_Blur,downsizes(downsize),'bicubic');
                if size(image,3)==3
                    image = im2double(image);
                    image_Blur = im2double(image_Blur);
                    HR_label = modcrop(image, scale);
                    Blur_label = modcrop(image_Blur, scale);
                    [hei,wid, c] = size(HR_label);
                                      
                    filepaths_HR(i).name
                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            %Crop HR patch
                            HR_patch_label = HR_label(x : x+size_label-1, y : y+size_label-1, :);
                            [dx,dy] = gradient(HR_patch_label);
                            gradSum = sqrt(dx.^2 + dy.^2);
                            gradValue = mean(gradSum(:));
                            if gradValue < 0
                                continue;
                            end    
                            %Crop Blur patch
                            Blur_patch_label = Blur_label(x : x+size_label-1, y : y+size_label-1, :);
                         
                            % figure; imshow(HR_patch_label);
                               %figure; imshow(Blur_patch_label);
                              %figure; imshow( Blur_bic_patch_label_down);
                              %figure; imshow( Blur_bic_patch_label_up);

                            count=count+1;
                            data(:, :, :, count) = Blur_patch_label;
                            label(:, :, :, count) = HR_patch_label;
                        end % end of y 
                    end % end of x
                end % end of if
    end %end of downsize
end % end of i
%end % end of parts

order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order); 

%% writing to HDF5
savepath = fullfile(save_root ,sprintf('RESIDE_Part%d.h5', n))
%savepath = fullfile(save_root ,'GOPRO_test.h5')

chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    startloc = struct( 'dat',[1,1,1,totalct+1],  'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5_2out(savepath,  batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
end % index fo n